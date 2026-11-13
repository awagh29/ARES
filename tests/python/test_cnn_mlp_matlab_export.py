"""Run-ID selection, checkpoint provenance and native MATLAB export schema."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.io import loadmat
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from cnn_mlp_inference import PoseToRangeCNN, encode_sensor_state
from export_cnn_mlp_matlab import export_episode, pack_model, resolve_training_run, sha256


class TestCnnMatlabExport(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.episode_id = "fixture-episode"
        self.run = self.root / "results" / "single"
        self.run.mkdir(parents=True)
        config = {"model": dict(
            input_features=12, hidden_activation="gelu", encoder_hidden=[8, 16],
            latent_channels=4, latent_length=45,
            decoder_channels=[8, 8, 4, 4, 15], decoder_lengths=[90, 179, 358, 716, 716],
            kernel_sizes=[5, 5, 3, 3, 3], upsample_mode="linear", align_corners=False,
            output_activation="gelu",
        )}
        self.network = PoseToRangeCNN(config["model"]).eval()
        raw = self.root / "data/raw" / self.episode_id / "manifest.json"
        raw.parent.mkdir(parents=True)
        raw.write_text(json.dumps({"runId": self.episode_id, "configuration": {
            "mission": {"initialPositionNed": [1, 0, -7]},
        }}))
        npz = self.root / "data/processed/lidar_episode_v1" / self.episode_id / "episode.npz"
        npz.parent.mkdir(parents=True)
        np.savez_compressed(npz, position_ned_m=np.ones((5, 3), np.float32),
                            velocity_ned_mps=np.zeros((5, 3), np.float32),
                            attitude_ned_rpy_rad=np.zeros((5, 3), np.float32))
        self.source = dict(episode_id=self.episode_id, manifest_sha256=sha256(raw),
                           npz_sha256=sha256(npz),
                           split_ranges_zero_based_end_exclusive={"validation": [4, 5]})
        self.checkpoint = dict(config=config, source_metadata=self.source,
                               epoch=2, best_epoch=2, model_state_dict=self.network.state_dict())
        torch.save(self.checkpoint, self.run / "best.pt")
        for name, value in (
            ("source.json", self.source), ("config.json", config),
            ("summary.json", dict(episode_id=self.episode_id, status="completed",
                                  best_epoch=2, best_val_mse=1.2,
                                  created_at_utc=datetime.now(timezone.utc).isoformat())),
        ):
            (self.run / name).write_text(json.dumps(value))

    def test_resolver_ignores_wrong_episode_and_unfinished_runs(self):
        for name, episode_id, status in (("other", "another-episode", "completed"),
                                          ("unfinished", self.episode_id, "running")):
            folder = self.root / "results" / name
            folder.mkdir()
            (folder / "source.json").write_text(json.dumps(dict(episode_id=episode_id)))
            (folder / "summary.json").write_text(json.dumps(dict(episode_id=episode_id, status=status)))
        self.assertEqual(resolve_training_run(self.root / "results", self.episode_id), self.run)
        with self.assertRaisesRegex(ValueError, "another episode"):
            resolve_training_run(self.root / "results", self.episode_id, self.root / "results/other")
        with self.assertRaisesRegex(ValueError, "No completed"):
            resolve_training_run(self.root / "results", "missing-episode")

    def test_mat_export_reference_fixtures_and_immutable_weights(self):
        before = sha256(self.run / "best.pt")
        exported = export_episode(self.root, self.episode_id)
        self.assertEqual(sha256(self.run / "best.pt"), before)
        mat = loadmat(exported["model_path"], simplify_cells=True)
        packed, fixtures = mat["model"], mat["fixtures"]
        self.assertEqual(packed["episodeId"], self.episode_id)
        np.testing.assert_array_equal(packed["encoder"][0]["W"], self.network.encoder[0].weight.detach().numpy())
        self.assertEqual(fixtures["predictedRanges"].shape[:2], (15, 716))
        with torch.inference_mode():
            for i, state in enumerate(fixtures["rawStates"]):
                encoded = encode_sensor_state(torch.from_numpy(state[None]))
                np.testing.assert_array_equal(encoded.numpy()[0], fixtures["encodedStates"][i])
                np.testing.assert_allclose(self.network(encoded).numpy()[0], fixtures["predictedRanges"][:, :, i], atol=1e-6)
        again = export_episode(self.root, self.episode_id)
        self.assertEqual(exported["model_path"], again["model_path"])
        self.assertEqual(exported["model_mat_sha256"], sha256(again["model_path"]))
        pointer = self.root / "data/models/cnn_mlp_single_episode" / self.episode_id / "deployment.json"
        self.assertEqual(json.loads(pointer.read_text())["checkpoint_path"], str(self.run / "best.pt"))

    def test_rejects_wrong_checkpoint_episode_and_changed_raw_manifest(self):
        bad = copy.deepcopy(self.checkpoint)
        bad["source_metadata"]["episode_id"] = "another-episode"
        torch.save(bad, self.run / "best.pt")
        with self.assertRaisesRegex(ValueError, "provenance"):
            export_episode(self.root, self.episode_id)
        torch.save(self.checkpoint, self.run / "best.pt")
        raw = self.root / "data/raw" / self.episode_id / "manifest.json"
        raw.write_text(raw.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            export_episode(self.root, self.episode_id)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
