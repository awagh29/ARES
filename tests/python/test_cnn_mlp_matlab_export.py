"""Run-ID selection, checkpoint provenance and native MATLAB export schema."""
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import numpy as np
from scipy.io import loadmat
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from cnn_mlp_inference import (
    PoseToRangeCNN, encode_sensor_state, describe_model_architecture, load_checkpoint_network,
)
from export_cnn_mlp_matlab import (
    export_episode, pack_model, resolve_training_run, sha256, verify_episode_integration,
)


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

    def test_resolver_selects_newest_matching_completed_run(self):
        newer = self.root / "results" / "newer"
        newer.mkdir()
        source = json.loads((self.run / "source.json").read_text())
        summary = json.loads((self.run / "summary.json").read_text())
        summary["created_at_utc"] = (
            datetime.fromisoformat(summary["created_at_utc"]) + timedelta(seconds=1)
        ).isoformat()
        (newer / "source.json").write_text(json.dumps(source))
        (newer / "summary.json").write_text(json.dumps(summary))
        (newer / "config.json").write_text((self.run / "config.json").read_text())
        (newer / "best.pt").write_bytes(b"resolver checks presence only")
        self.assertEqual(resolve_training_run(self.root / "results", self.episode_id), newer)
        self.assertEqual(resolve_training_run(self.root / "results", self.episode_id, self.run), self.run)

    def test_discovery_survives_deleted_runs_corrupt_metadata_and_other_models(self):
        root = self.root / "results"
        for name, fault in (("empty-summary", "empty"), ("nul-summary", "nul"),
                            ("missing-checkpoint", "missing"), ("empty-checkpoint", "zero"),
                            ("invalid-timestamp", "date"), ("unrelated-saits", "saits")):
            folder = root / name
            folder.mkdir()
            for filename in ("source.json", "summary.json", "config.json", "best.pt"):
                (folder / filename).write_bytes((self.run / filename).read_bytes())
            if fault == "empty":
                (folder / "summary.json").write_text("")
            elif fault == "nul":
                (folder / "summary.json").write_bytes(b"\x00" * 40)
            elif fault == "missing":
                (folder / "best.pt").unlink()
            elif fault == "zero":
                (folder / "best.pt").write_bytes(b"")
            elif fault == "date":
                summary = json.loads((folder / "summary.json").read_text())
                summary["created_at_utc"] = "invalid"
                (folder / "summary.json").write_text(json.dumps(summary))
            else:
                (folder / "config.json").write_text(json.dumps({"model": {"d_model": 128}}))
                (folder / "summary.json").write_text("not JSON")
        with warnings.catch_warnings(record=True) as skipped:
            warnings.simplefilter("always")
            self.assertEqual(resolve_training_run(root, self.episode_id), self.run)
        self.assertEqual(len(skipped), 5)  # SAITS is filtered before its summary is read.
        with self.assertRaisesRegex(ValueError, "Cannot use selected"):
            resolve_training_run(root, self.episode_id, root / "empty-summary")
        with self.assertRaisesRegex(ValueError, "not a PoseToRangeCNN"):
            resolve_training_run(root, self.episode_id, root / "unrelated-saits")
        with self.assertRaisesRegex(ValueError, "Cannot use selected"):
            resolve_training_run(root, self.episode_id, root / "deleted-run")

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

    def test_dropout_encoder_legacy_checkpoint_preserves_linear_latent(self):
        c = self.checkpoint["config"]["model"]
        widths = [12, *c["encoder_hidden"], c["latent_channels"] * c["latent_length"]]
        layers = []
        for i, (inputs, outputs) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(torch.nn.Linear(inputs, outputs))
            if i < len(widths) - 2:
                layers.extend([torch.nn.GELU(), torch.nn.Dropout(.1)])
        self.network.encoder = torch.nn.Sequential(*layers)
        self.network.eval()
        self.checkpoint["model_state_dict"] = self.network.state_dict()
        torch.save(self.checkpoint, self.run / "best.pt")
        rebuilt, _, profile = load_checkpoint_network(self.checkpoint)
        self.assertEqual(profile, "legacy-hidden-dropout-linear-latent")
        raw = torch.randn(3, 9)
        with torch.inference_mode():
            torch.testing.assert_close(rebuilt(encode_sensor_state(raw)),
                                       self.network(encode_sensor_state(raw)), rtol=0, atol=0)
        exported = export_episode(self.root, self.episode_id)
        self.assertEqual(exported["encoder_activations"], ["gelu", "gelu", "identity"])
        mat = loadmat(exported["model_path"], simplify_cells=True)
        for i, state in enumerate(mat["fixtures"]["rawStates"]):
            with torch.inference_mode():
                expected = self.network(encode_sensor_state(torch.from_numpy(state[None]))).numpy()[0]
            np.testing.assert_array_equal(expected, mat["fixtures"]["predictedRanges"][:, :, i])
        self.assertTrue((Path(exported["export_dir"]) / "architecture.json").is_file())

    def test_explicit_metadata_distinguishes_identically_named_layers(self):
        # Removing the last activation changes no parameter names. Metadata must
        # preserve that operation change instead of inferring it from the keys.
        self.network.encoder = torch.nn.Sequential(*list(self.network.encoder.children())[:-1])
        self.checkpoint["model_state_dict"] = self.network.state_dict()
        self.checkpoint["model_architecture"] = describe_model_architecture(self.network)
        rebuilt, description, profile = load_checkpoint_network(self.checkpoint)
        self.assertEqual(profile, "checkpoint-model-architecture")
        self.assertEqual(description["encoder"][-1]["type"], "Linear")
        with torch.inference_mode():
            states = encode_sensor_state(torch.randn(2, 9))
            torch.testing.assert_close(rebuilt(states), self.network(states), rtol=0, atol=0)
        self.checkpoint["model_state_dict"]["encoder.99.weight"] = torch.zeros(1)
        with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
            load_checkpoint_network(self.checkpoint)

    def test_read_only_integration_verification_checks_hashes_and_scan_logs(self):
        exported = export_episode(self.root, self.episode_id)
        initial = verify_episode_integration(self.root, self.episode_id)
        self.assertTrue(initial["deployment_matches_latest"])
        self.assertFalse(initial["latest_capture_used_latest"])
        self.assertFalse(initial["matlab_parity_verified"])
        (Path(exported["export_dir"]) / "matlab_validation.json").write_text(json.dumps({
            "allParityChecksPassed": True, "modelSha256": exported["model_mat_sha256"],
        }))
        capture = self.root / "data/cnn_single_episode_intermittent/capture-fixture"
        capture.mkdir(parents=True)
        (capture / "manifest.json").write_text(json.dumps({
            "runId": "capture-fixture", "createdAtUtc": datetime.now(timezone.utc).isoformat(),
            "inference": {"sourceEpisodeId": self.episode_id,
                "trainingRunDirectory": str(self.run),
                "sourceCheckpointSha256": exported["checkpoint_sha256"],
                "artifactSha256": exported["model_mat_sha256"],
                "runtimeProof": {"scanCount": 3, "reconstructionUsedScanCount": 2,
                    "controllerInfluencedScanCount": 1, "instrumentationChecksPassed": True}},
        }))
        (capture / "summary.json").write_text("{}")
        csv_path = capture / "scans.csv"
        csv_path.write_text("CnnReconstructionUsed,ControllerInfluencedDuringOutage,LidarMeasuredAvailable\n"
                            "0,0,1\n1,1,0\n1,0,0\n")
        report = verify_episode_integration(self.root, self.episode_id)
        self.assertTrue(report["latest_capture_used_latest"])
        self.assertTrue(report["capture_logging_verified"])
        self.assertTrue(report["matlab_parity_verified"])
        original_csv = csv_path.read_text()
        csv_path.write_text(original_csv.replace("1,1,0", "0,0,0"))
        self.assertFalse(verify_episode_integration(self.root, self.episode_id)["capture_logging_verified"])
        csv_path.write_text(original_csv)
        with (self.run / "best.pt").open("ab") as stream:
            stream.write(b"changed checkpoint")
        stale = verify_episode_integration(self.root, self.episode_id)
        self.assertFalse(stale["deployment_matches_latest"])
        self.assertFalse(stale["latest_capture_used_latest"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
