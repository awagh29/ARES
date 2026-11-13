"""Portable-weight, preprocessing, reference-vector and immutability checks."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("torch", "scipy"))
if AVAILABLE:
    import torch
    from scipy.io import loadmat
    from export_saits_matlab import (
        TRAINING_FORMAT,
        build_fixtures,
        export_checkpoint,
        load_checkpoint,
        normalize_features,
        pack_model,
        source_sha256,
    )
    from saits_endpoint_model import SAITSEndpointModel


@unittest.skipUnless(AVAILABLE, "PyTorch and SciPy are required for MATLAB export checks.")
class TestSaitsMatlabExport(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.old_threads)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        torch.manual_seed(19)
        kwargs = dict(n_steps=50, n_features=31, n_targets=14, d_model=16,
                      n_heads=2, d_ff=32, n_layers=2, dropout=0.1)
        self.model = SAITSEndpointModel(**kwargs).eval()
        names = ([f"sector_{index:02d}_distance_m" for index in range(7)]
                 + [f"sector_{index:02d}_return_fraction" for index in range(7)])
        transform = {
            "feature_names": names + [f"context_{index}" for index in range(17)],
            "target_columns": names,
            "context_mean": np.linspace(-0.321234567, 0.912345678, 17).tolist(),
            "context_std": np.linspace(0.0131234567, 1.512345678, 17).tolist(),
            "distance_scale": 7.0,
        }
        state = self.model.state_dict()
        self.checkpoint = {
            "format_version": TRAINING_FORMAT, "epoch": 7, "best_epoch": 7,
            "model_kwargs": kwargs, "transform": transform,
            "model_state_dict": state, "best_state_dict": state,
            "source_signature": {"saits_endpoint_model.py": source_sha256(SCRIPTS / "saits_endpoint_model.py")},
            "device_type": "cuda", "config": {},
        }
        self.path = self.directory / "best.pt"
        torch.save(self.checkpoint, self.path)

    def test_pack_preserves_every_linear_and_normalization_parameter(self):
        packed = pack_model(self.checkpoint)
        state = self.model.state_dict()
        for exported, original in (
            ("embeddingFirst", "embedding_first"),
            ("embeddingSecond", "embedding_second"),
            ("reduceFirst", "reduce_first"),
            ("reduceSecondBeta", "reduce_second_beta"),
            ("reduceSecondGamma", "reduce_second_gamma"),
            ("combinationGate", "combination_gate"),
        ):
            np.testing.assert_array_equal(packed[exported]["W"], state[original + ".weight"].numpy().T)
            np.testing.assert_array_equal(packed[exported]["b"].ravel(), state[original + ".bias"].numpy())
        for exported, original in (("firstBlock", "first_block"), ("secondBlock", "second_block")):
            self.assertEqual(packed[exported].shape, (1, 2))
            for index, layer in enumerate(packed[exported].flat):
                for name in ("query", "key", "value", "output"):
                    np.testing.assert_array_equal(layer[name + "W"],
                        state[f"{original}.{index}.{name}.weight"].numpy().T)
                self.assertEqual(layer["attentionNorm"]["epsilon"], 1e-5)
                self.assertEqual(layer["ffNorm"]["epsilon"], 1e-6)
        self.assertEqual(packed["contextMean"].dtype, np.float64)
        self.assertEqual(packed["contextStd"].dtype, np.float64)

    def test_preprocessing_cast_order_and_hidden_placeholders(self):
        transform = self.checkpoint["transform"]
        raw = np.linspace(-2, 8, 50 * 31, dtype=np.float64).reshape(50, 31)
        mask = np.ones_like(raw, dtype=bool)
        mask[-10:, :14] = False
        expected = raw.astype(np.float32)
        expected[:, :7] /= np.float32(7)
        expected[:, 14:] = ((expected[:, 14:] - np.asarray(transform["context_mean"]))
                            / np.asarray(transform["context_std"]))
        expected[~mask] = 0
        actual, actual_mask = normalize_features(raw, mask, transform)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual_mask, mask)
        corrupted = raw.copy()
        corrupted[~mask] = 1e6
        np.testing.assert_array_equal(normalize_features(corrupted, mask, transform)[0], actual)
        corrupted[~mask] = np.nan
        np.testing.assert_array_equal(normalize_features(corrupted, mask, transform)[0], actual)
        corrupted[0, 15] = np.nan
        with self.assertRaisesRegex(ValueError, "Observed inputs must be finite"):
            normalize_features(corrupted, mask, transform)

    def test_reference_predictions_are_batch_one_and_physical_units(self):
        fixtures = build_fixtures(self.model, self.checkpoint["transform"])
        self.assertEqual(fixtures["rawFeatures"].shape, (50, 31, 5))
        with torch.inference_mode():
            for index in range(5):
                x = np.ascontiguousarray(fixtures["normalizedValues"][:, :, index][None])
                mask = np.ascontiguousarray(fixtures["observedMask"][:, :, index][None])
                result = self.model(torch.from_numpy(x), torch.from_numpy(mask), return_attention=True)
                np.testing.assert_array_equal(fixtures["predictionNormalized"][index], result["prediction"].numpy()[0])
                expected = result["prediction"].numpy()[0] * np.asarray([7] * 7 + [1] * 7, dtype=np.float32)
                np.testing.assert_array_equal(fixtures["predictionPhysical"][index], expected)
                np.testing.assert_array_equal(fixtures["attention"][:, :, index], result["attention"].numpy()[0])

    def test_native_mat_roundtrip_checkpoint_unchanged_and_export_immutable(self):
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        output = export_checkpoint(self.path, self.directory / "export")
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), before)
        artifact = loadmat(output / "model.mat", struct_as_record=False, squeeze_me=False, mat_dtype=True)
        packed = artifact["model"][0, 0]
        self.assertEqual(packed.firstBlock.shape, (1, 2))
        self.assertEqual(packed.firstBlock[0, 0][0, 0].queryW.dtype, np.float32)
        self.assertEqual(packed.contextMean.dtype, np.float64)
        manifest = json.loads((output / "export_manifest.json").read_text())
        self.assertFalse(manifest["weights_modified"])
        self.assertFalse(manifest["matlab_parity_verified"])
        self.assertEqual(manifest["checkpoint_sha256"], before)
        mat_before = hashlib.sha256((output / "model.mat").read_bytes()).hexdigest()
        with self.assertRaises(FileExistsError):
            export_checkpoint(self.path, output)
        self.assertEqual(hashlib.sha256((output / "model.mat").read_bytes()).hexdigest(), mat_before)

    def test_wrong_source_or_nonbest_checkpoint_is_rejected(self):
        bad = dict(self.checkpoint)
        bad["source_signature"] = {"saits_endpoint_model.py": "bad"}
        torch.save(bad, self.path)
        with self.assertRaisesRegex(ValueError, "Model source differs"):
            load_checkpoint(self.path)
        bad = dict(self.checkpoint)
        bad["epoch"] = 9
        torch.save(bad, self.path)
        with self.assertRaisesRegex(ValueError, "Select best.pt"):
            load_checkpoint(self.path)


if __name__ == "__main__":
    unittest.main()
