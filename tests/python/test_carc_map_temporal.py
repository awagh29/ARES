"""Clone-relative paths, raw MATLAB parity, and the ordered cluster launcher."""
import ast
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scripts/carc"))
import prepare_map_temporal_data as prepare_data
import run_map_temporal as launcher


def path_functions():
    nb = json.loads((ROOT/"notebooks/train_sparse_reconstruction_map_temporal.ipynb").read_text(encoding="utf-8"))
    tree = ast.parse("".join(nb["cells"][1]["source"]))
    definitions = ast.Module(body=[item for item in tree.body if isinstance(item, ast.FunctionDef)
                                  and item.name in ("find_repo_root", "resolve_export_paths")], type_ignores=[])
    namespace = dict(Path=Path, os=os)
    exec(compile(definitions, "<notebook-paths>", "exec"), namespace)
    return namespace


class CarcPathTests(unittest.TestCase):
    def test_detects_checkout_from_notebooks_and_environment(self):
        functions = path_functions()
        with tempfile.TemporaryDirectory(dir=ROOT/"tmp") as folder, patch.dict(os.environ, {}, clear=True):
            checkout = Path(folder)/"cloned elsewhere"
            notebooks = checkout/"notebooks"
            notebooks.mkdir(parents=True)
            (notebooks/"train_sparse_reconstruction_map_temporal.ipynb").touch()
            self.assertEqual(functions["find_repo_root"](notebooks), checkout.resolve())
            os.environ["ARES_REPO_ROOT"] = str(checkout)
            self.assertEqual(functions["find_repo_root"](ROOT), checkout.resolve())
            os.environ["ARES_REPO_ROOT"] = str(checkout/"missing")
            with self.assertRaisesRegex(FileNotFoundError, "Cannot locate"):
                functions["find_repo_root"]()

    def test_prepared_exports_and_explicit_overrides(self):
        functions = path_functions()
        with tempfile.TemporaryDirectory(dir=ROOT/"tmp") as folder, patch.dict(os.environ, {}, clear=True):
            root, run_root = Path(folder)/"checkout", Path(folder)/"generated"
            config = dict(paths=dict(output_root=str(run_root/"results/model")), data=dict(batch_id="batch"))
            episodes, audit = functions["resolve_export_paths"](root, config)
            self.assertEqual(episodes, root/"data/processed/lidar_episode_v1")
            prepared = run_root/"exports/batch"
            prepared.mkdir(parents=True)
            (prepared/"python_export_qa.json").touch()
            episodes, audit = functions["resolve_export_paths"](root, config)
            self.assertEqual(audit, prepared/"python_export_qa.json")
            self.assertEqual(episodes, prepared/"lidar_episode_v1")
            os.environ["ARES_EPISODE_ROOT"] = str(root/"custom")
            with self.assertRaisesRegex(ValueError, "both"):
                functions["resolve_export_paths"](root, config)
            os.environ["ARES_EXPORT_AUDIT"] = str(root/"custom-report.json")
            self.assertEqual(functions["resolve_export_paths"](root, config), (root/"custom", root/"custom-report.json"))

    def test_launcher_preserves_order_settings_and_failure_logs(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tmp") as folder, patch.dict(os.environ, {}, clear=True):
            root = Path(folder)
            path = root/"fixture.ipynb"
            cells = [dict(cell_type="markdown", source=["Text"]), dict(cell_type="code", source=['''
CONFIG = {"training": {"epochs": 1000, "batch_size": 16}, "scheduler": {"t_max_epochs": 2000},
          "runtime": {"amp_dtype": "bf16"}, "data": {"batch_id": "fixture"}}
def validate_config(config): pass
trace = [1]
''']), dict(cell_type="code", source=["trace.append(2)\nassert trace == [1, 2]\nassert CONFIG['training']['epochs'] == 1000\n"])]
            path.write_text(json.dumps(dict(cells=cells)))
            args = launcher.parse_args(["--repo-root", str(root), "--run-root", str(root/"generated"),
                                       "--notebook", str(path), "--allow-local"])
            fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
            with patch.dict(sys.modules, {"torch": fake_torch}):
                launch = launcher.run_notebook(args)
            effective = json.loads((launch/"effective_config.json").read_text())
            self.assertEqual(effective["training"]["epochs"], 1000)
            self.assertEqual(effective["runtime"]["amp_dtype"], "fp16")
            self.assertEqual(json.loads((launch/"status.json").read_text())["status"], "completed")
            self.assertEqual(path.read_bytes(), (launch/"input_notebook.ipynb").read_bytes())
            cells[-1]["source"] = ["raise RuntimeError('intentional fixture failure')"]
            path.write_text(json.dumps(dict(cells=cells)))
            with patch.dict(sys.modules, {"torch": fake_torch}), self.assertRaisesRegex(RuntimeError, "fixture failure"):
                launcher.run_notebook(args)
            failed = [p for p in (root/"generated/launches").iterdir() if p != launch][0]
            self.assertEqual(json.loads((failed/"status.json").read_text())["status"], "failed")
            self.assertIn("intentional fixture failure", (failed/"console.log").read_text())


@unittest.skipUnless(importlib.util.find_spec("h5py"), "h5py needed for raw MATLAB conversion")
class RawConversionTests(unittest.TestCase):
    def test_actual_raw_numeric_parity_with_original_matlab_exports(self):
        batch = ROOT/"data/batches"/prepare_data.DEFAULT_BATCH
        with (batch/"index.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in (rows[0], rows[40]):  # one flight from each start axis
            raw = ROOT/"data/raw"/row["RunId"]
            arrays = prepare_data.convert_raw(next(raw.glob("*_telemetry.mat")), prepare_data.read_json(raw/"manifest.json"), row)
            with np.load(ROOT/"data/processed/lidar_episode_v1"/row["RunId"]/"episode.npz", allow_pickle=False) as original:
                for key, value in arrays.items():
                    if value.dtype.kind in "fiu":
                        np.testing.assert_allclose(value, original[key], rtol=1e-12, atol=1e-12, equal_nan=True, err_msg=key)
                    elif value.dtype.kind == "b":
                        np.testing.assert_array_equal(value, original[key], err_msg=key)

    def test_duplicate_last_angle_interpolation_and_no_extrapolation(self):
        import h5py
        with tempfile.TemporaryDirectory(dir=ROOT/"tmp") as folder:
            with h5py.File(Path(folder)/"signals.mat", "w") as f:
                signal = f.create_group("angle")
                signal["time_s"] = [[0., 1., 1., 2.]]
                signal["values"] = [[3.0, 123., -3.0, -2.8]]
                output = prepare_data.sample_signal(f, "angle", np.array([0., .5, 1., 2.]), 1, "angular")
                self.assertAlmostEqual(abs(output[1, 0]), np.pi, places=10)
                self.assertAlmostEqual(output[2, 0], -3.)
                with self.assertRaisesRegex(ValueError, "no extrapolation"):
                    prepare_data.sample_signal(f, "angle", np.array([-.1, 1.]), 1)

    def test_immutable_save_and_export_hash_rebase(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tmp") as folder:
            root = Path(folder)
            path = root/"run/episode.npz"
            arrays = {"test": np.array([1., np.nan])}
            prepare_data.save_episode(path, arrays)
            prepare_data.save_episode(path, arrays)
            with self.assertRaisesRegex(ValueError, "differs"):
                prepare_data.save_episode(path, {"test": np.array([2., 3.])})
            audit = root/"audit.json"
            audit.write_text(json.dumps(dict(allExportChecksPassed=True, partialTestExport=False,
                files=[dict(runId="run", npzSha256=prepare_data.sha256(path), npzPath=r"Z:\old\episode.npz")])) )
            self.assertTrue(prepare_data.check_exports(root, audit, expected_runs=1))
            with path.open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                prepare_data.check_exports(root, audit, expected_runs=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
