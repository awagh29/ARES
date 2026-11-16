"""Sparse checkpoint discovery, notebook parity and missing-scan semantics."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from export_sparse_reconstruction_matlab import architecture_fingerprint, resolve_training_run, validate_model_config
from sparse_reconstruction_inference import SparseRangeReconstructor, encode_sensor_state


class SparseExportTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(input_features=9, max_range=7., elevation_vectors=15, azimuth_vectors=716,
                           azimuth_degree=[-179., 179.], widths=[2, 4], use_position=False, kernel=3, state_hidden=4)
        torch.set_num_threads(1)

    def test_notebook_definitions_match_exported_architecture(self):
        notebook = ROOT / 'notebooks/train_sparse_reconstruction_single_episode_baseline.ipynb'
        self.assertEqual(len(architecture_fingerprint(notebook)), 64)
        changed = json.loads(notebook.read_text(encoding='utf-8'))
        for cell in changed['cells']:
            source = ''.join(cell['source'])
            if 'return self.max_range * torch.sigmoid' in source:
                cell['source'] = source.replace('torch.sigmoid(logits.squeeze(1))', 'torch.relu(logits.squeeze(1))')
        with tempfile.TemporaryDirectory(dir=ROOT / 'results') as directory:
            path = Path(directory) / 'changed.ipynb'
            path.write_text(json.dumps(changed), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'architecture changed'):
                architecture_fingerprint(path)

    def test_encoder_state_order_and_position_switch(self):
        raw = torch.tensor([[10., 20., -7., .1, .2, .3, .4, .5, .6]])
        angles = torch.stack((raw[:, 6:].sin(), raw[:, 6:].cos()), dim=-1).flatten(1)
        self.assertTrue(torch.equal(encode_sensor_state(raw, False), torch.cat((raw[:, 3:6], angles), 1)))
        self.assertTrue(torch.equal(encode_sensor_state(raw, True), torch.cat((raw[:, :6], angles), 1)))
        bad = copy.deepcopy(self.config); bad['input_features'] = 12
        with self.assertRaisesRegex(ValueError, 'position setting'):
            validate_model_config(bad)

    def test_full_outage_cannot_read_hidden_ranges_or_hit_masks(self):
        model = SparseRangeReconstructor(self.config).eval()
        available = torch.zeros(1, 15, 716, dtype=torch.bool)
        state = torch.randn(1, 9)
        with torch.inference_mode():
            first = model(torch.zeros(1, 15, 716), available, available, state)
            second = model(torch.rand(1, 15, 716) * 7, ~available, available, state)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(((first >= 0) & (first <= 7)).all())

    def test_discovery_filters_other_models_and_skips_corrupt_json(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'results') as directory:
            root = Path(directory)
            def candidate(name, config, stamp, summary=None):
                folder = root / name; folder.mkdir()
                (folder / 'source.json').write_text(json.dumps({'episode_id': 'episode'}))
                (folder / 'config.json').write_text(json.dumps({'model': config}))
                (folder / 'summary.json').write_text(summary if summary is not None else json.dumps(
                    dict(episode_id='episode', status='completed', created_at_utc=stamp)))
                (folder / 'best.pt').touch()
                return folder
            expected = candidate('sparse', self.config, '2026-11-15T00:00:00+00:00')
            candidate('cnn', {'type': 'cnn'}, '2027-01-01T00:00:00+00:00')
            candidate('interrupted', self.config, '', summary='')
            with warnings.catch_warnings(record=True) as messages:
                warnings.simplefilter('always')
                self.assertEqual(resolve_training_run(root, 'episode'), expected)
                self.assertTrue(any('Skipping' in str(m.message) for m in messages))
            with self.assertRaisesRegex(ValueError, 'Invalid selected sparse run'):
                resolve_training_run(root, 'episode', root / 'interrupted')
            with self.assertRaisesRegex(ValueError, 'not a sparse'):
                resolve_training_run(root, 'episode', root / 'cnn')


if __name__ == '__main__':
    unittest.main()
