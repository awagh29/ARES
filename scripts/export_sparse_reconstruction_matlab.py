"""Export the notebook's selected sparse CNN/FiLM checkpoint to native MATLAB."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import uuid
import warnings

import numpy as np
from scipy.io import savemat
import torch
import sparse_reconstruction_inference as inference
from export_dense_saits_matlab import atomic_json, read_json, require, sha256

SCHEMA = 'ares-sparse-reconstruction-single-episode-mat-v1'
DEFINITIONS = ('encode_sensor_state', 'pad_ring', 'SparseConv2d', 'RingConv2d', 'SparseRangeReconstructor')
NOTEBOOK = 'train_sparse_reconstruction_single_episode_baseline.ipynb'


def architecture_fingerprint(notebook_path):
    """Compare definitions without executing notebook imports, GPU or training cells."""
    notebook = read_json(notebook_path)
    found = {}
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        # Unrelated setup cells may contain unfinished user edits.
        if not any(re.search(r'^(?:def|class) ' + name + r'\b', source, re.M) for name in DEFINITIONS):
            continue
        for node in ast.parse(source).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in DEFINITIONS:
                require(node.name not in found, f'Duplicate notebook definition: {node.name}')
                found[node.name] = ast.dump(node, include_attributes=False)
    for name in DEFINITIONS:
        live = ast.dump(ast.parse(inspect.getsource(getattr(inference, name))).body[0], include_attributes=False)
        require(found.get(name) == live, f'Notebook architecture changed: {name}; update the native exporter before deployment')
    return hashlib.sha256(json.dumps(found, sort_keys=True).encode()).hexdigest()


def validate_model_config(c):
    require(c['elevation_vectors'] == 15 and c['azimuth_vectors'] == 716 and c['max_range'] == 7,
            'Only the recorded 15x716, 7m sensor is supported')
    require(c['azimuth_degree'] == [-179., 179.], 'Unsupported sensor azimuth convention')
    require(type(c['use_position']) is bool and c['input_features'] == (12 if c['use_position'] else 9),
            'State feature count does not match position setting')
    require(isinstance(c['kernel'], int) and c['kernel'] > 0 and c['kernel'] % 2 == 1 and c['kernel'] <= 715,
            'Expected a positive odd azimuth kernel')
    require(c['widths'] and all(isinstance(v, int) and v > 0 for v in c['widths']), 'Invalid encoder widths')
    require(716 % (2 ** (len(c['widths']) - 1)) == 0, 'Decoder upsampling would not match skip sizes')


def resolve_training_run(results_root, episode_id, training_run=None):
    explicit = training_run is not None
    candidates = [Path(training_run).resolve()] if explicit else sorted(p.parent for p in Path(results_root).rglob('source.json'))
    matches = []
    for folder in candidates:
        try:
            source = read_json(folder / 'source.json')
            if source.get('episode_id') != episode_id:
                require(not explicit, f'Selected run belongs to another episode: {folder}')
                continue
            config = read_json(folder / 'config.json')
            c = config.get('model', {})
            sparse = {'widths', 'state_hidden', 'kernel', 'use_position', 'azimuth_vectors'} <= c.keys()
            if not sparse:
                require(not explicit, f'Selected run is not a sparse reconstruction model: {folder}')
                continue
            summary = read_json(folder / 'summary.json')
            if summary.get('status') not in ('completed', 'early_stopped'):
                require(not explicit, f'Selected sparse run is not completed: {folder}')
                continue
            require(summary.get('episode_id') == episode_id, f'Source/summary identity mismatch: {folder}')
            require((folder / 'best.pt').is_file(), f'Missing sparse best.pt: {folder}')
            stamp = datetime.fromisoformat(summary['created_at_utc'])
            require(stamp.utcoffset() is not None, 'Creation timestamp needs a timezone')
            matches.append((stamp.astimezone(timezone.utc), str(folder), folder))
        except (ValueError, KeyError, TypeError) as error:
            if explicit:
                raise ValueError(f'Invalid selected sparse run {folder}: {error}') from error
            warnings.warn(f'Skipping unusable run {folder}: {error}', UserWarning, stacklevel=2)
    require(bool(matches), f'No completed sparse reconstruction run for {episode_id} under {results_root}')
    return max(matches)[2]


def pack_conv(module, *, sparse=False):
    conv = module.conv if hasattr(module, 'conv') else module
    weight = conv.weight.detach().cpu().numpy()
    out, channels, elevation, azimuth = weight.shape
    bias = module.bias if sparse else conv.bias
    return dict(weight=weight.transpose(0, 3, 2, 1).reshape(out, -1).copy(),
                bias=bias.detach().cpu().numpy().reshape(-1, 1),
                kernel=np.array([elevation, azimuth], np.int32),
                stride=np.array(conv.stride, np.int32), inputChannels=channels)


def cells(values):
    result = np.empty((1, len(values)), dtype=object)
    for i, value in enumerate(values):
        result[0, i] = value
    return result


def pack_model(model, episode_id):
    def linear(module):
        return dict(weight=module.weight.detach().numpy(), bias=module.bias.detach().numpy().reshape(-1, 1))
    return dict(schemaVersion=SCHEMA, episodeId=episode_id, gridSize=np.array([15, 716], np.int32),
                maxRange=np.float32(model.max_range), usePosition=model.model_config['use_position'],
                stateFeatures=model.model_config['input_features'],
                encoder=cells([dict(first=pack_conv(a, sparse=True), second=pack_conv(b, sparse=True)) for a, b in model.encoder]),
                decoder=cells([dict(first=pack_conv(block[0]), second=pack_conv(block[2])) for block in model.decoder]),
                filmFirst=linear(model.film[0]), filmSecond=linear(model.film[2]), head=pack_conv(model.head),
                beamOrder='elevation-index-fastest-then-azimuth-index', historyLength=1)


def export_episode(project_root, episode_id, *, training_run=None, results_root=None, output_root=None):
    require(bool(re.fullmatch('[A-Za-z0-9-]+', episode_id)), 'Invalid episode ID')
    root = Path(project_root).resolve()
    fingerprint = architecture_fingerprint(root / 'notebooks' / NOTEBOOK)
    folder = resolve_training_run(results_root or root / 'results', episode_id, training_run)
    config, source, summary = [read_json(folder / name) for name in ('config.json', 'source.json', 'summary.json')]
    cp_path = folder / 'best.pt'
    cp = torch.load(cp_path, map_location='cpu', weights_only=True)
    require(cp['config'] == config and cp['source_metadata'] == source, 'Checkpoint differs from saved config/source metadata')
    require(cp['split_ids'] == read_json(folder / 'splits.json'), 'Checkpoint differs from saved split IDs')
    require(source['episode_id'] == episode_id, 'Checkpoint belongs to another episode')
    require(cp['epoch'] == cp['best_epoch'] == summary['best_epoch'], 'best.pt is not the selected checkpoint')
    metric = config['training']['selection_metric']
    require(cp['selection_metric'] == summary['selection_metric'] == metric, 'Selection metrics differ')
    require(np.isfinite(cp['best_val_metric']) and cp['best_val_metric'] == summary['best_val_metric'], 'Best metric differs')
    history = cp['history']
    best = history['epoch'].index(cp['best_epoch'])
    require(history[metric][best] == cp['best_val_metric'] == min(history[metric]), 'Checkpoint selection does not match its history')
    validate_model_config(config['model'])
    manifest_path = root / 'data/raw' / episode_id / 'manifest.json'
    npz_path = root / 'data/processed/lidar_episode_v1' / episode_id / 'episode.npz'
    require(sha256(manifest_path) == source['manifest_sha256'], 'Source episode manifest changed after training')
    require(sha256(npz_path) == source['npz_sha256'], 'Training episode NPZ changed after training')
    require(read_json(manifest_path)['runId'] == episode_id, 'Raw episode ID differs')
    with np.load(npz_path, allow_pickle=False) as archive:
        require(str(archive['run_id'].item()) == episode_id, 'NPZ episode ID differs')
        require(str(archive['schema_version'].item()) == 'ares-python-lidar-episode-v1', 'Wrong NPZ schema')
        metadata = json.loads(str(archive['metadata_json'].item()))
        require(metadata['lidar_grid_size'] == [15, 716] and metadata['lidar_channel_order'] ==
                'elevation-index-fastest-then-azimuth-index', 'Incompatible beam ordering')
        count = len(archive['time_s'])
        require(count == source['total_scans'], 'Source scan count differs')
        require(archive['lidar_availability_mask'].all() and not archive['lidar_missing_mask'].any() and
                archive['lidar_enabled'].all(), 'Source is not a healthy episode')
        boundary = int(np.floor(count * config['data']['train_fraction']))
        expected = {'train': [f'{episode_id}:{i+1}' for i in range(boundary)],
                    'validation': [f'{episode_id}:{i+1}' for i in range(boundary, count)]}
        require(cp['split_ids'] == expected, 'Recorded chronological split differs')
        indices = [boundary, (boundary + count - 1) // 2, count - 1]
        states = np.column_stack([archive[k][indices] for k in ('position_ned_m', 'velocity_ned_mps', 'attitude_ned_rpy_rad')]).astype(np.float32)
        scans = archive['lidar_ranges_m'][indices].reshape(-1, 716, 15).transpose(0, 2, 1).astype(np.float32)
        valid = archive['lidar_valid_return_mask'][indices].reshape(-1, 716, 15).transpose(0, 2, 1).astype(bool)
    model = inference.SparseRangeReconstructor(config['model']).eval()
    model.load_state_dict(cp['model_state_dict'], strict=True)
    require(all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in model.state_dict().values()), 'Expected finite FP32 model weights')
    with np.load(folder / 'validation_outages.npz', allow_pickle=False) as saved:
        require(saved['scan_ids'].tolist() == expected['validation'], 'Validation outage IDs differ')
        validation_mask = saved['available'][0].astype(bool)
    fixtures = []
    for label, i, mask in [('healthy', 0, np.ones((15, 716), bool)), ('saved_spatial_outage', 0, validation_mask),
                           ('wrapped_spatial_outage', 1, np.tile((np.arange(716) >= 30) & (np.arange(716) < 680), (15, 1))),
                           ('fully_disabled', 1, np.zeros((15, 716), bool)), ('disabled_cold_start', 2, np.zeros((15, 716), bool)),
                           ('observed_no_returns', 1, np.ones((15, 716), bool))]:
        r, v = scans[i].copy(), valid[i].copy()
        if label == 'observed_no_returns':
            r.fill(7); v.fill(False)
        r[~mask] = 0; v &= mask
        with torch.inference_mode():
            encoded = inference.encode_sensor_state(torch.from_numpy(states[i:i+1]), config['model']['use_position'])
            prediction = model(torch.from_numpy(r[None]), torch.from_numpy(v[None]), torch.from_numpy(mask[None]), encoded)[0].numpy()
        require(np.isfinite(prediction).all(), 'Nonfinite reference prediction')
        fixtures.append((label, r, v, mask, states[i], prediction))
    packed_fixtures = dict(names=cells([f[0] for f in fixtures]), ranges=np.stack([f[1] for f in fixtures], -1),
                           validReturn=np.stack([f[2] for f in fixtures], -1), available=np.stack([f[3] for f in fixtures], -1),
                           rawState=np.stack([f[4] for f in fixtures]), predictedRanges=np.stack([f[5] for f in fixtures], -1),
                           absoluteTolerance=5e-4)
    checkpoint_sha, exporter_sha = sha256(cp_path), sha256(__file__)
    module_sha = sha256(inference.__file__)
    version = hashlib.sha256((checkpoint_sha + exporter_sha + module_sha + fingerprint).encode()).hexdigest()[:16]
    episode_folder = (Path(output_root).resolve() if output_root else root / 'data/models/sparse_reconstruction_single_episode') / episode_id
    episode_folder.mkdir(parents=True, exist_ok=True)
    destination = episode_folder / version
    if destination.exists():
        exported = read_json(destination / 'export_manifest.json')
        require(exported['checkpoint_sha256'] == checkpoint_sha and exported['model_mat_sha256'] == sha256(destination / 'model.mat'), 'Published export was modified')
    else:
        staging = episode_folder / ('.building_' + uuid.uuid4().hex)
        staging.mkdir()
        savemat(staging / 'model.mat', dict(model=pack_model(model, episode_id), fixtures=packed_fixtures), do_compression=True)
        exported = dict(schema_version=SCHEMA, episode_id=episode_id, training_run_dir=str(folder), checkpoint_path=str(cp_path),
                        checkpoint_sha256=checkpoint_sha, model_mat_sha256=sha256(staging / 'model.mat'),
                        source_manifest_sha256=source['manifest_sha256'], source_npz_sha256=source['npz_sha256'],
                        exporter_sha256=exporter_sha, inference_source_sha256=module_sha, notebook_architecture_sha256=fingerprint,
                        architecture_provenance='inspected-notebook-legacy-profile; checkpoint has no train-time source fingerprint',
                        best_epoch=cp['best_epoch'], selection_metric=metric, best_val_metric=cp['best_val_metric'],
                        state_features=config['model']['input_features'], use_position=config['model']['use_position'], history_length=1,
                        matlab_parity_verified=False, barrier_enabled=False, closed_loop_training=False,
                        training_masking=config['masking'],
                        deployment_scope='Per-scan model; whole-scan temporal outages are more severe than the training azimuth-block gaps.')
        atomic_json(staging / 'export_manifest.json', exported)
        os.rename(staging, destination)
    pointer = dict(exported, model_path=str(destination / 'model.mat'), export_dir=str(destination))
    atomic_json(episode_folder / 'deployment.json', pointer)
    return pointer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--episode-id', required=True)
    parser.add_argument('--training-run', type=Path)
    parser.add_argument('--results-root', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(export_episode(args.project_root, args.episode_id, training_run=args.training_run, results_root=args.results_root)['model_path'])


if __name__ == '__main__':
    main()
