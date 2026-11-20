"""Export a downloaded known-map/temporal checkpoint to native MATLAB FP32.

Reads only model definitions from the notebook AST, never data/training cells.
No CARC paths or original training exports are needed. Includes independent
PyTorch sequence fixtures so MATLAB must demonstrate numerical parity first.
"""
from __future__ import annotations
import argparse
import ast
from collections import deque
from contextlib import nullcontext
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import os

import numpy as np
from scipy.io import savemat
import torch
from torch import nn
from torch.nn import functional as F

SCHEMA = "ares-map-temporal-native-v1"
DEFINITIONS = ("require", "autocast_context", "rotation_ned", "body_ray_directions", "raycast_known_map",
               "encode_sensor_state", "pool_observed_history", "model_inputs", "MultiUavReconstruction",
               "pad_ring", "SparseConv2d", "RingConv2d", "SparseRangeReconstructor")
INPUT_KEYS = ("ranges", "valid", "available", "states", "goals", "history", "present", "ages", "obstacles")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_definitions(notebook):
    nodes = {}
    for cell in json.loads(Path(notebook).read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] == "code":
            for node in ast.parse("".join(cell["source"])).body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in DEFINITIONS:
                    if node.name in nodes:
                        raise ValueError(f"Duplicate architecture definition: {node.name}")
                    nodes[node.name] = node
    if set(nodes) != set(DEFINITIONS):
        raise ValueError(f"Missing architecture definitions: {set(DEFINITIONS)-set(nodes)}")
    namespace = dict(torch=torch, nn=nn, F=F, np=np, copy=copy, math=math, deque=deque,
                     nullcontext=nullcontext, N_EL=15, N_AZ=716, MAX_RANGE=7., INPUT_KEYS=INPUT_KEYS)
    tree = ast.Module(body=[nodes[name] for name in DEFINITIONS], type_ignores=[])
    fingerprint = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
    exec(compile(tree, "<export-map-temporal-definitions>", "exec"), namespace)
    return namespace, fingerprint


def cells(items):
    result = np.empty((1, len(items)), object)
    for i, value in enumerate(items): result[0, i] = value
    return result


def linear(layer):
    return dict(weight=layer.weight.detach().cpu().numpy(), bias=layer.bias.detach().cpu().numpy()[:, None])


def conv(layer, *, sparse=False, circular=True):
    actual = layer.conv if hasattr(layer, "conv") else layer
    weight = actual.weight.detach().cpu().numpy()
    out, channels, elevation, azimuth = weight.shape
    bias = layer.bias if sparse else actual.bias
    return dict(weight=weight.transpose(0, 3, 2, 1).reshape(out, -1).copy(),
                bias=bias.detach().cpu().numpy()[:, None], kernel=np.array([elevation, azimuth], np.int32),
                stride=np.array(actual.stride, np.int32), circularAzimuth=circular)


def pack_model(model, config, boxes):
    c = config["model"]
    # MATLAB beam vectors have elevation fastest. Preserve the checkpoint's
    # exact FP32 directions rather than independently regenerating linspace.
    body = model.rays_body.detach().numpy().reshape(15, 716, 3).transpose(1, 0, 2).reshape(-1, 3).copy()
    return dict(schemaVersion=SCHEMA, maxRange=np.float32(c["max_range"]),
        historySteps=config["data"]["history_steps"], samplePeriodS=config["data"]["sample_period_s"],
        historyHidden=c["history_hidden"], positionScale=np.float32(c["position_scale_m"]),
        velocityScale=np.float32(c["velocity_scale_mps"]), residualLimit=np.float32(c["residual_limit_m"]),
        obstacles=np.asarray(boxes, np.float32), raysBody=body,
        raysSensor=body @ np.asarray(c["sensor_to_body_rotation"], np.float32),
        mountingTranslation=np.asarray(c["mounting_translation_body_m"], np.float32),
        encoder=cells([dict(first=conv(a, sparse=True), second=conv(b, sparse=True)) for a, b in model.encoder]),
        decoder=cells([dict(first=conv(block[0]), second=conv(block[2])) for block in model.decoder]),
        mapFirst=conv(model.map_encoder[0]), mapSecond=conv(model.map_encoder[2]),
        historyConv=conv(model.history_encoder[0], circular=False), historyLinear=linear(model.history_encoder[4]),
        obstacleFirst=linear(model.obstacle_encoder[0]), obstacleSecond=linear(model.obstacle_encoder[2]),
        gruInputWeight=model.history_gru.weight_ih.detach().numpy(),
        gruHiddenWeight=model.history_gru.weight_hh.detach().numpy(),
        gruInputBias=model.history_gru.bias_ih.detach().numpy()[:, None],
        gruHiddenBias=model.history_gru.bias_hh.detach().numpy()[:, None],
        filmFirst=linear(model.film[0]), filmSecond=linear(model.film[2]), head=conv(model.head),
        poolStarts=(np.linspace(0, 716, 91, dtype=int)+1).astype(np.int32),
        worldBounds=np.array([config["map"]["north_bounds_m"], config["map"]["east_bounds_m"],
                             [-config["map"]["altitude_bounds_m"][1], -config["map"]["altitude_bounds_m"][0]]]))


def make_fixtures(namespace, model, config, boxes):
    config = copy.deepcopy(config)
    config["runtime"]["device"] = "cpu"
    runtime = namespace["MultiUavReconstruction"](model, config)
    fixtures = []
    cases = [("cold_blackout", 1, False, [0., 0., -7.], [0., 0., 0.]),
             ("history_then_blackout", 7, True, [10., 7., -5.], [.02, -.03, .41]),
             ("blackout_longer_than_history", config["data"]["history_steps"]+3, False, [-2., 7., -1.3], [.01, -.02, .17]),
             ("high_altitude_blackout", 1, False, [25., 29., -9.5], [-.03, .02, -.35]),
             ("healthy_recovery", 5, True, [8., 3., -7.], [0., 0., .1])]
    for name, steps, measured_history, initial, attitude in cases:
        runtime.reset()
        for step in range(steps):
            state = np.array([initial[0]+.03*step, initial[1], initial[2], .3, 0., 0., *attitude], np.float32)
            goal = np.array([30., 35., -7.], np.float32)
            with torch.inference_mode():
                ranges, valid = namespace["raycast_known_map"](torch.from_numpy(state[None]), torch.from_numpy(boxes[None]),
                    model.rays_body, model.mount)
            ranges, valid = ranges.numpy().reshape(15, 716), valid.numpy().reshape(15, 716)
            healthy = measured_history and (step < 3 or name == "healthy_recovery" and step == steps-1)
            available = np.full((15, 716), healthy, bool)
            observed = np.where(available, ranges, 0.).astype(np.float32)
            observed_valid = valid & available
            prediction = runtime.step("uav", step*.1, observed, observed_valid, available, state, goal, boxes)
        values = list(runtime.buffers["uav"])
        h = config["data"]["history_steps"]
        states = np.zeros((h, 9), np.float32); goals = np.zeros((h, 3), np.float32)
        history = np.zeros((h, 3, 3, 90), np.float32); present = np.zeros(h, bool); ages = np.zeros(h, np.float32)
        for i, value in enumerate(values, h-len(values)):
            states[i], goals[i], history[i], present[i], ages[i] = value["state"], value["goal"], value["summary"], True, value["age"]
        fixtures.append(dict(name=name, ranges=observed, valid=observed_valid, available=available,
            states=states, goals=goals, history=history.transpose(1, 2, 3, 0).copy(), present=present[:, None],
            ages=ages[:, None], expected=prediction))
    return cells(fixtures)


def export(project_root, checkpoint_path, output_root=None):
    root, checkpoint_path = Path(project_root).resolve(), Path(checkpoint_path).resolve(strict=True)
    if checkpoint_path.is_dir(): checkpoint_path = checkpoint_path/"best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config["model"]["state_features"] != 18 or config["model"]["history_grid"] != [3, 90]:
        raise ValueError("Expected the known-map temporal CNN checkpoint, not the single-episode network")
    namespace, architecture = load_definitions(root/"notebooks/train_sparse_reconstruction_map_temporal.ipynb")
    model = namespace["SparseRangeReconstructor"](config["model"], config["data"]["sample_period_s"]).eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError("Nonfinite checkpoint weights")
    boxes = np.asarray(checkpoint["source_metadata"]["scene"]["obstacles_ned_center_dimensions_m"], np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 6 or not np.isfinite(boxes).all() or not (boxes[:, 3:] > 0).all():
        raise ValueError("Invalid supplied obstacle map")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    signature = hashlib.sha256((sha256(checkpoint_path)+sha256(__file__)+architecture).encode()).hexdigest()[:20]
    folder = (Path(output_root) if output_root else root/"results/map_temporal_exports")/signature
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder/"model.mat"
    packed = pack_model(model, config, boxes)
    with torch.inference_mode():
        fixtures = make_fixtures(namespace, model, config, boxes)
    if not destination.is_file():
        temporary = folder/"model.mat.tmp"
        savemat(temporary, dict(model=packed, fixtures=fixtures, absoluteTolerance=np.float64(5e-4)),
                appendmat=False, do_compression=True, long_field_names=True)
        os.replace(temporary, destination)
    manifest = dict(schemaVersion=SCHEMA, checkpoint_path=str(checkpoint_path), checkpoint_sha256=sha256(checkpoint_path),
        completed_epoch=checkpoint["epoch"], best_epoch=checkpoint["best_epoch"], resume_state=checkpoint.get("resume_state"),
        model_path=str(destination), model_mat_sha256=sha256(destination), exporter_sha256=sha256(__file__),
        architecture_sha256=architecture, created_at_utc=datetime.now(timezone.utc).isoformat(),
        weights="selected checkpoint model_state_dict; FP32", runtime="native MATLAB; no online Python",
        history_steps=config["data"]["history_steps"], matlab_parity_verified=False)
    (folder/"export_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    (folder.parent/"deployment.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    export(args.project_root, args.checkpoint, args.output_root)
