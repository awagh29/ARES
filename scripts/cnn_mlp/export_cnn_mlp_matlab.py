"""Resolve a completed single-episode run and export its best weights to MATLAB."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import warnings

import numpy as np
from scipy.io import savemat
import torch

from cnn_mlp_inference import encode_sensor_state, load_checkpoint_network

SCHEMA = "ares-cnn-mlp-single-episode-mat-v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_training_run(results_root, episode_id, training_run=None):
    """Select the newest intact CNN result; tolerate deleted/in-progress records.

    Automatic discovery skips invalid records. An explicit override remains
    strict: a broken or wrong-family pinned run never silently selects another.
    """
    candidates = [Path(training_run).resolve()] if training_run else sorted(Path(results_root).rglob("source.json"))
    matches = []
    for candidate in candidates:
        folder = candidate if training_run else candidate.parent
        try:
            source = read_json(folder / "source.json")
            require(isinstance(source, dict), "source.json must contain an object.")
            if source.get("episode_id") != episode_id:
                if training_run:
                    raise ValueError("Selected training run belongs to another episode.")
                continue
            config = read_json(folder / "config.json")
            model = config.get("model") if isinstance(config, dict) else None
            cnn_fields = {"input_features", "encoder_hidden", "latent_channels", "latent_length",
                          "decoder_channels", "decoder_lengths", "kernel_sizes", "upsample_mode"}
            is_cnn = isinstance(model, dict) and cnn_fields.issubset(model)
            if not is_cnn:
                if training_run:
                    raise ValueError("Selected training run is not a PoseToRangeCNN result.")
                continue
            # Filter the model family BEFORE reading a different model's summary.
            summary = read_json(folder / "summary.json")
            require(isinstance(summary, dict), "summary.json must contain an object.")
            if summary.get("status") not in ("completed", "early_stopped"):
                if training_run:
                    raise ValueError("Selected training run has not completed successfully.")
                continue
            require(summary.get("episode_id") == episode_id, "Training source/summary episode IDs differ.")
            checkpoint = folder / "best.pt"
            require(checkpoint.is_file() and checkpoint.stat().st_size > 0,
                    f"Completed run has no nonempty best.pt: {folder}")
            created = datetime.fromisoformat(summary["created_at_utc"])
            require(created.tzinfo is not None, "Run creation timestamp must include its timezone.")
            matches.append((created.astimezone(timezone.utc), str(folder), folder))
        except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
            if training_run:
                raise ValueError(f"Cannot use selected training run {folder}: {exc}") from exc
            warnings.warn(f"Skipping incomplete/invalid result {folder}: {exc}", RuntimeWarning)
    require(bool(matches), f"No completed single-episode model found for {episode_id} under {results_root}")
    return max(matches)[2]


def verify_episode_integration(project_root, episode_id):
    """Read-only trace: newest completed training -> deployed weights -> flown weights."""
    root = Path(project_root).resolve()
    folder = resolve_training_run(root / "results", episode_id)
    summary = read_json(folder / "summary.json")
    checkpoint_hash = sha256(folder / "best.pt")
    report = dict(
        episode_id=episode_id, selection_policy="newest completed summary.created_at_utc; best.pt within that run",
        latest_training_run=str(folder), best_epoch=summary["best_epoch"],
        checkpoint_sha256=checkpoint_hash, deployment_matches_latest=False,
        deployed_model_hash_verified=False, matlab_parity_verified=False,
        latest_capture_used_latest=False, capture_logging_verified=False,
    )
    pointer = root / "data/models/cnn_mlp_single_episode" / episode_id / "deployment.json"
    if pointer.is_file():
        deployment = read_json(pointer)
        model_path = Path(deployment["model_path"])
        report.update(deployed_training_run=deployment["training_run_dir"], model_path=str(model_path))
        model_ok = model_path.is_file() and sha256(model_path) == deployment["model_mat_sha256"]
        report["deployed_model_hash_verified"] = model_ok
        report["deployment_matches_latest"] = model_ok and deployment["checkpoint_sha256"] == checkpoint_hash
        validation_path = model_path.parent / "matlab_validation.json"
        if validation_path.is_file():
            validation = read_json(validation_path)
            report["matlab_parity_verified"] = bool(model_ok and validation["allParityChecksPassed"]
                and validation["modelSha256"] == deployment["model_mat_sha256"])

    captures = []
    for path in (root / "data/cnn_single_episode_intermittent").glob("*/manifest.json"):
        manifest = read_json(path)
        if (manifest.get("inference", {}).get("sourceEpisodeId") == episode_id
                and (path.parent / "summary.json").is_file()):
            captures.append((datetime.fromisoformat(manifest["createdAtUtc"].replace("Z", "+00:00")),
                             str(path), manifest))
    if captures:
        _, path, manifest = max(captures)
        inference = manifest["inference"]
        proof = inference["runtimeProof"]
        report.update(latest_capture_manifest=path, latest_capture_run_id=manifest["runId"],
                      captured_training_run=inference["trainingRunDirectory"],
                      captured_checkpoint_sha256=inference["sourceCheckpointSha256"],
                      reconstruction_used_scans=proof["reconstructionUsedScanCount"],
                      controller_influenced_scans=proof["controllerInfluencedScanCount"],
                      total_scans=proof["scanCount"])
        scan_path = Path(path).parent / "scans.csv"
        if scan_path.is_file():
            with scan_path.open(newline="", encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            enabled = lambda row, name: row.get(name, "").strip().lower() in ("1", "true")
            logged_used = sum(enabled(row, "CnnReconstructionUsed") for row in rows)
            logged_influenced = sum(enabled(row, "ControllerInfluencedDuringOutage") for row in rows)
            report["capture_logging_verified"] = bool(
                rows and len(rows) == proof["scanCount"]
                and logged_used == proof["reconstructionUsedScanCount"]
                and logged_influenced == proof["controllerInfluencedScanCount"]
                and not any(enabled(row, "CnnReconstructionUsed")
                            and enabled(row, "LidarMeasuredAvailable") for row in rows))
        report["latest_capture_used_latest"] = bool(
            inference["sourceCheckpointSha256"] == checkpoint_hash
            and report["deployment_matches_latest"]
            and inference["artifactSha256"] == deployment["model_mat_sha256"]
            and proof["instrumentationChecksPassed"] and report["capture_logging_verified"]
            and proof["reconstructionUsedScanCount"] > 0)
    return report


def pack_model(network, config, episode_id):
    def array(tensor):
        require(tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all()),
                "Export requires finite FP32 weights.")
        return tensor.detach().cpu().numpy().copy()

    def cells(items):
        result = np.empty((1, len(items)), dtype=object)
        for i, item in enumerate(items):
            result[0, i] = item
        return result

    c = config["model"]
    def activation(layers):
        # Dropout is an identity in eval mode. The final Linear may have NO activation.
        active = [layer for layer in layers if not isinstance(layer, (torch.nn.Dropout, torch.nn.Identity))]
        require(len(active) <= 1, "MATLAB export supports one activation per affine layer.")
        if not active:
            return "identity"
        names = {torch.nn.GELU: "gelu", torch.nn.ReLU: "relu",
                 torch.nn.SiLU: "silu", torch.nn.Softplus: "softplus"}
        require(type(active[0]) in names, f"Unsupported operation after affine layer: {type(active[0]).__name__}")
        return names[type(active[0])]

    def bias(layer, width):
        return (array(layer.bias).reshape(-1, 1) if layer.bias is not None
                else np.zeros((width, 1), dtype=np.float32))

    layers = list(network.encoder)
    linear_indices = [i for i, layer in enumerate(layers) if isinstance(layer, torch.nn.Linear)]
    widths = [c["input_features"], *c["encoder_hidden"], c["latent_channels"] * c["latent_length"]]
    require(linear_indices and linear_indices[0] == 0 and len(linear_indices) == len(widths) - 1,
            "Encoder layer count differs from its configured widths.")
    encoder = []
    for block, i in enumerate(linear_indices):
        layer = layers[i]
        stop = linear_indices[block + 1] if block + 1 < len(linear_indices) else len(layers)
        require((layer.in_features, layer.out_features) == (widths[block], widths[block + 1]),
                "Encoder dimensions differ from saved configuration.")
        encoder.append({"W": array(layer.weight), "b": bias(layer, layer.out_features),
                        "activation": activation(layers[i + 1:stop])})

    decoder = []
    layers = list(network.decoder)
    starts = [i for i, layer in enumerate(layers) if isinstance(layer, torch.nn.Upsample)]
    require(starts and starts[0] == 0 and len(starts) == len(c["decoder_lengths"]),
            "Decoder layout differs from its configured blocks.")
    inputs = c["latent_channels"]
    for block, i in enumerate(starts):
        upsample = layers[i]
        stop = starts[block + 1] if block + 1 < len(starts) else len(layers)
        require(i + 1 < stop and isinstance(layers[i + 1], torch.nn.Conv1d),
                "Each exported upsample block must start with a Conv1d.")
        layer = layers[i + 1]
        length = upsample.size if isinstance(upsample.size, int) else upsample.size[0]
        kernel = c["kernel_sizes"][block]
        require(upsample.mode == "linear" and upsample.align_corners == c["align_corners"]
                and length == c["decoder_lengths"][block], "Unsupported decoder interpolation.")
        require(layer.in_channels == inputs and layer.out_channels == c["decoder_channels"][block]
                and layer.kernel_size == (kernel,) and kernel % 2 == 1
                and layer.stride == (1,) and layer.dilation == (1,) and layer.groups == 1
                and layer.padding == (kernel // 2,) and layer.padding_mode == "zeros",
                "Unsupported or misconfigured Conv1d geometry.")
        decoder.append({"W": array(layer.weight), "b": bias(layer, layer.out_channels),
                        "length": length, "activation": activation(layers[i + 2:stop])})
        inputs = layer.out_channels
    require(c["input_features"] == 12 and c["upsample_mode"] == "linear", "Unsupported input/interpolation contract.")
    require(c["decoder_channels"][-1] == 15 and c["decoder_lengths"][-1] == 716,
            "Expected full 15x716 range output.")
    return dict(schemaVersion=SCHEMA, episodeId=episode_id, encoder=cells(encoder),
                decoder=cells(decoder), latentChannels=c["latent_channels"],
                latentLength=c["latent_length"], alignCorners=c["align_corners"],
                maximumRangeM=7.0, gridSize=np.array([[15, 716]], dtype=np.int32),
                sensorAzimuthDeg=np.linspace(-179, 179, 716).reshape(1, -1),
                sensorElevationDeg=np.linspace(-15, 15, 15).reshape(-1, 1))


def export_episode(project_root, episode_id, *, results_root=None, training_run=None):
    require(bool(re.fullmatch(r"[A-Za-z0-9-]+", episode_id)), "Invalid episode ID.")
    root = Path(project_root).resolve()
    results = Path(results_root).resolve() if results_root else root / "results"
    folder = resolve_training_run(results, episode_id, training_run)
    checkpoint_path = folder / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config, source, summary = read_json(folder / "config.json"), read_json(folder / "source.json"), read_json(folder / "summary.json")
    require(checkpoint["config"] == config, "Saved configuration differs from best.pt.")
    require(checkpoint["source_metadata"] == source, "Saved episode provenance differs from best.pt.")
    require(source["episode_id"] == episode_id, "Checkpoint belongs to another episode.")
    require(checkpoint["epoch"] == checkpoint["best_epoch"] == summary["best_epoch"], "Select the best-validation checkpoint.")
    manifest_path = root / "data" / "raw" / episode_id / "manifest.json"
    raw_manifest = read_json(manifest_path)
    require(raw_manifest["runId"] == episode_id and sha256(manifest_path) == source["manifest_sha256"],
            "Raw episode manifest differs from the training source.")
    npz = root / "data" / "processed" / "lidar_episode_v1" / episode_id / "episode.npz"
    require(sha256(npz) == source["npz_sha256"], "Recorded training episode changed.")
    network, architecture, architecture_source = load_checkpoint_network(checkpoint)
    architecture_path = folder / "architecture.json"
    if architecture_path.is_file():
        require(read_json(architecture_path) == architecture,
                "Saved architecture.json differs from the checkpoint architecture.")
    packed = pack_model(network, config, episode_id)
    with np.load(npz, allow_pickle=False) as episode:
        states = np.column_stack((episode["position_ned_m"], episode["velocity_ned_mps"],
                                  episode["attitude_ned_rpy_rad"])).astype(np.float32)
    boundary = source["split_ranges_zero_based_end_exclusive"]["validation"][0]
    indices = sorted({0, len(states)//2, boundary, len(states)-1})
    raw = states[indices].copy()
    # Include periodic-angle checks alongside actual recorded motion states.
    periodic = raw[0].copy(); periodic[6:] += np.float32(2 * np.pi)
    raw = np.concatenate((raw, periodic[None]), axis=0)
    outputs, encoded = [], []
    with torch.inference_mode():
        for row in raw:
            features = encode_sensor_state(torch.from_numpy(row[None]))
            encoded.append(features.numpy()[0])
            outputs.append(network(features).numpy()[0])
    fixtures = dict(rawStates=raw, encodedStates=np.stack(encoded),
                    predictedRanges=np.stack(outputs, axis=-1), absoluteTolerance=5e-4)
    checkpoint_sha = sha256(checkpoint_path)
    exporter_sha = sha256(Path(__file__))
    model_source_sha = sha256(Path(__file__).with_name("cnn_mlp_inference.py"))
    version = hashlib.sha256((checkpoint_sha + exporter_sha + model_source_sha).encode()).hexdigest()[:16]
    episode_folder = root / "data" / "models" / "cnn_mlp_single_episode" / episode_id
    destination = episode_folder / version
    episode_folder.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        exported = read_json(destination / "export_manifest.json")
        require(exported["checkpoint_sha256"] == checkpoint_sha
                and exported["model_mat_sha256"] == sha256(destination / "model.mat"),
                "Existing exported artifact was modified.")
    else:
        with tempfile.TemporaryDirectory(prefix="export-", dir=episode_folder) as temporary:
            temporary = Path(temporary)
            savemat(temporary / "model.mat", {"model": packed, "fixtures": fixtures}, do_compression=True)
            atomic_json(temporary / "architecture.json", architecture)
            exported = {
                "schema_version": SCHEMA, "episode_id": episode_id,
                "training_run_dir": str(folder), "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha, "model_mat_sha256": sha256(temporary / "model.mat"),
                "source_manifest_sha256": source["manifest_sha256"], "source_npz_sha256": source["npz_sha256"],
                "initial_position_ned_m": raw_manifest["configuration"]["mission"]["initialPositionNed"],
                "best_epoch": checkpoint["best_epoch"], "validation_mse": float(summary["best_val_mse"]),
                "exporter_sha256": exporter_sha, "model_source_sha256": model_source_sha,
                "architecture_source": architecture_source,
                "encoder_activations": [layer["activation"] for layer in packed["encoder"].flat],
                "matlab_parity_verified": False, "reference_precision": "CPU FP32; not BF16 training autocast",
            }
            atomic_json(temporary / "export_manifest.json", exported)
            destination.mkdir()
            # Both files are prepared before publishing the episode's pointer.
            for name in ("model.mat", "export_manifest.json", "architecture.json"):
                os.replace(temporary / name, destination / name)
    pointer = dict(exported, model_path=str(destination / "model.mat"), export_dir=str(destination))
    atomic_json(episode_folder / "deployment.json", pointer)
    return pointer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--training-run", type=Path)
    parser.add_argument("--verify-integration", action="store_true",
                        help="Read-only: verify newest/deployed/flown checkpoint hashes and scan logs.")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.verify_integration:
        print(json.dumps(verify_episode_integration(args.project_root, args.episode_id), indent=2))
        return
    selected = resolve_training_run(
        args.results_root or args.project_root / "results", args.episode_id, args.training_run
    )
    print(f"Selected completed training run: {selected}", flush=True)
    result = export_episode(args.project_root, args.episode_id,
                            results_root=args.results_root, training_run=selected)
    print(result["model_path"])


if __name__ == "__main__":
    main()
