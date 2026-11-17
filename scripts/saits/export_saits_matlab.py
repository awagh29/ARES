"""Export a trained endpoint SAITS checkpoint for toolbox-free MATLAB inference.

Usage:
    python scripts/export_saits_matlab.py --checkpoint saits/episode_001_t4_seed42/best.pt

Requires NumPy, PyTorch and SciPy. Training files are read only. The exported
MAT-v5 file contains ordinary numeric structs/cells; MATLAB needs no Python,
ONNX importer, or Deep Learning Toolbox. Reference predictions use CPU FP32
evaluation, not the original T4 FP16 autocast execution. Cross-library parity
is numerical, not a promise of bitwise identity across BLAS implementations.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
from scipy.io import loadmat, savemat
import torch

from saits_endpoint_model import SAITSEndpointModel


SCHEMA = "ares-saits-native-mat-v1"
TRAINING_FORMAT = "ares-saits-endpoint-training-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_sha256(path):
    # Matches the trainer's newline-normalized source signature.
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def state_sha256(state):
    digest = hashlib.sha256()
    for name, tensor in state.items():
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def validate_transform(transform):
    require(len(transform["feature_names"]) == 31 and len(transform["target_columns"]) == 14,
            "Expected the 31-input/14-target schema.")
    require(transform["feature_names"][:14] == transform["target_columns"],
            "Targets must be the first 14 input features.")
    expected = ([f"sector_{index:02d}_distance_m" for index in range(7)]
                + [f"sector_{index:02d}_return_fraction" for index in range(7)])
    require(transform["target_columns"] == expected, "Sector target order changed.")
    mean = np.asarray(transform["context_mean"], dtype=np.float64)
    std = np.asarray(transform["context_std"], dtype=np.float64)
    require(mean.shape == std.shape == (17,) and np.isfinite(mean).all()
            and np.isfinite(std).all() and (std > 0).all(), "Invalid context normalization.")
    require(float(transform["distance_scale"]) == 7.0, "Expected a 7 metre distance scale.")


def normalize_features(raw_features, observed_mask, transform):
    """Reproduce FeatureTransform's cast order, then fill missing entries.

    Masked finite placeholders are allowed; NaNs are allowed only where the
    explicit mask is false. Context statistics intentionally remain float64.
    """
    validate_transform(transform)
    raw = np.asarray(raw_features, dtype=np.float32).copy()
    mask_input = np.asarray(observed_mask)
    require(raw.shape[-1] == 31 and mask_input.shape == raw.shape,
            "Feature and mask dimensions must match with 31 trailing features.")
    require(np.isin(mask_input, (0, 1)).all(), "Observation mask must be binary.")
    mask = mask_input.astype(bool, copy=False)
    require(np.isfinite(raw[mask]).all() and not np.isinf(raw).any(),
            "Observed inputs must be finite; infinite inputs are not supported.")
    raw[~mask] = np.nan
    raw[..., :7] /= np.float32(transform["distance_scale"])
    mean = np.asarray(transform["context_mean"], dtype=np.float64)
    std = np.asarray(transform["context_std"], dtype=np.float64)
    raw[..., 14:] = (raw[..., 14:] - mean) / std
    raw[~mask] = np.float32(0)
    require(np.isfinite(raw).all(), "Normalized features must be finite.")
    return np.ascontiguousarray(raw), np.ascontiguousarray(mask)


def _numpy(state, name):
    tensor = state[name]
    require(tensor.dtype == torch.float32, f"Expected unchanged FP32 parameter: {name}")
    value = tensor.detach().cpu().contiguous().numpy().copy()
    require(np.isfinite(value).all(), f"Nonfinite model parameter: {name}")
    return value


def _linear(state, prefix):
    return {"W": np.ascontiguousarray(_numpy(state, prefix + ".weight").T),
            "b": _numpy(state, prefix + ".bias").reshape(1, -1)}


def _norm(state, prefix, epsilon):
    return {"scale": _numpy(state, prefix + ".weight").reshape(1, -1),
            "bias": _numpy(state, prefix + ".bias").reshape(1, -1),
            "epsilon": float(epsilon)}


def _layer(state, prefix):
    return {
        "attentionNorm": _norm(state, prefix + ".attention_norm", 1e-5),
        "queryW": np.ascontiguousarray(_numpy(state, prefix + ".query.weight").T),
        "keyW": np.ascontiguousarray(_numpy(state, prefix + ".key.weight").T),
        "valueW": np.ascontiguousarray(_numpy(state, prefix + ".value.weight").T),
        "outputW": np.ascontiguousarray(_numpy(state, prefix + ".output.weight").T),
        "ffNorm": _norm(state, prefix + ".ff_norm", 1e-6),
        "ffFirst": _linear(state, prefix + ".ff_first"),
        "ffSecond": _linear(state, prefix + ".ff_second"),
    }


def _cellstr(values):
    return np.asarray(list(values), dtype=object).reshape(1, -1)


def pack_model(checkpoint):
    kwargs = dict(checkpoint["model_kwargs"])
    require((kwargs["n_steps"], kwargs["n_features"], kwargs["n_targets"]) == (50, 31, 14),
            "This deployment contract requires a 50x31 window and 14 targets.")
    transform = checkpoint["transform"]
    validate_transform(transform)
    state = checkpoint["model_state_dict"]
    first = np.empty((1, kwargs["n_layers"]), dtype=object)
    second = np.empty_like(first)
    for index in range(kwargs["n_layers"]):
        first[0, index] = _layer(state, f"first_block.{index}")
        second[0, index] = _layer(state, f"second_block.{index}")
    packed = {
        "schemaVersion": SCHEMA,
        "nSteps": kwargs["n_steps"], "nFeatures": kwargs["n_features"],
        "nTargets": kwargs["n_targets"], "dModel": kwargs["d_model"],
        "nHeads": kwargs["n_heads"], "headDim": kwargs["d_model"] // kwargs["n_heads"],
        "nLayers": kwargs["n_layers"],
        "contextMean": np.asarray(transform["context_mean"], dtype=np.float64).reshape(1, 17),
        "contextStd": np.asarray(transform["context_std"], dtype=np.float64).reshape(1, 17),
        "distanceScale": np.float32(transform["distance_scale"]),
        "targetScale": np.asarray([7.0] * 7 + [1.0] * 7, dtype=np.float32).reshape(1, 14),
        "positionEncoding": _numpy(state, "position_encoding")[0],
        "featureNames": _cellstr(transform["feature_names"]),
        "targetColumns": _cellstr(transform["target_columns"]),
        "targetUnits": _cellstr(["m"] * 7 + ["fraction"] * 7),
        "firstBlock": first, "secondBlock": second,
        "embeddingFirst": _linear(state, "embedding_first"),
        "embeddingSecond": _linear(state, "embedding_second"),
        "reduceFirst": _linear(state, "reduce_first"),
        "reduceSecondBeta": _linear(state, "reduce_second_beta"),
        "reduceSecondGamma": _linear(state, "reduce_second_gamma"),
        "combinationGate": _linear(state, "combination_gate"),
        "diagonalAttentionBlocked": True,
        "causalAttention": False,
        "inferenceAttentionDropout": 0.0,
        "layerNormVariance": "population",
        "predictionClipped": False,
        "sensorAzimuthDeg": np.linspace(-179, 179, 716, dtype=np.float64).reshape(1, -1),
        "sectorBodyBearingEdgesDeg": np.linspace(-60, 60, 8, dtype=np.float64).reshape(1, -1),
        "sensorToBodyBearingSign": -1.0,
        "lidarGridSize": np.asarray([[15, 716]], dtype=np.int32),
        "outputDefinition": "first fourteen features at last window row; seven distances then seven fractions",
    }
    return packed


def make_reference_inputs(transform):
    """Deterministic verification vectors, not new training/evaluation samples."""
    steps = 50
    time = np.arange(steps, dtype=np.float64) * 0.1
    base = np.zeros((steps, 31), dtype=np.float64)
    for sector in range(7):
        base[:, sector] = 3.8 + 1.5 * np.sin(0.31 * time + 0.2 * sector)
        base[:, 7 + sector] = 0.25 + 0.2 * np.cos(0.2 * time + 0.15 * sector)
    base[:, 17] = 0.7 + 0.1 * np.sin(time)
    base[:, 18] = 1.1 + 0.05 * np.cos(time)
    base[:, 19] = 0.01 * np.sin(time)
    for index, angle in enumerate((0.05 * np.sin(time), 0.08 * np.cos(time), 0.5 + 0.1 * time)):
        base[:, 20 + 2 * index] = np.sin(angle)
        base[:, 21 + 2 * index] = np.cos(angle)
    base[:, 26] = 0.05 * np.cos(time)
    base[:, 27] = -0.08 * np.sin(time)
    base[:, 28] = 0.1
    base[:, 29] = 1
    names, raw_cases, masks = [], [], []
    for name, missing_count, no_returns in (
        ("healthy_returns", 0, False),
        ("healthy_no_return", 0, True),
        ("outage_1_scan", 1, False),
        ("outage_10_scans", 10, False),
        ("outage_30_scans", 30, False),
    ):
        raw = base.copy()
        mask = np.ones_like(raw, dtype=bool)
        if no_returns:
            raw[:, :7] = 7
            raw[:, 7:14] = 0
        if missing_count:
            start = steps - missing_count
            mask[start:, :14] = False
            raw[start:, :14] = 0  # explicit finite placeholders, mask stays false
            age = time[start:] - time[start - 1]
            raw[start:, 14:17] = age[:, None] * raw[start:, 17:20]
            raw[start:, 29] = 0
            raw[start:, 30] = age
        names.append(name)
        raw_cases.append(raw.astype(np.float32))
        masks.append(mask)
    return names, np.stack(raw_cases), np.stack(masks)


def build_fixtures(model, transform):
    names, raw, masks = make_reference_inputs(transform)
    normalized, masks = normalize_features(raw, masks, transform)
    predicted, first, second, attention = [], [], [], []
    model.eval()
    # Always run each case with batch size one, matching the deployment call.
    with torch.inference_mode():
        for index in range(len(names)):
            result = model(torch.from_numpy(normalized[index:index + 1]),
                           torch.from_numpy(masks[index:index + 1]), return_attention=True)
            predicted.append(result["prediction"].cpu().numpy()[0])
            first.append(result["stage1"].cpu().numpy()[0])
            second.append(result["stage2"].cpu().numpy()[0])
            attention.append(result["attention"].cpu().numpy()[0])
    predicted = np.stack(predicted).astype(np.float32, copy=False)
    scale = np.asarray([7.0] * 7 + [1.0] * 7, dtype=np.float32)
    physical = predicted * scale
    require(np.isfinite(physical).all(), "Reference prediction is nonfinite.")
    return {
        "caseNames": _cellstr(names),
        "rawFeatures": np.transpose(raw, (1, 2, 0)),
        "observedMask": np.transpose(masks, (1, 2, 0)),
        "normalizedValues": np.transpose(normalized, (1, 2, 0)),
        "predictionNormalized": predicted,
        "predictionPhysical": physical,
        "stage1Normalized": np.stack(first),
        "stage2Normalized": np.stack(second),
        "attention": np.transpose(np.stack(attention), (1, 2, 0)),
        "normalizedAbsoluteTolerance": 2e-5,
        "physicalAbsoluteTolerance": 2e-4,
        "relativeTolerance": 2e-5,
        "definition": "synthetic deterministic parity vectors; not training/test metrics",
    }


def load_checkpoint(checkpoint_path):
    checkpoint_path = Path(checkpoint_path).resolve(strict=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require(checkpoint.get("format_version") == TRAINING_FORMAT, "Unsupported training checkpoint.")
    for name in ("model_kwargs", "model_state_dict", "transform", "source_signature",
                 "epoch", "best_epoch"):
        require(name in checkpoint, f"Missing checkpoint field: {name}")
    expected_source = checkpoint["source_signature"].get("saits_endpoint_model.py")
    require(expected_source == source_sha256(Path(__file__).with_name("saits_endpoint_model.py")),
            "Model source differs from the source used to train this checkpoint.")
    require(checkpoint["epoch"] == checkpoint["best_epoch"],
            "Select best.pt: the checkpoint's current epoch is not its selected best epoch.")
    state = checkpoint["model_state_dict"]
    if checkpoint.get("best_state_dict") is not None:
        best = checkpoint["best_state_dict"]
        require(list(best) == list(state) and all(torch.equal(state[key], best[key]) for key in state),
                "The checkpoint's current weights differ from the selected best weights.")
    model = SAITSEndpointModel(**checkpoint["model_kwargs"]).cpu().float().eval()
    model.load_state_dict(state, strict=True)
    for filename, expected in (("preprocessing.json", checkpoint["transform"]),):
        sidecar = checkpoint_path.parent / filename
        if sidecar.is_file():
            require(json.loads(sidecar.read_text(encoding="utf-8-sig")) == expected,
                    f"Checkpoint and {filename} disagree.")
    config_path = checkpoint_path.parent / "run_config.json"
    if config_path.is_file():
        run_config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        require(run_config["model_kwargs"] == checkpoint["model_kwargs"],
                "Checkpoint and run_config architecture disagree.")
        require(run_config["source_signature"] == checkpoint["source_signature"],
                "Checkpoint and run_config source provenance disagree.")
    validate_transform(checkpoint["transform"])
    return checkpoint, model


def _validate_saved_mat(path, packed, fixtures):
    restored = loadmat(path, struct_as_record=False, squeeze_me=False, mat_dtype=True)
    native = restored["model"][0, 0]
    vectors = restored["fixtures"][0, 0]
    np.testing.assert_array_equal(native.embeddingFirst[0, 0].W, packed["embeddingFirst"]["W"])
    np.testing.assert_array_equal(native.contextMean, packed["contextMean"])
    np.testing.assert_array_equal(native.contextStd, packed["contextStd"])
    np.testing.assert_array_equal(native.positionEncoding, packed["positionEncoding"])
    np.testing.assert_array_equal(vectors.observedMask, fixtures["observedMask"])
    np.testing.assert_array_equal(vectors.predictionPhysical, fixtures["predictionPhysical"])
    require(native.embeddingFirst[0, 0].W.dtype == np.float32,
            "MAT weights lost FP32 precision.")
    require(native.contextMean.dtype == np.float64 and native.contextStd.dtype == np.float64,
            "MAT context statistics lost FP64 precision.")
    require(vectors.observedMask.dtype == np.bool_, "MAT observation mask is not logical.")


def export_checkpoint(checkpoint_path, output_dir=None):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve(strict=True)
    output_dir = (Path(output_dir).expanduser().resolve() if output_dir is not None
                  else Path(__file__).resolve().parents[1] / "data" / "models" / "saits_endpoint")
    if output_dir.exists():
        raise FileExistsError(f"Export directory already exists; it will not be overwritten: {output_dir}")
    checkpoint, model = load_checkpoint(checkpoint_path)
    packed = pack_model(checkpoint)
    fixtures = build_fixtures(model, checkpoint["transform"])
    checkpoint_digest = file_sha256(checkpoint_path)
    provenance = {
        "schemaVersion": SCHEMA,
        "checkpointPath": str(checkpoint_path),
        "checkpointSha256": checkpoint_digest,
        "modelStateSha256": state_sha256(checkpoint["model_state_dict"]),
        "modelSourceSha256": checkpoint["source_signature"]["saits_endpoint_model.py"],
        "exporterSourceSha256": file_sha256(__file__),
        "selectedEpoch": int(checkpoint["best_epoch"]),
        "torchVersion": str(torch.__version__),
        "numpyVersion": str(np.__version__),
        "referenceDevice": "cpu",
        "referencePrecision": "float32; no autocast",
        "trainedDevice": str(checkpoint.get("device_type", "unavailable")),
        "trainingConfigurationJson": json.dumps(checkpoint.get("config", {}), sort_keys=True),
        "sourceSignatureJson": json.dumps(checkpoint["source_signature"], sort_keys=True),
        "dataSignatureJson": json.dumps(checkpoint.get("data_signature", {}), sort_keys=True),
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "weightsModified": False,
        "referenceNote": "CPU FP32 parity; T4 FP16 inference can differ numerically",
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # On Windows, tempfile.mkdtemp requests owner-only mode 0700, which can
    # survive the publication rename and deny a separately launched MATLAB
    # process access. A normal staging directory inherits the parent's ACL.
    # A UUID and exclusive mkdir retain collision protection without changing
    # permissions on this directory, its parent, or any existing artifact.
    temporary = output_dir.parent / (".saits-export-" + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        mat_path = temporary / "model.mat"
        savemat(mat_path, {"model": packed, "fixtures": fixtures, "provenance": provenance},
                do_compression=True, long_field_names=True, oned_as="row")
        _validate_saved_mat(mat_path, packed, fixtures)
        manifest = {
            "schema_version": SCHEMA,
            "artifact": "model.mat",
            "artifact_sha256": file_sha256(mat_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_digest,
            "model_state_sha256": provenance["modelStateSha256"],
            "selected_epoch": provenance["selectedEpoch"],
            "model_kwargs": checkpoint["model_kwargs"],
            "preprocessing": checkpoint["transform"],
            "source_signature": checkpoint["source_signature"],
            "exporter_sha256": provenance["exporterSourceSha256"],
            "reference_execution": "CPU FP32 batch size 1, evaluation mode, no autocast",
            "reference_cases": [str(name) for name in fixtures["caseNames"].flat],
            "input_layout": "50 x 31 physical features plus same-shaped observation mask",
            "output_layout": "1 x 14 physical values: seven metres, seven fractions",
            "matrix_convention": "W is input-by-output; linear(x)=x@W+b",
            "context_normalization": "cast raw to float32, subtract/divide FP64 context statistics, cast result to float32",
            "missing_values": "zero-fill after normalization; explicit mask one means observed",
            "weights_modified": False,
            "prediction_clipped": False,
            "native_mat_readback_passed": True,
            "matlab_parity_verified": False,
            "created_at_utc": provenance["createdAtUtc"],
        }
        for filename, value in (("export_manifest.json", manifest),
                                ("preprocessing.json", checkpoint["transform"])):
            (temporary / filename).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                                              encoding="utf-8")
        require(file_sha256(checkpoint_path) == checkpoint_digest,
                "Checkpoint changed during export; publication refused.")
        # Same-filesystem directory rename publishes a complete package at once.
        # The explicit nonexistence check also gives a useful race diagnostic.
        require(not output_dir.exists(), "Export destination appeared while exporting.")
        os.rename(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"SAITS_MATLAB_EXPORT {output_dir / 'model.mat'}", flush=True)
    print(f"CHECKPOINT_SHA256 {checkpoint_digest} EPOCH {checkpoint['best_epoch']}", flush=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
