"""Locate Atharva's map-temporal checkpoint, reusing an existing MATLAB export when possible."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .matlab_bridge import windows_path
from .paths import REPO_ROOT

EXPORT_ROOT = REPO_ROOT / "results" / "map_temporal_exports"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_model(model_path: str) -> tuple[str, dict]:
    """Return (path for MATLAB, info). A .pt with a matching export becomes that model.mat."""
    if not model_path:
        raise FileNotFoundError("Set controller.model_path in the scenario YAML (best.pt or model.mat).")
    path = Path(model_path)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    info = {"requested": str(path)}
    if path.suffix.lower() == ".pt":
        checksum = sha256(path)
        info["checkpoint_sha256"] = checksum
        for manifest_path in sorted(EXPORT_ROOT.glob("*/export_manifest.json")):
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            exported = manifest_path.parent / "model.mat"
            if manifest.get("checkpoint_sha256") == checksum and exported.exists():
                info.update(reused_export=str(exported), completed_epoch=manifest.get("completed_epoch"),
                            best_epoch=manifest.get("best_epoch"))
                return windows_path(exported), info
    return windows_path(path), info
