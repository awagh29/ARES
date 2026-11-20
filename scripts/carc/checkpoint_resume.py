"""Checkpoint selection, exact training state, allocation time and run locking."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import time


def load_checkpoint(path):
    import torch
    value = torch.load(Path(path), map_location="cpu", weights_only=True)
    required = {"epoch", "best_epoch", "best_val_metric", "model_state_dict", "optimizer_state_dict",
                "scheduler_state_dict", "scaler_state_dict", "config", "history", "source_metadata", "split_ids"}
    if not isinstance(value, dict) or not required.issubset(value):
        raise ValueError(f"Incomplete training checkpoint: {path}")
    epoch = value["epoch"]
    history = value["history"]
    if not isinstance(epoch, int) or epoch < 0 or history.get("epoch") != list(range(1, epoch+1)):
        raise ValueError(f"Checkpoint epoch/history mismatch: {path}")
    if any(not isinstance(v, list) or len(v) != epoch for v in history.values()):
        raise ValueError(f"Incomplete metric history: {path}")
    position = value.get("resume_state", {"epoch": epoch+1, "next_batch": 0, "phase": "epoch_end"})
    if position.get("phase") not in ("train", "epoch_end") or position["epoch"] != epoch+1 or position["next_batch"] < 0:
        raise ValueError(f"Invalid training cursor: {path}")
    return value


def checkpoint_progress(checkpoint):
    cursor = checkpoint.get("resume_state", {})
    return checkpoint["epoch"], int(cursor.get("next_batch", 0))


def resolve_checkpoint(selection, output_root):
    """Pick the most advanced readable checkpoint in the newest run with saves.

    best.pt is a fallback; last.pt is the usual continuation. Incomplete .tmp
    writes are ignored. Explicit checkpoint files fail rather than silently
    choosing another file. Directory/auto selection can recover from corruption.
    """
    if str(selection) == "auto":
        root = Path(output_root).expanduser().resolve()
        folders = [p for p in root.iterdir() if p.is_dir() and any(p.glob("*.pt"))] if root.is_dir() else []
        if not folders:
            raise FileNotFoundError(f"No saved training runs under {root}. Supply --resume /path/to/run or set ARES_RUN_ROOT.")
        # A newer empty run must not hide an older resumable run.
        folders.sort(key=lambda p: max(f.stat().st_mtime_ns for f in p.glob("*.pt")), reverse=True)
        folder = folders[0]
    else:
        folder = Path(selection).expanduser().resolve()
        if folder.is_file():
            return folder, load_checkpoint(folder)
        if not folder.is_dir():
            raise FileNotFoundError(f"Checkpoint/run does not exist: {folder}")
    candidates = [folder/name for name in ("last.pt", "last.prev.pt", "best.pt")]
    candidates += sorted(folder.glob("epoch_*.pt"), reverse=True)
    selected_path, selected, errors = None, None, []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            value = load_checkpoint(path)
        except Exception as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        if selected is None or checkpoint_progress(value) > checkpoint_progress(selected):
            selected_path, selected = path, value
    if selected is None:
        raise ValueError(f"No readable training checkpoint in {folder}: {'; '.join(errors)}")
    for error in errors:
        print("Skipping unreadable checkpoint:", error, flush=True)
    print("Resume checkpoint:", selected_path, "| completed epoch/batch:", checkpoint_progress(selected), flush=True)
    return selected_path, selected


def validate_resume(saved, config, split_ids, source_metadata):
    old, current = copy.deepcopy(saved["config"]), copy.deepcopy(config)
    # Only storage locations, visible GPU index, and the requested total epoch
    # target may vary. Optimizer, batch size, precision, scheduler horizon, data,
    # masks, seeds and model features must keep their original meaning.
    for item in (old, current):
        item.pop("paths", None)
        item["runtime"].pop("gpu_index", None)
        item["training"].pop("epochs", None)
    if old != current:
        raise ValueError("Resume settings differ from the checkpoint. Keep the saved architecture, batch size, precision, data, optimizer and scheduler; only paths/GPU index/total epochs may change.")
    if config["training"]["epochs"] < saved["epoch"]:
        raise ValueError("Requested total epochs is below the completed epoch.")
    if saved["split_ids"] != split_ids or saved["source_metadata"].get("fingerprint") != source_metadata.get("fingerprint"):
        raise ValueError("Resume corpus fingerprint or frozen flight splits changed. Reuse the original data/cache and generation settings.")


def capture_rng():
    import numpy as np
    import torch
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=dict(kind=state[0], keys=state[1].tolist(),
        position=state[2], has_gauss=state[3], cached_gaussian=state[4]), torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(checkpoint):
    import numpy as np
    import torch
    state = checkpoint.get("rng_state")
    if state:
        random.setstate(state["python"])
        n = state["numpy"]
        np.random.set_state((n["kind"], np.asarray(n["keys"], dtype=np.uint32), n["position"], n["has_gauss"], n["cached_gaussian"]))
        torch.set_rng_state(state["torch"].cpu())
        cuda = state["cuda"]
    else:  # Backward compatibility with checkpoints written before resume support.
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        cuda = checkpoint.get("cuda_rng_states", [])
    if torch.cuda.is_available():
        for index, tensor in enumerate(cuda[:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(tensor.cpu(), index)


@contextmanager
def run_lock(folder):
    """OS advisory lock: released even if a job is killed; no stale-lock override."""
    path = Path(folder)/"training.lock"
    stream = path.open("a+b")
    try:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                stream.write(b"0"); stream.flush(); stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f"Another trainer is using {folder}") from exc
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(f"Another trainer is using {folder}") from exc
        yield
    finally:
        stream.close()


def slurm_remaining_seconds():
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None
    try:
        result = subprocess.run(["squeue", "--jobs", job_id, "--noheader", "--format=%L"],
                                capture_output=True, text=True, timeout=10, check=True)
        value = result.stdout.strip().splitlines()[0]
        if value in ("UNLIMITED", "NOT_SET", "INVALID"):
            return None
        day, clock = value.split("-", 1) if "-" in value else ("0", value)
        parts = [int(x) for x in clock.split(":")]
        if len(parts) == 2:
            hours, minutes, seconds = 0, *parts
        elif len(parts) == 3:
            hours, minutes, seconds = parts
        else:
            return None
        return int(day)*86400+hours*3600+minutes*60+seconds
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


class TrainingControl:
    """Periodic saves and a graceful stop at the next finished batch."""
    def __init__(self, checkpoint_minutes=5., stop_after_minutes=None, walltime_margin_minutes=5.):
        if not math.isfinite(checkpoint_minutes) or checkpoint_minutes <= 0:
            raise ValueError("checkpoint_minutes must be positive and finite")
        self.checkpoint_seconds = checkpoint_minutes*60
        self.reason = None
        now = time.monotonic()
        deadlines = []
        if stop_after_minutes is not None:
            deadlines.append(now+stop_after_minutes*60)
        remaining = slurm_remaining_seconds()
        if remaining is not None:
            deadlines.append(now+max(0., remaining-walltime_margin_minutes*60))
            print(f"Slurm time remaining: {remaining/60:.1f} minutes; stopping {walltime_margin_minutes:g} minutes before the limit.", flush=True)
        self.deadline = min(deadlines) if deadlines else None

    def should_stop(self, **position):
        if self.reason is None and self.deadline is not None and time.monotonic() >= self.deadline:
            self.reason = "time budget reached"
        return self.reason is not None

    @contextmanager
    def signals(self):
        previous = {}
        def request_stop(signum, frame):
            self.reason = f"signal {signum}"
        for name in ("SIGTERM", "SIGINT", "SIGUSR1"):
            sig = getattr(signal, name, None)
            if sig is not None:
                previous[sig] = signal.signal(sig, request_stop)
        try:
            yield self
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
