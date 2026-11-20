#!/usr/bin/env python3
"""Execute the actual map-temporal notebook, in order, in a CARC salloc session.

No Jupyter server, port forwarding, nbconvert, or second copy of training code.
Run --check-only first; it performs source/GPU checks without synthetic generation
or training. The notebook's training settings are preserved unless overridden.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

from prepare_map_temporal_data import prepare
from checkpoint_resume import TrainingControl, resolve_checkpoint, validate_resume

ROOT = Path(__file__).resolve().parents[2]


class Tee:
    def __init__(self, console, stream):
        self.console, self.stream = console, stream

    def write(self, text):
        self.console.write(text)
        self.stream.write(text)
        self.stream.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


def run_notebook(args):
    root, run_root = args.repo_root.expanduser().resolve(), args.run_root.expanduser().resolve()
    notebook = (args.notebook or root/"notebooks/train_sparse_reconstruction_map_temporal.ipynb").resolve()
    os.environ["ARES_REPO_ROOT"], os.environ["ARES_RUN_ROOT"] = str(root), str(run_root)
    os.environ["MPLBACKEND"] = "Agg"
    cpus = os.environ.get("SLURM_CPUS_PER_TASK", "1")
    os.environ.setdefault("OMP_NUM_THREADS", cpus)
    os.environ.setdefault("MKL_NUM_THREADS", cpus)
    payload = notebook.read_bytes()
    cells = json.loads(payload)["cells"]
    resume_path, resume_saved = None, None
    if args.resume:
        resume_path, resume_saved = resolve_checkpoint(args.resume, run_root/"results/sparse_reconstruction_map_temporal")
    control = TrainingControl(args.checkpoint_minutes, args.stop_after_minutes, args.walltime_margin_minutes)
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    launch = run_root/"launches"/f"{job_id}_{stamp}"
    launch.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(notebook, launch/"input_notebook.ipynb")
    status = dict(status="running", notebook=str(notebook), notebook_sha256=hashlib.sha256(payload).hexdigest(),
                  repo_root=str(root), run_root=str(run_root), job_id=job_id, check_only=args.check_only,
                  resume_checkpoint=str(resume_path) if resume_path else None)
    state_path = launch/"status.json"
    def save_status():
        state_path.write_text(json.dumps(status, indent=2)+"\n", encoding="utf-8")
    save_status()
    namespace = {"__name__": "__main__", "__file__": str(notebook),
                 "RESUME_CHECKPOINT": resume_path, "TRAINING_CONTROL": control}
    print("Launch directory:", launch, flush=True)
    with (launch/"console.log").open("w", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)), control.signals():
            try:
                print("Slurm job:", job_id, "| allocated CPUs:", cpus, flush=True)
                for index, cell in enumerate(cells):
                    if cell["cell_type"] != "code":
                        continue
                    if args.check_only and index >= 18:
                        namespace["discover_sources"](root, namespace["CONFIG"])
                        print("Source hashes/map, packages, GPU and blackout forward checks passed. No training or synthetic cache generation.", flush=True)
                        break
                    print(f"Executing notebook cell {index} ...", flush=True)
                    exec(compile("".join(cell["source"]), f"{notebook.name}:cell-{index}", "exec"), namespace)
                    if index == 1:
                        config = namespace["CONFIG"]
                        if resume_saved is not None:
                            detected_paths = config["paths"]
                            config = copy.deepcopy(resume_saved["config"])
                            config["paths"]["repo_root"] = str(root)
                            config["paths"]["output_root"] = str(resume_path.parent.parent)
                            if not Path(config["paths"]["cache_root"]).is_dir():
                                config["paths"]["cache_root"] = detected_paths["cache_root"]
                            namespace["CONFIG"] = config
                            namespace["OUTPUT_ROOT"] = Path(config["paths"]["output_root"])
                        if args.batch_id:
                            config["data"]["batch_id"] = args.batch_id
                        if args.epochs is not None:
                            config["training"]["epochs"] = args.epochs
                            if resume_saved is None:
                                config["scheduler"]["t_max_epochs"] = max(args.epochs, config["scheduler"]["t_max_epochs"])
                        if args.batch_size is not None:
                            config["training"]["batch_size"] = args.batch_size
                        import torch
                        if args.amp_dtype == "auto" and resume_saved is None:
                            config["runtime"]["amp_dtype"] = "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp16"
                        elif args.amp_dtype != "auto":
                            config["runtime"]["amp_dtype"] = args.amp_dtype
                        namespace["validate_config"](config)
                        if resume_saved is not None:
                            validate_resume(resume_saved, config, resume_saved["split_ids"], resume_saved["source_metadata"])
                        if args.prepare_data:
                            episode_root, audit = prepare(root, run_root, config["data"]["batch_id"])
                            os.environ["ARES_EPISODE_ROOT"], os.environ["ARES_EXPORT_AUDIT"] = str(episode_root), str(audit)
                        (launch/"effective_config.json").write_text(json.dumps(config, indent=2)+"\n", encoding="utf-8")
                        print("Epochs:", config["training"]["epochs"], "| batch size:", config["training"]["batch_size"],
                              "| AMP:", config["runtime"]["amp_dtype"], flush=True)
                    if namespace.get("TRAINING_PAUSED"):
                        status["status"] = "paused"
                        break
                status.update(status="checked" if args.check_only else "paused" if namespace.get("TRAINING_PAUSED") else "completed",
                              training_run_dir=str(namespace["run_dir"]) if "run_dir" in namespace else None)
            except BaseException as exc:
                status.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                              error_type=type(exc).__name__, error=str(exc))
                traceback.print_exc()
                raise
            finally:
                for name in ("train_dataset", "val_dataset", "test_dataset"):
                    if name in namespace:
                        namespace[name].close()
                save_status()
    print("Finished:", status["status"], "| launch log:", launch/"console.log", flush=True)
    return launch


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(os.environ.get("ARES_REPO_ROOT", ROOT)))
    parser.add_argument("--run-root", type=Path, default=Path(os.environ.get("ARES_RUN_ROOT", ROOT)))
    parser.add_argument("--notebook", type=Path)
    parser.add_argument("--batch-id")
    parser.add_argument("--prepare-data", action="store_true", help="Reuse checked NPZs, or rebuild from raw MAT data if missing")
    parser.add_argument("--check-only", action="store_true", help="Check data, packages and GPU without training or synthetic generation")
    parser.add_argument("--epochs", type=int, help="Override notebook epochs for an initial timing run")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--amp-dtype", choices=("auto", "bf16", "fp16"), default="auto")
    parser.add_argument("--resume", nargs="?", const="auto", help="Continue latest run, or supply a run directory/checkpoint")
    parser.add_argument("--checkpoint-minutes", type=float, default=5., help="Save an in-epoch continuation every N minutes")
    parser.add_argument("--stop-after-minutes", type=float, help="Gracefully stop this session after N minutes")
    parser.add_argument("--walltime-margin-minutes", type=float, default=5., help="Stop this many minutes before Slurm expiry")
    parser.add_argument("--allow-local", action="store_true", help="Allow an intentional local run outside Slurm")
    args = parser.parse_args(argv)
    for name in ("epochs", "batch_size", "checkpoint_minutes", "stop_after_minutes", "walltime_margin_minutes"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main():
    args = parse_args()
    if not os.environ.get("SLURM_JOB_ID") and not args.allow_local:
        raise SystemExit("No Slurm allocation detected. Run salloc first, then run this launcher on the allocated node. Use --allow-local only for an intentional local check.")
    run_notebook(args)


if __name__ == "__main__":
    main()
