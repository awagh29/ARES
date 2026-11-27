"""Submit one dependent allocation after a saved training time-limit pause."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess


def submit_continuation(launch_dir, script):
    """Use this launch's status and explicit run folder, never the latest run."""
    launch_dir, script = Path(launch_dir), Path(script).expanduser().resolve()
    status_path = launch_dir/"status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    previous = status.get("continuation", {})
    if previous.get("status") == "submitted":
        return previous["job_id"]
    if status["status"] != "paused":
        print("No continuation needed; launch status:", status["status"], flush=True)
        return None

    run_dir = Path(status["training_run_dir"]).resolve()
    summary = json.loads((run_dir/"summary.json").read_text(encoding="utf-8"))
    target = status["target_epochs"]
    if summary["status"] != "paused":
        raise RuntimeError("Run summary does not confirm a saved pause; continuation not submitted")
    if summary["completed_epochs"] >= target:
        print(f"Reached {target} total epochs; no continuation needed.", flush=True)
        return None
    # A user cancellation must not restart training. USR1 is the Slurm warning.
    warning = getattr(signal, "SIGUSR1", None)
    allowed_reasons = {"time budget reached"}
    if warning is not None:
        allowed_reasons.add(f"signal {int(warning)}")
    if summary.get("stop_reason") not in allowed_reasons:
        print("No automatic continuation for stop reason:", summary.get("stop_reason"), flush=True)
        return None
    progress = (summary["completed_epochs"], summary["next_batch"])
    if status.get("resume_progress") is not None and progress <= tuple(status["resume_progress"]):
        raise RuntimeError("Training made no checkpoint progress; continuation not submitted")
    if not (run_dir/"last.pt").is_file():
        raise FileNotFoundError(f"Saved continuation checkpoint missing: {run_dir/'last.pt'}")
    if not script.is_file():
        raise FileNotFoundError(f"Continuation batch script missing: {script}")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id.isdigit() or status["job_id"] != job_id:
        raise RuntimeError("Continuation submission requires the original active Slurm job")

    command = ["sbatch", "--parsable", f"--dependency=afterok:{job_id}",
               "--kill-on-invalid-dep=yes", f"--chdir={status['repo_root']}"]
    for variable, option in (("SLURM_JOB_ACCOUNT", "account"), ("SLURM_JOB_QOS", "qos")):
        if os.environ.get(variable):
            command.append(f"--{option}={os.environ[variable]}")
    command += [str(script), str(run_dir), str(target)]

    def persist():
        temporary = status_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(status, indent=2)+"\n", encoding="utf-8")
        temporary.replace(status_path)

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        status["continuation"] = dict(status="failed", error=detail, command=command)
        persist()
        raise RuntimeError(f"Continuation submission failed; checkpoint is saved in {run_dir}: {detail}") from exc
    successor = result.stdout.strip()
    status["continuation"] = dict(status="submitted", job_id=successor,
        parent_job_id=job_id, run_dir=str(run_dir), target_epochs=target, command=command)
    persist()
    print(f"Submitted continuation job {successor} for {run_dir} toward {target} total epochs.", flush=True)
    return successor
