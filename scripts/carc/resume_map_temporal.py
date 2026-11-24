#!/usr/bin/env python3
"""Continue your newest saved map-temporal run in a fresh salloc allocation.

Defaults to /scratch1/<username>/ares-training when that directory exists;
ARES_RUN_ROOT or --run-root may select another location. Accepts the same options
as run_map_temporal.py. Use --resume /path/to/run to choose a particular run.
"""
import getpass
import os
from pathlib import Path
import sys

from run_map_temporal import main


if __name__ == "__main__":
    scratch = Path("/scratch1")/getpass.getuser()/"ares-training"
    if "ARES_RUN_ROOT" not in os.environ and scratch.is_dir():
        os.environ["ARES_RUN_ROOT"] = str(scratch)
    if not any(arg == "--resume" or arg.startswith("--resume=") for arg in sys.argv[1:]):
        sys.argv.append("--resume")
    main()
