"""Run the real-data two-stage, GPU-oriented augmented SAITS/BarrierNet pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ares_safe_training.config import PipelineConfig
from ares_safe_training.pipeline import run_pipeline


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True,
                        help="Validated calibration.json with sibling fixtures.mat and validation.json.")
    parser.add_argument("--config", type=Path, default=root / "configs" / "augmented_saits_barrier.json")
    parser.add_argument("--stage", choices=("all", "pretrain", "finetune", "evaluate"), default="all")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = PipelineConfig.load(args.config)
    if args.device:
        config.device = args.device
    result = run_pipeline(args.repo_root, args.output_dir, args.calibration, config,
                          stage=args.stage, resume=args.resume)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
