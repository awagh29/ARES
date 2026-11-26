#!/usr/bin/env python3
"""Replay finished runs smoothly in Gazebo and save MP4 videos.

  python3 scripts/replay.py results/first_flight/S1_intermittent/seed0513_xxxx          # one run -> videos
  python3 scripts/replay.py results/single_drone_map-temporal-cnn                     # every run in a campaign
  python3 scripts/replay.py <run folder> --watch                                       # just watch in the Gazebo GUI
  python3 scripts/replay.py <run folder> --views overview --speed 2                    # one view, 2x speed

Videos are written to each run folder (replay_overview.mp4, replay_top.mp4) and
copied to <campaign>/videos/. Needs ffmpeg in WSL: sudo apt install -y ffmpeg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ares_gz.replay import VIEWS, organize_videos, render_replay, watch_replay  # noqa: E402


def run_folders(path: Path) -> list[Path]:
    if (path / "control.csv").exists():
        return [path]
    return sorted(folder.parent for folder in path.rglob("control.csv"))


def campaign_folder(run: Path) -> Path:
    """The campaign folder: the run's ancestor named in its manifest (old runs had no start level)."""
    import json
    name = json.loads((run / "manifest.json").read_text()).get("campaign")
    for parent in run.parents:
        if parent.name == name:
            return parent
    return run.parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="a run folder or a campaign folder")
    parser.add_argument("--views", nargs="+", choices=sorted(VIEWS), default=["overview", "top"])
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed factor (2 = twice as fast)")
    parser.add_argument("--watch", action="store_true", help="open the Gazebo GUI instead of recording")
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--organize", action="store_true",
                        help="only sort the campaign's existing videos/ into per-experiment folders")
    args = parser.parse_args()

    if args.organize:
        videos = args.path / "videos" if (args.path / "videos").is_dir() else args.path
        moved = organize_videos(videos, campaign_dir=videos.parent)
        print(f"Moved {moved} video(s) into <experiment>/<PASSED|FAILED> folders under {videos}")
        return 0

    folders = run_folders(args.path.resolve())
    if not folders:
        parser.error(f"no runs (control.csv) under {args.path}")
    if args.watch:
        watch_replay(folders[0], args.speed)
        return 0
    for folder in folders:
        render_replay(folder, args.views, args.speed, args.keep_frames, copy_to=campaign_folder(folder) / "videos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
