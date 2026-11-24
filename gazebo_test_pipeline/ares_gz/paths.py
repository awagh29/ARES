"""Locations inside the ARES repository."""

from __future__ import annotations

from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_ROOT.parent
INITIALIZER = REPO_ROOT / "scripts" / "initializeSingleUav.m"
NAVIGATION_SOURCE = REPO_ROOT / "src" / "+ares" / "+navigation"
RESULTS_ROOT = PIPELINE_ROOT / "results"
FIXTURES_ROOT = PIPELINE_ROOT / "tests" / "fixtures"
