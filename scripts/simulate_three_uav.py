"""Run the three-UAV closed loop without setting PYTHONPATH."""

from pathlib import Path
import sys

PACKAGE_PARENT = Path(__file__).resolve().parents[1] / "ros2" / "src" / "ares_three_uav"
sys.path.insert(0, str(PACKAGE_PARENT))

from ares_three_uav.simulate import main

if __name__ == "__main__":
    main()
