import os
from glob import glob

from setuptools import setup

package_name = "ares_three_uav"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "worlds"), glob("worlds/*")),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="ARES",
    maintainer_email="ares@example.com",
    description="Three-UAV Gazebo/ROS 2 port of the ARES single-UAV flight stack.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "ares_three_uav_sim = ares_three_uav.simulate:main",
            "ares_three_uav_node = ares_three_uav.ros_node:main",
        ],
    },
)
