from setuptools import setup, find_packages
from glob import glob
from pathlib import Path
from collections import defaultdict
#TODO rename to ULTRA_baselines

setup(
    name="ultra",
    description="Unprotected Left Turn using Reinforcement-learning Agents",
    version="0.1",
    packages=find_packages(exclude="tests"),
    include_package_data=True,
    zip_safe=True,
    python_requires=">=3.7",
    install_requires=[
        "smarts@git+https://github.com/huawei-noah/SMARTS.git",
        "setuptools>=41.0.0,!=50.0",
        "gym",
        "dill",
        "panda3d",
        "panda3d-gltf",
        "numpy",
        "torch",
        "qpsolvers",
        "cvxopt",
        "shapely",
        "networkx",
        "trimesh",  # Used for writing .glb files
        "rtree",  # Used by sumolib
        "lz4",
        "filelock",
        "pandas",
        "psutil",
        "opencv-python",
        "visdom",
        "pybullet",
        "sklearn",  # KDTree from sklearn is used by waypoints
        "tableprint",
        "pynput",  # Used by HumanKeyboardPolicy
        "sh",
        "ray[rllib]==1.0.0",  # We use Ray for our multiprocessing needs
        "yattag",
        "pytest>=6.0.0",
        "pytest-benchmark",
        "pytest-xdist",
        "pytest-cov",
        "tornado",
        "websocket-client",
        "cloudpickle<1.4.0",
        "matplotlib",
        "scikit-image",
        # The following are for evaluation
        "shyaml",
        "twisted",
        "black==20.8b1",
    ],
    dependency_links=[
        # "smarts@git+https://gitlab.smartsai.xyz/smarts/SMARTS/-/releases/v0.4.1"
    ],
)
