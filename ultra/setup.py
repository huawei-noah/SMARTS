from setuptools import setup, find_packages
from glob import glob
from pathlib import Path
from collections import defaultdict

# TODO rename to ULTRA_baselines

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
        "dill",
        "black==20.8b1",
    ],
    dependency_links=[
        # "smarts@git+https://gitlab.smartsai.xyz/smarts/SMARTS/-/releases/v0.4.1"
    ],
)
