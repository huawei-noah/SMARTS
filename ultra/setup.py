from setuptools import setup, find_packages
from glob import glob
from pathlib import Path
from collections import defaultdict

''' Modified setup.py to include option for changing SMARTS version or by default
the latest stable version SMARTS will used'''
setup(
    name="ultra",
    description="Unprotected Left Turn using Reinforcement-learning Agents",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=True,
    python_requires=">=3.7",
    install_requires=[
        "smarts[train]==0.4.6", # Stable version 
        "setuptools>=41.0.0,!=50.0",
        "dill",
        "black==19.10b0",
    ],
)
