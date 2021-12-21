from os import path
from setuptools import setup, find_packages

this_dir = path.abspath(path.dirname(__file__))
with open(
    path.join(this_dir, "utils", "setup", "README.pypi.md"), encoding="utf-8"
) as f:
    long_description = f.read()

setup(
    name="smarts",
    description="Scalable Multi-Agent RL Training School",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.4.18",
    packages=find_packages(exclude=("tests", "examples")),
    include_package_data=True,
    zip_safe=True,
    python_requires=">=3.7",
    install_requires=[
        # setuptools:
        #   tensorboard needs >=41
        #   50.0 is broken: https://github.com/pypa/setupatools/issues/2353
        "setuptools>=41.0.0,!=50.0",
        "cached-property>=1.5.2",
        "click>=8.0.3",  # used in scl
        "gym==0.18.3",
        "numpy>=1.19.5",
        "pandas>=1.3.4",
        "psutil>=5.8.0",
        "pybullet==3.0.6",
        "pynput>=1.7.4",  # Used by HumanKeyboardAgent
        "rich>=10.13.0",
        "rtree>=0.9.7",  # Used by sumolib
        "sh>=1.14.2",
        "shapely>=1.8.0",
        "scikit-learn>=1.0.1",  # KDTree from scikit-learn is used by sumo lanepoints
        "tableprint>=0.9.1",
        "trimesh==3.9.29",  # Used for writing .glb files
        "visdom>=0.1.8.9",
        # The following are for Scenario Studio
        "yattag>=1.14.0",
        # The following is for both SS and Envision
        "cloudpickle>=1.3.0,<1.4.0",
        # The following are for /envision
        "tornado>=6.1",
        "websocket-client>=1.2.1",
        # The following is used for imitation learning and envision
        "ijson>=3.1.4",
        # The following are for the /smarts/algorithms
        "matplotlib>=3.4.3",
        "scikit-image>=0.18.3",
        # The following are for /smarts/zoo and remote agents
        "grpcio==1.32.0",
        "protobuf>=3.19.1",
        "PyYAML>=6.0",
        "twisted>=21.7.0",
    ],
    extras_require={
        "test": [
            # The following are for testing
            "ipykernel>=6.5.0",
            "jupyter-client==6.1.12",
            "pytest>=6.2.5",
            "pytest-benchmark>=3.4.1",
            "pytest-cov>=3.0.0",
            "pytest-notebook>=0.6.1",
            "pytest-xdist>=2.4.0",
            "ray[rllib]==1.0.1.post1",  # We use Ray for our multiprocessing needs
            "tensorflow>=2.4.0",  # For rllib tests
        ],
        "train": [
            "ray[rllib]==1.0.1.post1",  # We use Ray for our multiprocessing needs
            # XXX: TF requires specific version of scipy
            "scipy==1.4.1",
            "tensorflow>=2.4.0",
            "torch==1.4.0",
            "torchvision==0.5.0",
        ],
        "dev": [
            "black==20.8b1",
            "grpcio-tools==1.32.0",
            "isort==5.7.0",
            "sphinx",
            "sphinx-rtd-theme",
            "sphinxcontrib-apidoc",
        ],
        "camera-obs": [
            "Panda3D==1.10.9",
            "panda3d-gltf==0.13",
        ],
        "ros": [
            "catkin_pkg",
            "rospkg",
        ],
        "waymo": [
            "waymo-open-dataset-tf-2-2-0",
        ],
    },
    entry_points={"console_scripts": ["scl=cli.cli:scl"]},
)
