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
        "cached-property",
        "click",  # used in scl
        "gym==0.18.3",
        "numpy",
        "pandas",
        "psutil",
        "pybullet==3.0.6",
        "pynput",  # Used by HumanKeyboardAgent
        "rich",
        "rtree",  # Used by sumolib
        "sh",
        "shapely",
        "sklearn",  # KDTree from sklearn is used by sumo lanepoints
        "tableprint",
        "trimesh==3.9.29",  # Used for writing .glb files
        "visdom",
        # The following are for Scenario Studio
        "yattag",
        # The following are for /envision
        "cloudpickle<1.4.0",
        "tornado",
        "websocket-client",
        # The following is used for imitation learning and envision
        "ijson",
        # The following are for the /smarts/algorithms
        "matplotlib",
        "scikit-image",
        # The following are for /smarts/zoo and remote agents
        "grpcio==1.32.0",
        "protobuf",
        "PyYAML",
        "twisted",
    ],
    extras_require={
        "test": [
            # The following are for testing
            "ipykernel",
            "jupyter-client==6.1.12",
            "pytest",
            "pytest-benchmark",
            "pytest-cov",
            "pytest-notebook",
            "pytest-xdist",
            "ray[rllib]==1.0.1.post1",  # We use Ray for our multiprocessing needs
            "tensorflow==2.2.1",  # For rllib tests
        ],
        "train": [
            "ray[rllib]==1.0.1.post1",  # We use Ray for our multiprocessing needs
            # XXX: TF requires specific version of scipy
            "scipy==1.4.1",
            "tensorflow==2.2.1",
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
