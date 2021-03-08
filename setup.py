from os import path
from setuptools import setup, find_packages

this_dir = path.abspath(path.dirname(__file__))
with open(path.join(this_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="smarts",
    description="Scalable Multi-Agent RL Training School",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.4.14",
    packages=find_packages(exclude="tests"),
    include_package_data=True,
    zip_safe=True,
    python_requires=">=3.7",
    install_requires=[
        # setuptools:
        #   tensorboard needs >=41
        #   50.0 is broken: https://github.com/pypa/setupatools/issues/2353
        "setuptools>=41.0.0,!=50.0",
        "click",  # used in scl
        "gym",
        "panda3d",
        "panda3d-gltf",
        "numpy",
        "rich",
        "rtree",  # Used by sumolib
        "filelock",
        "lz4",
        "networkx",
        "opencv-python",
        "pandas",
        "psutil",
        "visdom",
        "pybullet",
        "sklearn",  # KDTree from sklearn is used by waypoints
        "tableprint",
        "trimesh",  # Used for writing .glb files
        "pynput",  # Used by HumanKeyboardAgent
        "sh",
        "shapely",
        "supervisor",
        # HACK: There is a bug where if we only install the base ray dependency here
        #       and ray[rllib] under [train] it  prevents rllib from getting installed.
        #       For simplicity we just install both here. In the future we may want to
        #       address this bug head on to keep our SMARTS base install more lean.
        "ray[rllib]==1.0.1.post1",  # We use Ray for our multiprocessing needs
        # The following are for Scenario Studio
        "yattag",
        # The following are for /envision
        "cloudpickle<1.4.0",
        "tornado",
        "websocket-client",
        # The following are for the /smarts/algorithms
        "matplotlib",
        "scikit-image",
        # The following are for /smarts/zoo
        "grpcio==1.30.0",
        "PyYAML",
        "twisted",
    ],
    extras_require={
        "test": [
            # The following are for testing
            "pytest",
            "pytest-benchmark",
            "pytest-cov",
            "pytest-notebook",
            "pytest-xdist",
        ],
        "train": [
            "tensorflow==2.2.1",
            # XXX: TF requires specific version of scipy
            "scipy==1.4.1",
            "torch==1.4.0",
            "torchvision==0.5.0",
        ],
        "dev": [
            "black==20.8b1",
            "grpcio-tools==1.30.0",
            "isort==5.7.0",
            "sphinx",
            "sphinx-rtd-theme",
            "sphinxcontrib-apidoc",
        ],
    },
    entry_points={"console_scripts": ["scl=cli.cli:scl"]},
)
