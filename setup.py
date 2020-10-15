from setuptools import setup, find_packages


setup(
    name="smarts",
    description="Scalable Multi-Agent RL Training School",
    version="0.4.1",
    packages=find_packages(exclude="tests"),
    include_package_data=True,
    zip_safe=True,
    python_requires=">=3.7",
    install_requires=[
        # setuptools:
        #   tensorboard needs >=41
        #   50.0 is broken: https://github.com/pypa/setupatools/issues/2353
        "setuptools>=41.0.0,!=50.0",
        "gym",
        "panda3d",
        "panda3d-gltf",
        "numpy",
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
        "rich",
        "supervisor",
        # HACK: There is a bug where if we only install the base ray dependency here
        #       and ray[rllib] under [train] it  prevents rllib from getting installed.
        #       For simplicity we just install both here. In the future we may want to
        #       address this bug head on to keep our SMARTS base install more lean.
        "ray[rllib]",  # We use Ray for our multiprocessing needs
        # The following are for Scenario Studio
        "yattag",
        # The following are for testing
        "pytest",
        "pytest-benchmark",
        "pytest-xdist",
        "pytest-cov",
        # The following are for /envision
        "tornado",
        "websocket-client",
        "cloudpickle<1.4.0",
        # The following are for the /smarts/algorithms
        "matplotlib",
        "scikit-image",
        # The following are for /smarts/zoo
        "twisted",
    ],
    extras_require={
        "train": ["tensorflow==1.15", "torch==1.3.0", "torchvision==0.4.1"],
        "dev": ["black", "sphinx", "sphinx-rtd-theme", "sphinxcontrib-apidoc",],
    },
    entry_points={"console_scripts": ["scl=cli.cli:scl"]},
)
