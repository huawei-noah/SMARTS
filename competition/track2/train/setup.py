from os import path

from setuptools import find_packages, setup

this_dir = path.abspath(path.dirname(__file__))
with open(path.join(this_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="offline_RL",
    description="Offline RL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=True,
    python_requires=">=3.7",
    install_requires=[
        "setuptools>=41.0.0,!=50.0",
        "protobuf==3.20.1",
        "smarts[camera-obs] @ git+https://github.com/huawei-noah/SMARTS.git@comp-1",
        "tensorboard==2.2.0",
        "d3rlpy==1.1.1",
        "Pillow==9.2.0",
        "PyYAML==6.0",
        "torch==1.11.0",
    ],
)
