from os import path

from setuptools import find_packages, setup

this_dir = path.abspath(path.dirname(__file__))
with open(path.join(this_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="sb3",
    description="Driving in traffic using SB3",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=True,
    python_requires=">=3.7",
    install_requires=[
        "setuptools>=41.0.0,!=50.0",
        "smarts[camera-obs] @ git+https://git@github.com/huawei-noah/SMARTS@develop",
        "stable-baselines3[extra]==1.4.0",
        "tensorflow==2.4.0",
        "ruamel.yaml==0.17.17",
        "ruamel.yaml.clib==0.2.6",
    ],
)
