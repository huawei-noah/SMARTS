from os import path

from setuptools import find_packages, setup

this_dir = path.abspath(path.dirname(__file__))
with open(path.join(this_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="racing",
    description="Racing in traffic using DreamerV2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.1.1",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=True,
    python_requires="~=3.8",
    install_requires=[
        "setuptools>=41.0.0,!=50.0",
        "smarts[camera_obs]==1.0.0",
        "numpy<=1.23.0,>=1.19",
        "dreamerv2==2.2.0",
        "tensorflow~=2.7.0",
        "tensorflow-probability==0.12.2",
    ],
)
