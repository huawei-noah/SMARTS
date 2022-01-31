from os import path

from setuptools import find_packages, setup

this_dir = path.abspath(path.dirname(__file__))
with open(path.join(this_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="driving-in-traffic",
    description="Driving in traffic using DreamerV2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=True,
    python_requires=">=3.7",
    install_requires=[
        "setuptools>=41.0.0,!=50.0",
        "smarts[camera-obs]==0.5.1",
        "dreamerv2==2.1.1",
        "tensorflow==2.4.0",
        "tensorflow-probability==0.12.2",
    ],
)
