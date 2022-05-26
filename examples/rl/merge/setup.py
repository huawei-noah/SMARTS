from os import path

from setuptools import find_packages, setup

this_dir = path.abspath(path.dirname(__file__))
with open(path.join(this_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="merge",
    description="Merging into a freeway.",
    # long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=True,
    python_requires=">=3.7",
    install_requires=[
        "setuptools>=41.0.0,!=50.0",
        "protobuf==3.19.4",
        "ruamel.yaml==0.17.17",
        "smarts[camera-obs] @ git+https://github.com/huawei-noah/SMARTS.git@sb3-3",
        "tensorflow==2.8.0",
        "tensorflow-probability==0.16.0",
        "tf-agents[reverb]==0.12.1",
    ],
)
