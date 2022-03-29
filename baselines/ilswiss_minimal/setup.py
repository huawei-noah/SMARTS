from setuptools import find_packages, setup

setup(
    name="ilswiss_minimal",
    description="Minimal Version of ILSwiss Designed for SMARTS",
    version="0.1.0",
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    zip_safe=True,
    python_requires=">=3.6",
    install_requires=[
        "smarts[train]",
        "black==20.8b1",
        "gtimer",
        "gym",
    ],
)
