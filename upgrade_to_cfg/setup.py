from setuptools import setup
from os import path
from setuptools import setup, find_packages

this_dir = path.abspath(path.dirname(__file__))
with open(
    path.join(this_dir, "utils", "setup", "README.pypi.md"), encoding="utf-8"
) as f:
    long_description = f.read()

setup(
    long_description=long_description,
    long_description_content_type="text/markdown",
)
