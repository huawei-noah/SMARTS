# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from collections import defaultdict
from glob import glob
from os import path
from pathlib import Path

from setuptools import find_packages, setup

this_dir = path.abspath(path.dirname(__file__))
with open(path.join(this_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


""" Modified setup.py to include option for changing SMARTS version or, by default,
the latest stable version SMARTS will used """
setup(
    name="ultra-rl",
    description="Unprotected Left Turn for Robust Agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.2.0",
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    zip_safe=True,
    python_requires=">=3.7",
    install_requires=[
        "smarts[train,test]==0.4.14",
        "setuptools>=41.0.0,!=50.0",
        "dill",
        "black==20.8b1",
        "ray[rllib]==1.0.1.post1",
    ],
)
