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

from setuptools import setup, find_packages
from glob import glob
from pathlib import Path
from collections import defaultdict

""" Modified setup.py to include options for changing SMARTS version or set default
to the latest stable version of SMARTS """
setup(
    name="ultra",
    description="Unprotected Left Turn using Reinforcement-learning Agents",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=True,
    python_requires=">=3.7",
    install_requires=[
        "smarts[train]==0.4.6",  # Stable version
        "setuptools>=41.0.0,!=50.0",
        "ijson",
        "dill",
        "black==19.10b0",
    ],
)
