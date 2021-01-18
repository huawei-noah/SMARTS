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
import ray
from packaging import version


def default_ray_kwargs(**kwargs):
    ray_kwargs = {}
    _system_config = {
        # Needed to deal with cores that lock up for > 10 seconds
        "num_heartbeats_timeout": 10000,
        # "raylet_heartbeat_timeout_milliseconds": 10,
        # "object_timeout_milliseconds": 200,
    }
    if version.parse(ray.__version__) > version.parse("0.8"):
        ray_kwargs["_system_config"] = _system_config
    else:
        ray_kwargs["_internal_config"] = _system_config
    ray_kwargs.update(kwargs)

    return ray_kwargs


# TODO: Perhaps start cluster manually instead of `ray.init()`
# See https://github.com/ray-project/ray/blob/174bef56d452b6f86db167ecb80e7f23176079b6/python/ray/tests/conftest.py#L110
