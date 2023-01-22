# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

try:
    from gymnasium.envs.registration import register

    register(
        id="hiway-v1",
        entry_point="smarts.env.gymnasium.hiway_env_v1:HiWayEnvV1",
        disable_env_checker=True,
    )

    register(
        id="driving-smarts-competition-v0",
        entry_point="smarts.env.gymnasium.driving_smarts_competition_env:driving_smarts_competition_v0_env",
        disable_env_checker=True,
    )
except ModuleNotFoundError:
    import warnings

    warnings.warn(
        "Gymnasium cannot be imported likely due to numpy version compatibility `numpy>=1.21.0`. "
        "Gymnasium environments will be unavailable. Gymnasium imports may cause a crash."
    )
