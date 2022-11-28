# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
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


class RendererException(Exception):
    """An exception raised if a renderer is required but not available."""

    @classmethod
    def required_to(cls, thing: str) -> "RendererException":
        """Generate a `RenderException` requiring a render to do `thing`."""
        return cls(
            f"""A renderer is required to {thing}. You may not have installed the [camera_obs] dependencies required to render the camera sensor observations. Install them first using the command `pip install -e .[camera_obs]` at the source directory."""
        )


class RayException(Exception):
    """An exception raised if ray package is required but not available."""

    @classmethod
    def required_to(cls, thing):
        """Generate a `RayException` requiring a render to do `thing`."""
        return cls(
            f"""Ray Package is required to {thing}.
               You may not have installed the [rllib] or [train] dependencies required to run the ray dependent example.
               Install them first using the command `pip install -e .[train, rllib]` at the source directory to install the package ray[rllib]==1.0.1.post1"""
        )


class OpenDriveException(Exception):
    """An exception raised if opendrive utilities are required but not available."""

    @classmethod
    def required_to(cls, thing):
        """Generate an instance of this exception that describes what can be done to remove the exception"""
        return cls(
            f"""OpenDRIVE Package is required to {thing}.
               You may not have installed the [opendrive] dependencies required to run the OpenDRIVE dependent example.
               Install them first using the command `pip install -e .[opendrive]` at the source directory to install the necessary packages"""
        )
