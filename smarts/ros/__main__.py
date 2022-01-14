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

import os
import subprocess
import sys


def _usage_error(msg: str = None):
    if msg:
        print(f"ERROR:  {msg}")
    print("usage:  python -m smarts.ros setup_node [install_path]")
    print(
        "\twhere install_path is an optional existing folder into which to put the node"
    )
    sys.exit(-1)


if __name__ == "__main__":
    if not 2 <= len(sys.argv) <= 3 or sys.argv[1] != "setup_node":
        _usage_error()

    # Note:  use of this script may require installing the SMARTS package with the "[ros]" extensions.

    mod_path = os.path.dirname(__file__)

    install_arg = ""
    install_path = f"{mod_path}/install"
    if len(sys.argv) >= 3:
        install_path = sys.argv[2]
        if not os.path.isdir(install_path):
            _usage_error(f"install_path ({install_path}) does not exist.")
        install_arg = f"-DCMAKE_INSTALL_PREFIX={install_path}"

    source_ros = ""
    if (
        os.environ.get("ROS_VERSION", 0) != 1
        or os.environ.get("ROS_DISTRO", "a")[0] < "k"
    ):
        ros_base = "/opt/ros"
        for distro in ["noetic", "melodic", "lunar", "kinetic"]:
            ros_distro = os.path.join(ros_base, distro)
            if os.path.isdir(ros_distro):
                break
        else:
            print(
                "cannot find appropriate ROS v1 distribution.  SMARTS requires kinetic or newer."
            )
            sys.exit()
        source_ros = f". {ros_distro}/setup.sh && "
    print("setting up ROS node for SMARTS...")
    # need to override path because this is likely to be being run in a venv,
    # but catkin_make for ROS v1 distro should be done with python=python2 not python3.
    default_path = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    subprocess.check_call(
        f"{source_ros}catkin_make {install_arg} install",
        shell=True,
        cwd=mod_path,
        env={"PATH": default_path},
    )
    print(f"\nnow run:  source {install_path}/setup.bash")
