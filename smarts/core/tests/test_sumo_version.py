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
import os
import subprocess


def test_sumo_lib():
    # import does runtime check by necessity
    from smarts.core.utils.sumo import sumolib


def test_sumo_version():
    from smarts.core.utils import networking
    from smarts.core.utils.sumo import SUMO_PATH, traci

    load_params = [
        "--start",
        "--quit-on-end",
        "--net-file=scenarios/loop/map.net.xml",
        "--no-step-log",
        "--no-warnings=1",
    ]

    sumo_port = networking.find_free_port()
    sumo_cmd = [
        os.path.join(SUMO_PATH, "bin", "sumo"),
        "--remote-port=%s" % sumo_port,
        *load_params,
    ]

    sumo_proc = subprocess.Popen(
        sumo_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    traci_conn = traci.connect(
        sumo_port, numRetries=10, proc=sumo_proc, waitBetweenRetries=0.1
    )

    assert (
        traci_conn.getVersion()[0] >= 20
    ), "TraCI API version must be >= 20 (SUMO 1.5.0)"
