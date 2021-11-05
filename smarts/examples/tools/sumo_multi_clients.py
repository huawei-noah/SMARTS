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
import random
import subprocess
import threading
import time

from smarts.core.utils.sumo import SUMO_PATH, sumolib, traci

PORT = 8001

"""
Conclusions:
1. connected clients < num-clients: SUMO will block, only start once all clients have connected.
2. connected clients > num-clients: Extra connection will be closed by SUMO.
3. The simulation does not advance to the next step until all clients have called the 'simulationStep' command.
4. For multi client scenarios currently only TargetTime 0 is supported, which means 'simulationStep' performs exactly one time step.
"""


def start_sumo_server():
    sumo_binary = "sumo"
    sumo_cmd = [
        os.path.join(SUMO_PATH, "bin", sumo_binary),
        "--net-file=scenarios/loop/map.net.xml",
        "--num-clients=3",
        "--remote-port=%s" % PORT,
    ]
    sumo_proc = subprocess.Popen(
        sumo_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    time.sleep(0.1)
    traci_conn = traci.connect(
        PORT, numRetries=100, proc=sumo_proc, waitBetweenRetries=0.01
    )
    return traci_conn


def connect(port, order=None):
    traci_conn = traci.connect(port, numRetries=100, proc=None, waitBetweenRetries=0.1)
    if order is not None:
        traci_conn.setOrder(order)
    return traci_conn


def test_client_connection(client, client_name):
    for i in range(10):
        print(f"{client_name} steping simulation")
        client.simulationStep()

    client.close()


def init_client():
    client = start_sumo_server()
    client.setOrder(1)

    test_client_connection(client, "client 1")


def run_client_2():
    client2 = connect(PORT, 2)

    test_client_connection(client2, "client 2")


def run_client_3():
    client3 = connect(PORT, 3)

    test_client_connection(client3, "client 3")


def main():
    t1 = threading.Thread(target=init_client, args=())
    t1.start()

    t2 = threading.Thread(target=run_client_2, args=())
    t2.start()

    t3 = threading.Thread(target=run_client_3, args=())
    t3.start()

    t1.join()
    t2.join()
    t3.join()


if __name__ == "__main__":
    main()
