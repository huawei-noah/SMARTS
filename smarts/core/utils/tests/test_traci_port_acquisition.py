# MIT License
#
# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
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
from __future__ import annotations

import logging
import multiprocessing
import os
import random
import time
from multiprocessing.pool import AsyncResult
from typing import List, Tuple

from smarts.core import config
from smarts.core.utils.file import make_dir_in_smarts_log_dir
from smarts.core.utils.sumo_utils import RemoteSumoProcess, TraciConn

load_params = [
    "--net-file=%s" % "./scenarios/sumo/loop/map.net.xml",
    "--quit-on-end",
    "--no-step-log",
    "--no-warnings=1",
    "--seed=%s" % random.randint(0, 2147483648),
    "--time-to-teleport=%s" % -1,
    "--collision.check-junctions=true",
    "--collision.action=none",
    "--lanechange.duration=3.0",
    # TODO: 1--lanechange.duration1 or 1--lateral-resolution`, in combination with `route_id`,
    # causes lane change crashes as of SUMO 1.6.0.
    # Controlling vehicles that have been added to the simulation with a route causes
    # lane change related crashes.
    # "--lateral-resolution=100",  # smooth lane changes
    "--step-length=%f" % 0.1,
    "--default.action-step-length=%f" % 0.1,
    "--begin=0",  # start simulation at time=0
    "--end=31536000",  # keep the simulation running for a year
    "--start",
]

MAX_PARALLEL = 32
ITERATIONS = 60000  # 64512 ports available by Ubuntu standard
LOGGING_STEP = 1000


def run_client(t):
    conn = None
    try:
        f = os.path.abspath(make_dir_in_smarts_log_dir("_sumo_run_logs")) + f"/{t}"
        lsp = RemoteSumoProcess(
            remote_host=config()("sumo", "central_host"),
            remote_port=config()("sumo", "central_port", cast=int),
        )
        lsp.generate(
            base_params=load_params
            + [
                "--log=%s.log" % f,
                "--message-log=%s" % f,
                "--error-log=%s.err" % f,
            ],
            sumo_binary="sumo",
        )
        conn = TraciConn(
            sumo_process=lsp,
            name=f"Client@{t}",
        )
        conn.connect(
            timeout=5,
            minimum_traci_version=20,
            minimum_sumo_version=(1, 10, 0),
        )
        time.sleep(0.1)
        conn.getVersion()
    except KeyboardInterrupt:
        if conn is not None:
            conn.close_traci_and_pipes(False)
        raise
    except Exception as err:
        logging.error("Primary occurred. [%s]", err)
        logging.exception(err)
        raise
    finally:
        # try:
        #     conn.close_traci_and_pipes()
        # except Exception as err:
        #     logging.error("Secondary occurred. [%s]", err)
        diff = time.time() - t
        if diff > 9:
            logging.error("Client took %ss to close", diff)
        if conn is not None:
            conn.teardown()


def test_traffic_sim_with_multi_client():
    with multiprocessing.Pool(processes=MAX_PARALLEL) as pool:
        clients: List[Tuple[AsyncResult, float]] = []
        start = time.time()
        # Attempt to run out of ports.
        for i in range(ITERATIONS):
            while len(clients) > MAX_PARALLEL:
                for j, (c, t) in reversed(
                    [(j, (c, t)) for j, (c, t) in enumerate(clients) if c.ready()]
                ):
                    clients.pop(j)
            current = time.time()
            if i % LOGGING_STEP == 0:
                logging.error("Working on %s at %ss", i, current - start)
            clients.append((pool.apply_async(run_client, args=(current,)), current))

        for j, (c, t) in reversed(
            [(j, (c, t)) for j, (c, t) in enumerate(clients) if c.ready()]
        ):
            clients.pop(j)
            logging.error("Popping remaining ready clients %s", t)

        for (c, t) in clients:
            if time.time() - t > 0.2:
                logging.error("Stuck clients %s", t)

        logging.error("Finished")
        pool.close()
        logging.error("Closed")
        pool.join()
        logging.error("Joined")
