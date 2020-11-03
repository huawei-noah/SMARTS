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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""
Run an agent in it's own (independent) process.

What Agent code does is out of our direct control, we want to avoid any interactions with global state that might be present in the SMARTS process.

To protect and isolate Agents from any pollution of global state in the main SMARTS process, we spawn Agents in their fresh and independent python process.

This script is called from within SMARTS to instantiate a remote agent.
The protocal is as follows:

1. SMARTS Calls: run_agent.py /tmp/agent_007.sock # sets a unique path the domain socket per agent
2. run_agent.py will create the /tmp_agent_007.sock domain socket and begin listening
3. SMARTS connects to /tmp/agent_007.sock as a client
4. SMARTS sends the `AgentSpec` over the socket to run_agent.py
5. run_agent.py recvs the AgentSpec instances and builds the Agent
6. SMARTS sends observations and listens for actions
7. run_agent.py listens for observations and responds with actions
"""

import argparse
import importlib
import logging
import os
import time
from multiprocessing.connection import Listener

import cloudpickle

# front-load some expensive imports as to not block the simulation

modules = [
    "pybullet",
    "smarts.core.utils.sumo",
    "smarts.core.sumo_road_network",
    "numpy",
    "sklearn",
    "shapely",
    "scipy",
    "trimesh",
    "panda3d",
    "gym",
    "ray",
]

for mod in modules:
    try:
        importlib.import_module(mod)
    except ImportError:
        pass

# end front-loaded imports

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(f"PID({os.getpid()}) run_agent.py")

parser = argparse.ArgumentParser("Spawn an agent in it's own independent process")
parser.add_argument("socket_file", help="AF_UNIX domain socket file to be used for IPC")
parser.add_argument("--with_adaptation", action="store_true")
args = parser.parse_args()


log.debug(
    f"run_agent.py: with_adaptation={args.with_adaptation} socket_file={args.socket_file}"
)

with Listener(args.socket_file, family="AF_UNIX") as listener:
    with listener.accept() as conn:
        log.debug(f"connection accepted from {listener.last_accepted}")
        agent = None
        try:
            msg = conn.recv()
            assert (
                msg["type"] == "agent_spec"
            ), f"invalid initial msg type: {msg['type']}"
            # We use cloudpickle only for the agent_spec to allow for serialization of things like
            # lambdas or other non-pickleable things

            time_start = time.time()
            agent_spec = cloudpickle.loads(msg["payload"])
            pickle_load_time = time.time()
            agent = agent_spec.build_agent()
            agent_build_time = time.time()
            log.debug(
                "build agent timings:\n"
                f"  total ={agent_build_time - time_start:.2}\n"
                f"  pickle={pickle_load_time - time_start:.2}\n"
                f"  build ={agent_build_time - pickle_load_time:.2}\n"
            )

            while True:
                msg = conn.recv()
                if msg["type"] == "obs":
                    obs = msg["payload"]
                    if args.with_adaptation:
                        action = agent.act_with_adaptation(obs)
                    else:
                        action = agent.act(obs)
                    conn.send(action)
                else:
                    log.error(f"run_agent.py dropping malformed msg: {repr(msg)}")

        except (EOFError, ConnectionResetError, BrokenPipeError):
            # We treat the closing of the socket as a signal to terminate the process
            log.debug("Closed connection, terminating run_agent")
            pass

        del agent

log.debug("Exiting")
