# MIT License

# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import argparse
import cloudpickle
import importlib
import logging
import os
import signal

from smarts.proto import worker_pb2
from smarts.rpc import worker as learner_agent

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(f"learner.py - pid({os.getpid()})")


def learner_loop(addr, agent_specs):

    # Spawn a grpc server to host the learner
    conn = learner_agent.spawn(learner_agent.serve, addr[1])
    conn.stub.build(worker_pb2.Specification(payload=cloudpickle.dumps(agent_specs)))

    def stop_learner(unused_signum, unused_frame):
        log.debug(f"learner.py - pid({os.getpid()}): Learner stopped.")

    # Catch keyboard interrupt signal
    signal.signal(signal.SIGINT, stop_learner)


def main(config: dict):
    assert len(config["agent_policies"]) == len(
        config["agent_ids"]
    ), "Length of `agent_policies` does not match that of `agent_ids`."

    agent_specs = {
        agent_id: importlib.import_module(
            "zoo.policies." + agent_policy + ".agent"
        ).create_agent_spec()
        for agent_id, agent_policy in zip(config["agent_ids"], config["agent_policies"])
    }

    learner_loop((config["learner_address"], config["learner_port"]), agent_specs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("distributed-learner")
    parser.add_argument(
        "--agent_policies",
        help="Learner agent policies to use. In multi-agent simulation, agents may have different policies.",
        type=str,
        nargs="+",
        default=["keep_lane", "keep_lane", "keep_lane", "keep_lane"],
    )
    parser.add_argument(
        "--agent_ids",
        help="List of string, representing agent names. Length of `agent_ids` should match that of `agent_policies`.",
        type=str,
        nargs="+",
        default=["Agent_001", "Agent_002", "Agent_003", "Agent_004"],
    )
    parser.add_argument(
        "--learner_address",
        help="Server IP address in which the learner (i.e., RL ego agent) runs.",
        type=str,
        default="localhost",
    )
    parser.add_argument(
        "--learner_port",
        help="Server port at which the learner (i.e., RL ego agent) runs.",
        type=int,
        default=6001,
    )
    parser.add_argument(
        "--max_episode_steps",
        help="Maximum number of steps per episode.",
        type=int,
        default=None,
    )
    config = parser.parse_args()

    main(vars(config))
