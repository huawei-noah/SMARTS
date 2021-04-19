import argparse
import cloudpickle
import importlib
import os
import signal

from examples import policies
from smarts.proto import worker_pb2
from smarts.rpc import worker as learner_agent


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
            "examples.policies." + agent_policy + ".agent"
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
