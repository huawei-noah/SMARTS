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

import gym
import importlib

from examples import default_argument_parser
from smarts.core import action as act_util
from smarts.core import observation as obs_util
from smarts.core.utils.episodes import episodes
from smarts.proto import worker_pb2_grpc as learner_pb2_grpc
from smarts.proto import observation_pb2
from smarts.rpc import worker as learner_agent


def create_env(scenarios, agent_specs, sim_name, headless, seed):
    return gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs=agent_specs,
        sim_name=sim_name,
        headless=headless,
        seed=seed,
    )


def actor_loop(env, addr, num_episodes):

    # Setup rpc channel and stub of learner
    learner_chan = learner_agent.get_channel(addr)
    learner_stub = learner_pb2_grpc.WorkerStub(learner_chan)

    for episode in episodes(n=num_episodes):
        obs = env.reset()
        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        while not dones["__all__"]:

            # Send observations and receive actions for all agents
            learner_actions_proto = learner_stub.act(
                observation_pb2.ObservationsBoid(
                    boids=obs_util.observations_to_proto(obs),
                )
            )
            learner_actions = act_util.proto_to_actions(learner_actions_proto)
            obs, rewards, dones, infos = env.step(learner_actions)
            episode.record_step(obs, rewards, dones, infos)

    env.close()


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

    env = create_env(
        scenarios=config["scenarios"],
        agent_specs=agent_specs,
        sim_name=config["sim_name"],
        headless=config["headless"],
        seed=config["seed"],
    )

    actor_loop(
        env,
        (config["learner_address"], config["learner_port"]),
        num_episodes=config["episodes"],
    )


if __name__ == "__main__":
    parser = default_argument_parser("distributed-env")
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
