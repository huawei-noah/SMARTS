import logging
import tempfile

import ray
import torch
import gym
import numpy as np

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.utils.episodes import episodes
from smarts.core.agent import AgentSpec, Agent

from examples import default_argument_parser


logging.basicConfig(level=logging.INFO)

AGENT_ID = "Agent-007"


class PyTorchAgent(Agent):
    def __init__(self, input_dims, hidden_dims, output_dims, model_path=None):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dims, hidden_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims, output_dims),
        )

        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        else:
            # initialize weights randomly
            def init_weights(m):
                if type(m) == torch.nn.Linear:
                    torch.nn.init.xavier_uniform_(m.weight)
                    m.bias.data.fill_(0.01)

            self.model.apply(init_weights)

    def act(self, obs):
        batched_obs = np.array([obs])
        x = torch.from_numpy(batched_obs)
        y = self.model(x)
        batched_actions = y.detach().numpy()
        return batched_actions[0]

    def save(self, path):
        torch.save(self.model.state_dict(), path)


def observation_adapter(env_obs):
    ego = env_obs.ego_vehicle_state
    waypoint_paths = env_obs.waypoint_paths
    wps = [path[0] for path in waypoint_paths]

    # distance of vehicle from center of lane
    closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))
    signed_dist_from_center = closest_wp.signed_lateral_error(ego.position)
    lane_hwidth = closest_wp.lane_width * 0.5
    dist_from_center = signed_dist_from_center / lane_hwidth
    angle_error = closest_wp.relative_heading(ego.heading)

    return np.array(
        [dist_from_center, angle_error, ego.speed, ego.steering], dtype=np.float32,
    )


@ray.remote
def train(training_scenarios, evaluation_scenarios, headless, num_episodes, seed):
    agent_params = {"input_dims": 4, "hidden_dims": 7, "output_dims": 3}
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(AgentType.Standard, max_episode_steps=5000),
        agent_params=agent_params,
        agent_builder=PyTorchAgent,
        observation_adapter=observation_adapter,
    )

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=training_scenarios,
        agent_specs={AGENT_ID: agent_spec},
        headless=headless,
        timestep_sec=0.1,
        seed=seed,
    )

    steps = 0
    for episode in episodes(n=num_episodes):
        agent = agent_spec.build_agent()
        observations = env.reset()
        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        while not dones["__all__"]:
            agent_obs = observations[AGENT_ID]
            agent_action = agent.act(agent_obs)
            observations, rewards, dones, infos = env.step({AGENT_ID: agent_action})
            episode.record_step(observations, rewards, dones, infos)
            steps += 1

            if steps % 500 == 0:
                print("Evaluating agent")

                # We construct an evaluation agent based on the saved
                # state of the agent in training.
                model_path = tempfile.mktemp()
                agent.save(model_path)

                eval_agent_spec = agent_spec.replace(
                    agent_params=dict(agent_params, model_path=model_path)
                )

                # Remove the call to ray.wait if you want evaluation to run
                # in parallel with training
                ray.wait(
                    [
                        evaluate.remote(
                            eval_agent_spec, evaluation_scenarios, headless, seed
                        )
                    ]
                )

    env.close()


@ray.remote
def evaluate(agent_spec, evaluation_scenarios, headless, seed):
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=evaluation_scenarios,  # we evaluate against the loop scenario
        agent_specs={AGENT_ID: agent_spec},
        headless=headless,
        timestep_sec=0.1,
        seed=seed,
    )
    agent = agent_spec.build_agent()

    accumulated_reward = 0
    observations = env.reset()
    dones = {"__all__": False}
    while not dones["__all__"]:
        agent_obs = observations[AGENT_ID]
        agent_action = agent.act(agent_obs)
        observations, rewards, dones, _infos = env.step({AGENT_ID: agent_action})
        accumulated_reward = rewards[AGENT_ID]

    env.close()

    print(f"Finished Evaluating Agent: {accumulated_reward:.2f}")


def main(
    training_scenarios, evaluation_scenarios, headless, num_episodes, seed,
):
    ray.init()
    ray.wait(
        [
            train.remote(
                training_scenarios, evaluation_scenarios, headless, num_episodes, seed,
            )
        ]
    )


if __name__ == "__main__":
    parser = default_argument_parser("pytorch-example")
    parser.add_argument(
        "--evaluation-scenario",
        default="scenarios/loop",
        help="The scenario to use for evaluation.",
        type=str,
    )
    args = parser.parse_args()

    main(
        training_scenarios=args.scenarios,
        evaluation_scenarios=[args.evaluation_scenario],
        headless=args.headless,
        num_episodes=args.num_episodes,
        seed=args.seed,
    )
