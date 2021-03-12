import argparse
import logging

import gym
from pynput.keyboard import Key, Listener

from examples import default_argument_parser
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.utils.episodes import episodes

logging.basicConfig(level=logging.INFO)

AGENT_ID = "Agent-007"


class HumanKeyboardAgent(Agent):
    def __init__(self):
        # initialize the keyboard listener
        self.listener = Listener(on_press=self.on_press)
        self.listener.start()

        # Parameters for the human-keyboard agent
        # you need to change them to suit the scenario
        # These values work the best with zoo_intersection

        self.INC_THROT = 0.01
        self.INC_STEER = 5.0

        self.MAX_THROTTLE = 0.6
        self.MIN_THROTTLE = 0.45

        self.MAX_BRAKE = 1.0
        self.MIN_BRAKE = 0.0

        self.MAX_STEERING = 1.0
        self.MIN_STEERING = -1.0

        self.THROTTLE_DISCOUNTING = 0.99
        self.BRAKE_DISCOUNTING = 0.95
        self.STEERING_DISCOUNTING = 0.9

        # initial values
        self.steering_angle = 0.0
        self.throttle = 0.48
        self.brake = 0.0

    def on_press(self, key):
        """To control, use the keys:
        Up: to increase the throttle
        Left Alt: to increase the brake
        Left: to decrease the steering angle
        Right: to increase the steering angle
        """
        if key == Key.up:
            self.throttle = min(self.throttle + self.INC_THROT, self.MAX_THROTTLE)
        elif key == Key.alt_l:
            self.brake = min(self.brake + 10.0 * self.INC_THROT, self.MAX_BRAKE)
        elif key == Key.right:
            self.steering_angle = min(
                self.steering_angle + self.INC_STEER, self.MAX_STEERING
            )
        elif key == Key.left:
            self.steering_angle = max(
                self.steering_angle - self.INC_STEER, self.MIN_STEERING
            )

    def act(self, obs):
        # discounting ..
        self.throttle = max(
            self.throttle * self.THROTTLE_DISCOUNTING, self.MIN_THROTTLE
        )
        self.steering_angle = self.steering_angle * self.STEERING_DISCOUNTING
        self.brake = self.brake * self.BRAKE_DISCOUNTING
        # send the action
        self.action = [self.throttle, self.brake, self.steering_angle]
        return self.action


def main(
    scenarios,
    sim_name,
    headless,
    num_episodes,
    seed,
):
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.StandardWithAbsoluteSteering, max_episode_steps=3000
        ),
        agent_builder=HumanKeyboardAgent,
    )

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={AGENT_ID: agent_spec},
        sim_name=sim_name,
        headless=headless,
        timestep_sec=0.1,
        seed=seed,
    )

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

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("human-in-the-loop-example")
    args = parser.parse_args()

    main(
        scenarios=args.scenarios,
        sim_name=args.sim_name,
        headless=args.headless,
        num_episodes=args.episodes,
        seed=args.seed,
    )
