import argparse
import logging
from typing import Dict

from smarts.core.agent import Agent
from smarts.core.agent_manager import AgentManager
from smarts.core.bubble_manager import BubbleManager
from smarts.core.scenario import Scenario
from smarts.core.sensors import Observation
from smarts.core.smarts import SMARTS

from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation

logging.basicConfig(level=logging.INFO)

NUM_EPISODES = 1

class SpinningAgent(Agent):
    def act(self, obs: dict):
        return [0, 0.1]

def main(server_config):
    smarts = SMARTS(
        agent_interfaces={},
        traffic_sim=SumoTrafficSimulation(headless=False),
        envision=None,
    )

    agent = SpinningAgent()
    class obs_c:
        last_observations: Dict[str, Observation] = None
    def observation_callback(obs):
        obs_c.last_observations = obs

    agent_manager: AgentManager = smarts.agent_manager
    agent_manager.add_social_observation_callback(observation_callback, "bubble_watcher")

    for _ in range(NUM_EPISODES):
        scenarios = ["scenarios/loop_bubble_capture"]
        scenarios_iterator = Scenario.scenario_variations(scenarios, [])
        smarts.reset(next(scenarios_iterator))
        bubble_manager: BubbleManager = smarts._bubble_manager
        bubbles = bubble_manager.bubbles

        for _ in range(1000):
            for agent_ids in [bubble_manager.agent_ids_for_bubble(b, smarts) for b in bubbles]:
                if agent_ids == None:
                    continue
                for agent_id in agent_ids:
                    if agent_id not in obs_c.last_observations:
                        continue
                    agent_manager.reserve_social_agent_action(agent_id, agent.act(obs_c.last_observations[agent_id]))
            smarts.step({})
            bubbles = bubble_manager.bubbles
            

    smarts.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("local-service-example")
    # parser.add_argument(
    #     "server_config",
    #     help="A configuration file for the server",
    #     type=str,
    #     nargs=1,
    # )
    args = parser.parse_args()

    main(
        None, # args.server_config
    )
