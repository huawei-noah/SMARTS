from contextlib import contextmanager
from dataclasses import replace
from functools import lru_cache
import logging
import math
from pathlib import Path
import random
import shutil
import tempfile
from typing import Dict, Iterable, Sequence, Tuple
from unittest.mock import Mock

from envision.client import Client as Envision
from smarts.core import seed as random_seed
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType, DoneCriteria
from smarts.core.agent_manager import AgentManager
from smarts.core.bubble_manager import BubbleManager
from smarts.core.scenario import Scenario
from smarts.core.sensors import Observation
from smarts.core.smarts import SMARTS
from smarts.core.traffic_history_provider import TrafficHistoryProvider
from smarts.sstudio.types import Bubble, MapZone, PositionalZone, SocialAgentActor
from smarts.zoo.agent_spec import AgentSpec

from smarts.zoo.registry import register

try:
    from argument_parser import default_argument_parser
except ImportError:
    from .argument_parser import default_argument_parser

logging.basicConfig(level=logging.INFO)

NUM_EPISODES = 1


class DummyAgent(Agent):
    """This is just a place holder that is used for the default agent used by the bubble."""

    def act(self, obs: Observation) -> Tuple[float, float]:
        acceleration = 0.0
        angular_velocity = 0.0
        return (acceleration, angular_velocity)


class UsedAgent(Agent):
    """This is just a place holder for the actual agent."""

    def __init__(self, logger) -> None:
        self._logger: logging.Logger = logger

    @lru_cache(maxsize=1)
    def called(self):
        self._logger.info("Agent act called first time.")

    def act(self, obs: Observation) -> Tuple[float, float]:
        self.called()
        acceleration = 0.0
        angular_velocity = 0.0
        return (acceleration, angular_velocity)


# Register the dummy agent for use by the bubbles 
#  referenced as `"<module>:<locator>"` (i.e. `"examples:dummy_agent-v0"`)
register(
    "dummy_agent-v0",
    entry_point=lambda **kwargs: AgentSpec(
        interface=AgentInterface.from_type(AgentType.Direct, done_criteria= DoneCriteria(not_moving=False, off_road=False, off_route=False, on_shoulder=False, wrong_way=False)),
        agent_builder=DummyAgent,
    ),
)

# Locations from inspecting the `map.net.xml` using netedit
bubble_locations = {
    "i80": [
        Bubble(
            zone=PositionalZone(pos=(0, 0), size=(20, 100)),
            actor=SocialAgentActor(
                name="dummy0", agent_locator="examples:dummy_agent-v0"
            ),
            follow_vehicle_id="history-vehicle-314",
            exclusion_prefixes=("history-vehicle-314",),
            follow_offset=(0, 0),
            margin=10
        ),
    ],
    "peachtree": [
        Bubble(
            zone=MapZone(start=("E9", 0, 5), length=40, n_lanes=2),
            actor=SocialAgentActor(
                name="dummy0", agent_locator="examples:dummy_agent-v0"
            ),
        )
    ],
    "us101": [
        Bubble(
            zone=MapZone(start=("gneE01.132", 0, 1), length=120, n_lanes=5),
            actor=SocialAgentActor(
                name="dummy0", agent_locator="examples:dummy_agent-v0"
            ),
        )
    ]
}


def main(
    script: str,
    ngsim_example: str,
    headless: bool,
    seed: int,
    episodes: int,
    start_time: float,
    run_time: float,
):
    logger = logging.getLogger(script)
    logger.setLevel(logging.INFO)

    logger.debug("initializing SMARTS")

    timestep = 0.1
    run_steps = int(run_time / timestep)
    smarts = SMARTS(
        agent_interfaces={},
        traffic_sim=None,
        envision=None if headless else Envision(),
        fixed_timestep_sec=timestep,
    )
    random_seed(seed)

    scenario = str(Path(__file__).parent.parent / "scenarios/NGSIM" / ngsim_example)
    scenario_list = Scenario.get_scenario_list([scenario])
    scenarios_iterator = Scenario.variations_for_all_scenario_roots(scenario_list, [])

    class ObservationState:
        def __init__(self) -> None:
            self.last_observations: Dict[str, Observation] = None

        def observation_callback(self, obs: Observation):
            self.last_observations = obs
    
    obs_state = ObservationState()

    for scenario in scenarios_iterator:
        scenario.bubbles.clear()
        scenario.bubbles.extend(bubble_locations[ngsim_example])

        # XXX replace with AgentSpec appropriate for IL model
        agent_manager: AgentManager = smarts.agent_manager
        agent_manager.add_social_observation_callback(
            obs_state.observation_callback, "bubble_watcher"
        )

        for episode in range(episodes):
            logger.info(f"starting episode {episode}...")
            smarts.reset(scenario, start_time=start_time)
            traffic_history_provider: TrafficHistoryProvider = smarts.get_provider_by_type(TrafficHistoryProvider)
            used_history_ids = set()

            agent = UsedAgent(logger=logger)

            for _ in range(run_steps):
                for agent_id in obs_state.last_observations:
                    if agent_id not in obs_state.last_observations:
                        continue
                    # Replace the original action that the social agent would do
                    agent_manager.reserve_social_agent_action(
                        agent_id, agent.act(obs_state.last_observations[agent_id])
                    )
                    used_history_ids |= {obs_state.last_observations[agent_id].ego_vehicle_state.id}
                # Step SMARTS
                _, _, _, _ = smarts.step({}) # observation_callback is called, obs_state updated
                # Currently ensure vehicles are removed permanently when they leave bubble
                traffic_history_provider.set_replaced_ids(used_history_ids)
                # Update the current bubble in case there are new active bubbles

                # Iterate through the observations
                # The agent ids of agents can be found here.
                for agent_id in obs_state.last_observations.keys():
                    if obs_state.last_observations[agent_id].events.reached_goal:
                        logger.info(
                            "agent_id={} reached goal @ sim_time={}".format(
                                agent_id, smarts.elapsed_sim_time
                            )
                        )
                        logger.debug(
                            "   ... with {}".format(
                                obs_state.last_observations[agent_id].events
                            )
                        )

    smarts.destroy()


if __name__ == "__main__":
    parser = default_argument_parser("history-vehicles-replacement-example")
    parser.add_argument(
        "--replacements-per-episode",
        "-k",
        help="The number vehicles to randomly replace with agents per episode.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--ngsim-example",
        "-g",
        help="The NGSIM example to run.",
        type=str,
        default="i80",
    )
    parser.add_argument(
        "--history-start-time",
        "-s",
        help="The start time of the simulated history.",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--run-time",
        "-r",
        help="The running time (s) of the simulation.",
        type=float,
        default=40,
    )
    args = parser.parse_args()

    main(
        script=parser.prog,
        ngsim_example=args.ngsim_example,
        headless=args.headless,
        seed=args.seed,
        episodes=args.episodes,
        start_time=args.history_start_time,
        run_time=args.run_time,
    )
