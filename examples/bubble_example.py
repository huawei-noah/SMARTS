from functools import lru_cache
import logging
from pathlib import Path
from typing import Dict, Tuple

from envision.client import Client as Envision
from smarts.core import seed as random_seed
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType, DoneCriteria
from smarts.core.agent_manager import AgentManager
from smarts.core.local_traffic_provider import LocalTrafficProvider
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


class DummyAgent(Agent):
    """This is just a place holder that is used for the default agent used by the bubble."""

    def act(self, obs: Observation) -> Tuple[float, float]:
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

class BubbleOverrideAgent(Agent):
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


class MainAgent(Agent):
    def act(self, obs: Observation):
        nearest = obs.via_data.near_via_points[0]
        return (
            nearest.required_speed,
            nearest.lane_index,
        )



def create_moving_bubble(vehicle_id=None, or_agent_id=None):
    assert vehicle_id or or_agent_id

    follow = dict()
    exclusion_prefixes=()
    if vehicle_id:
        follow=dict(follow_vehicle_id=vehicle_id)
        exclusion_prefixes=(vehicle_id,)
    else:
        follow=dict(follow_actor_id=or_agent_id)
    
    bubble = Bubble(
        zone=PositionalZone(pos=(0, 0), size=(20, 100)),
        actor=SocialAgentActor(
            name="dummy0", agent_locator="examples:dummy_agent-v0"
        ),
        exclusion_prefixes=exclusion_prefixes,
        follow_offset=(0, 0),
        margin=10,
        **follow
    )
    return bubble


def main(
    script: str,
    example: str,
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
        traffic_sims=[LocalTrafficProvider(endless_traffic=False)],
        envision=None if headless else Envision(),
        fixed_timestep_sec=timestep,
    )
    random_seed(seed)

    scenario = str(Path(example).absolute())
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
        scenario.bubbles.extend([create_moving_bubble("history-vehicle-314")])

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

            agent = BubbleOverrideAgent(logger=logger)

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
        "--example",
        "-g",
        help="The NGSIM example to run.",
        type=str,
        default="scenarios/NGSIM/i80",
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
        example=args.example,
        headless=args.headless,
        seed=args.seed,
        episodes=args.episodes,
        start_time=args.history_start_time,
        run_time=args.run_time,
    )
