import logging
import random
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

from argument_parser import default_argument_parser

from envision.client import Client as Envision
from smarts.core import seed as random_seed
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType, DoneCriteria
from smarts.core.agent_manager import AgentManager
from smarts.core.local_traffic_provider import LocalTrafficProvider
from smarts.core.scenario import Scenario
from smarts.core.sensors import Observation
from smarts.core.smarts import SMARTS
from smarts.core.utils.logging import timeit
from smarts.sstudio.types import Bubble, PositionalZone, SocialAgentActor
from smarts.zoo.agent_spec import AgentSpec
from smarts.zoo.registry import register

logging.basicConfig(level=logging.DEBUG)


class SocialAgent(Agent):
    """This is just a place holder for the actual agent."""

    def __init__(self, logger) -> None:
        self._logger: logging.Logger = logger

    @lru_cache(maxsize=1)
    def called(self):
        # prints once per instance because of method cache.
        self._logger.info("social agent instance act called first time.")

    def act(self, obs: Observation) -> Tuple[float, float]:
        self.called()
        acceleration = 0.0
        angular_velocity = 0.0
        return (acceleration, angular_velocity)


class EgoAgent(Agent):
    def act(self, obs: Observation):
        speed_limit = int(obs.waypoint_paths[0][0].speed_limit)
        return (
            random.randint(speed_limit - 15, speed_limit + 4),
            random.randint(-1, 1),
        )


def register_dummy_locator(interface, name="dummy_agent-v0"):
    class DummyAgent(Agent):
        """This is just a place holder that is used for the default agent used by the bubble."""

        def act(self, obs: Observation) -> Tuple[float, float]:
            acceleration = 0.0
            angular_velocity = 0.0
            return (acceleration, angular_velocity)

    # Register the dummy agent for use by the bubbles
    #  referenced as `"<module>:<locator>"` (i.e. `"examples:dummy_agent-v0"`)
    register(
        name,
        entry_point=lambda **kwargs: AgentSpec(
            interface=interface,
            agent_builder=DummyAgent,
        ),
    )


def create_moving_bubble(
    follow_vehicle_id=None, follow_agent_id=None, social_agent_name="dummy_agent-v0"
):
    assert follow_vehicle_id or follow_agent_id, "Must follow a vehicle or agent"
    assert not (
        follow_vehicle_id and follow_agent_id
    ), "Must follow only one of vehicle or agent"

    follow = dict()
    exclusion_prefixes = ()
    if follow_vehicle_id:
        follow = dict(follow_vehicle_id=follow_vehicle_id)
        exclusion_prefixes = (follow_vehicle_id,)
    else:
        follow = dict(follow_actor_id=follow_agent_id)

    bubble = Bubble(
        zone=PositionalZone(pos=(0, 0), size=(20, 40)),
        actor=SocialAgentActor(
            name="dummy0", agent_locator=f"examples:{social_agent_name}"
        ),
        exclusion_prefixes=exclusion_prefixes,
        follow_offset=(0, 0),
        margin=0,
        **follow,
    )
    return bubble


def resolve_agent_missions(
    scenario: Scenario,
    start_time: float,
    run_time: float,
    count: int,
):
    # pytype: disable=attribute-error
    agent_missions = {
        f"agent-{mission.vehicle_spec.veh_id}": mission
        for mission in random.sample(
            scenario.history_missions_for_window(
                start_time, start_time + run_time, run_time / 2
            ),
            k=count,
        )
    }
    # pytype: enable=attribute-error

    return agent_missions


def main(
    script: str,
    example: str,
    headless: bool,
    seed: int,
    episodes: int,
    start_time: float,
    run_time: float,
    num_agents: int,
):
    logger = logging.getLogger(script)
    logger.setLevel(logging.INFO)

    logger.info("initializing SMARTS")

    timestep = 0.1
    run_steps = int(run_time / timestep)
    social_interface = AgentInterface.from_type(
        AgentType.Direct,
        done_criteria=DoneCriteria(
            not_moving=False,
            off_road=False,
            off_route=False,
            on_shoulder=False,
            wrong_way=False,
        ),
    )
    register_dummy_locator(social_interface)

    ego_interface = AgentInterface.from_type(
        AgentType.LanerWithSpeed,
        done_criteria=DoneCriteria(
            not_moving=False,
            off_road=False,
            off_route=False,
            on_shoulder=False,
            wrong_way=False,
        ),
        max_episode_steps=run_steps,
    )

    smarts = SMARTS(
        agent_interfaces={},
        traffic_sims=[LocalTrafficProvider()],
        envision=None if headless else Envision(),
        fixed_timestep_sec=timestep,
    )
    random_seed(seed)

    scenario_str = str(Path(example).absolute())
    scenario_list = Scenario.get_scenario_list([scenario_str])
    scenarios_iterator = Scenario.variations_for_all_scenario_roots(scenario_list, [])

    class ObservationState:
        def __init__(self) -> None:
            self.last_observations: Dict[str, Observation] = {}

        def observation_callback(self, obs: Observation):
            self.last_observations = obs

    obs_state = ObservationState()

    scenario: Scenario
    for scenario in scenarios_iterator:
        # XXX replace with AgentSpec appropriate for IL model
        agent_manager: AgentManager = smarts.agent_manager
        agent_manager.add_social_agent_observations_callback(
            obs_state.observation_callback, "bubble_watcher"
        )

        for episode in range(episodes):
            with timeit(f"setting up ego agents in scenario...", logger.info):
                agent_missions = resolve_agent_missions(
                    scenario, start_time, run_time, num_agents
                )
                agent_interfaces = {
                    a_id: ego_interface for a_id in agent_missions.keys()
                }

            with timeit(f"setting up moving bubbles...", logger.info):
                scenario.bubbles.clear()
                scenario.bubbles.extend(
                    [
                        create_moving_bubble(follow_agent_id=a_id)
                        for a_id in agent_missions
                    ]
                )

                scenario.set_ego_missions(agent_missions)
                smarts.switch_ego_agents(agent_interfaces)

            with timeit(f"initializing agent policy...", logger.info):
                ego_agent = EgoAgent()
                social_agent = SocialAgent(logger=logger)

            with timeit(f"resetting episode {episode}...", logger.info):
                ego_observations = smarts.reset(scenario, start_time=start_time)
                dones = {a_id: False for a_id in ego_observations}
                dones_count = 0

            with timeit(f"running episode {episode}...", logger.info):
                while dones_count < num_agents:
                    with timeit(
                        f"SMARTS simulation/scenario step with {len(obs_state.last_observations)} social agents",
                        logger.info,
                    ):
                        for agent_id in obs_state.last_observations:
                            social_agent_ob = obs_state.last_observations[agent_id]
                            # Replace the original action that the social agent would do
                            agent_manager.reserve_social_agent_action(
                                agent_id, social_agent.act(social_agent_ob)
                            )
                        # Step SMARTS
                        ego_actions = {
                            ego_agent_id: ego_agent.act(obs)
                            for ego_agent_id, obs in ego_observations.items()
                            if not dones[ego_agent_id]
                        }
                        ego_observations, _, dones, _ = smarts.step(
                            ego_actions
                        )  # observation_callback is called, obs_state updated
                        for a_id in dones:
                            if dones[a_id]:
                                dones_count += 1
                                logger.info(
                                    f"agent=`{a_id}` is done because `{ego_observations[a_id].events}`..."
                                )

                        # Update the current bubble in case there are new active bubbles

                        # Iterate through the observations
                        # The agent ids of agents can be found here.
                        for agent_id in obs_state.last_observations.keys():
                            if obs_state.last_observations[agent_id].events.collisions:
                                logger.info(
                                    "social_agent={} collided @ {}".format(
                                        agent_id,
                                        obs_state.last_observations[
                                            agent_id
                                        ].events.collisions,
                                    )
                                )
                            elif obs_state.last_observations[
                                agent_id
                            ].events.reached_goal:
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
    logger.info(f"ending simulation...")
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
        help="The start time of the simulation relative to the history dataset.",
        type=float,
        default=90,
    )
    parser.add_argument(
        "--run-time",
        "-r",
        help="The running time (s) of the simulation.",
        type=float,
        default=40,
    )
    parser.add_argument(
        "--num-agents",
        "-n",
        help="The number of agents to add to the scenario.",
        type=float,
        default=2,
    )
    args = parser.parse_args()

    main(
        script=parser.prog,
        example=args.example,
        headless=args.headless,
        seed=args.seed,
        episodes=3,
        start_time=args.history_start_time,
        run_time=args.run_time,
        num_agents=args.num_agents,
    )
