import logging
from typing import Any, Callable, Dict, Sequence

from envision.client import Client as Envision
from smarts.core import seed as random_seed
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.core.plan import Mission, Start, TraverseGoal, default_entry_tactic
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.traffic_history_provider import TrafficHistoryProvider
from smarts.core.vehicle import Vehicle

from examples.argument_parser import default_argument_parser

logging.basicConfig(level=logging.INFO)


class BasicAgent(Agent):
    def act(self, obs):
        return "keep_lane"


class Trigger:
    def __init__(
        self, predicate: Callable[[], bool], callback: Callable[[], Any]
    ) -> None:
        self._fired = False
        self._predicate = predicate
        self._callback = callback

    def update(self):
        if not self._fired and self._predicate():
            self._callback()
            self._fired = True


def hijack_vehicles(
    smarts, vehicles_to_hijack: Dict[str, AgentSpec], agents, traffic_history_provider
):
    for veh_id, agent_spec in vehicles_to_hijack.items():
        # Save agent/vehicle info
        agent_id = f"agent-history-vehicle-{veh_id}"
        agent = agent_spec.build_agent()
        agents[agent_id] = agent

        # Create trap to be triggered immediately
        vehicle: Vehicle = smarts.vehicle_index.vehicle_by_id(
            f"history-vehicle-{veh_id}"
        )
        mission = Mission(
            start=Start(vehicle.position, vehicle.heading),
            entry_tactic=default_entry_tactic(vehicle.speed),
            goal=TraverseGoal(smarts.road_map),
        )
        # smarts._trap_manager.add_trap_for_agent(agent_id, mission, smarts.road_map)
        smarts.add_agent_with_mission(agent_id, agent_spec.interface, mission)

    # Register chosen agents and remove from traffic history provider
    traffic_history_provider.set_replaced_ids(vehicles_to_hijack.keys())


def main(
    script: str,
    scenarios: Sequence[str],
    headless: bool,
    seed: int,
    episodes: int,
):
    assert episodes > 0
    logger = logging.getLogger(script)
    logger.setLevel(logging.INFO)

    logger.debug("initializing SMARTS")
    smarts = SMARTS(
        agent_interfaces={},
        traffic_sim=None,
        envision=None if headless else Envision(),
    )
    random_seed(seed)
    traffic_history_provider = smarts.get_provider_by_type(TrafficHistoryProvider)
    assert traffic_history_provider

    scenarios_iterator = Scenario.scenario_variations(scenarios, [])
    scenario = next(scenarios_iterator)
    assert scenario.traffic_history.dataset_source == "Waymo"

    # select vehicles and assign agents
    agent_spec = AgentSpec(
        interface=AgentInterface(waypoints=True, action=ActionSpaceType.Lane),
        agent_builder=BasicAgent,
    )
    vehicles_to_hijack = {
        "1067": agent_spec,
        "1069": agent_spec,
        "1072": agent_spec,
        "1131": agent_spec,
    }

    def should_trigger() -> bool:
        return smarts.elapsed_sim_time > 10

    def on_trigger():
        # hijack vehicles
        hijack_vehicles(smarts, vehicles_to_hijack, agents, traffic_history_provider)

    for episode in range(episodes):
        logger.info(f"starting episode {episode}...")
        observations = smarts.reset(scenario)
        agents = {}
        dones = {}

        trigger = Trigger(should_trigger, on_trigger)

        while not dones or not all(done for done in dones.values()):
            trigger.update()

            actions = {
                agent_id: agents[agent_id].act(agent_obs)
                for agent_id, agent_obs in observations.items()
            }
            logger.debug(
                "stepping @ sim_time={} for agents={}...".format(
                    smarts.elapsed_sim_time, list(observations.keys())
                )
            )
            observations, rewards, dones, infos = smarts.step(actions)

            for agent_id in agents.keys():
                if dones.get(agent_id, False):
                    if not observations[agent_id].events.reached_goal:
                        logger.warning(
                            "agent_id={} exited @ sim_time={}".format(
                                agent_id, smarts.elapsed_sim_time
                            )
                        )
                        logger.warning(
                            "   ... with {}".format(observations[agent_id].events)
                        )
                    else:
                        logger.info(
                            "agent_id={} reached goal @ sim_time={}".format(
                                agent_id, smarts.elapsed_sim_time
                            )
                        )
                        logger.debug(
                            "   ... with {}".format(observations[agent_id].events)
                        )
                    del observations[agent_id]

    smarts.destroy()


if __name__ == "__main__":
    parser = default_argument_parser("waymo-replay")
    args = parser.parse_args()

    main(
        script=parser.prog,
        scenarios=args.scenarios,
        headless=args.headless,
        seed=args.seed,
        episodes=args.episodes,
    )
