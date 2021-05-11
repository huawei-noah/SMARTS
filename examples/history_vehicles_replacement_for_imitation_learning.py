import logging
from dataclasses import replace
import random
import sys

from envision.client import Client as Envision
from examples import default_argument_parser
from smarts.core import seed as random_seed
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.scenario import Mission, Scenario
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.traffic_history_provider import TrafficHistoryProvider

logging.basicConfig(level=logging.INFO)


class KeepLaneAgent(Agent):
    def __init__(self, target_speed=15.0):
        self._target_speed = target_speed

    def act(self, obs):
        return (self._target_speed, 0)


def _choose_random_sample(vehicle_missions, k, traffic_history, logger):
    # ensure overlapping time intervals across sample
    # this is inefficient, but it's not that important
    choice = random.choice(list(vehicle_missions.keys()))
    sample = {choice}
    sample_start_time = vehicle_missions[choice].start_time
    sample_end_time = traffic_history.vehicle_last_seen_time(choice)
    while len(sample) < k:
        choices = list(
            traffic_history.vehicle_ids_active_between(
                sample_start_time, sample_end_time
            )
        )
        if len(choices) <= len(sample):
            logger.warning(f"Unable to choose {k} overlapping missions.")
            break
        choice = str(random.choice(choices)[0])
        sample_start_time = min(vehicle_missions[choice].start_time, sample_start_time)
        sample_end_time = max(
            traffic_history.vehicle_last_seen_time(choice), sample_end_time
        )
        sample.add(choice)
    return sample


def main(script, scenarios, headless, seed, vehicles_to_replace, episodes):
    assert vehicles_to_replace > 0
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
    for _ in scenarios:
        scenario = next(scenarios_iterator)
        logger.debug("working on scenario {}".format(scenario.name))
        veh_missions = scenario.discover_missions_of_traffic_histories()
        if not veh_missions:
            logger.warning(
                "no vehicle missions found for scenario {}.".format(scenario.name)
            )
            continue

        k = vehicles_to_replace
        if vehicles_to_replace > len(veh_missions):
            logger.warning(
                "vehicles_to_replace={} is greater than the number of vehicle missions ({}).".format(
                    vehicles_to_replace, len(veh_missions)
                )
            )
            k = len(veh_missions)

        # XXX replace with AgentSpec appropriate for IL model
        agent_spec = AgentSpec(
            interface=AgentInterface.from_type(
                AgentType.LanerWithSpeed, max_episode_steps=None
            ),
            agent_builder=KeepLaneAgent,
            agent_params=scenario.traffic_history.target_speed,
        )

        for episode in range(episodes):
            logger.info(f"starting episode {episode}...")

            # Pick k vehicle missions to hijack with agent
            # and figure out which one starts the earliest
            agentid_to_vehid = {}
            agent_interfaces = {}
            history_start_time = None
            sample = _choose_random_sample(
                veh_missions, k, scenario.traffic_history, logger
            )
            logger.info(f"chose vehicles: {sample}")
            for veh_id in sample:
                agent_id = f"ego-agent-IL-{veh_id}"
                agentid_to_vehid[agent_id] = veh_id
                agent_interfaces[agent_id] = agent_spec.interface
                mission = veh_missions[veh_id]
                if not history_start_time or mission.start_time < history_start_time:
                    history_start_time = mission.start_time

            # Build the Agents for the to-be-hijacked vehicles
            # and gather their missions
            agents = {}
            dones = {}
            ego_missions = {}
            for agent_id in agent_interfaces.keys():
                agents[agent_id] = agent_spec.build_agent()
                dones[agent_id] = False
                mission = veh_missions[agentid_to_vehid[agent_id]]
                ego_missions[agent_id] = replace(
                    mission, start_time=mission.start_time - history_start_time
                )

            # Tell the traffic history provider to start traffic
            # at the point when the earliest agent enters...
            traffic_history_provider.start_time = history_start_time
            # and all the other agents to offset their missions by this much too
            scenario.set_ego_missions(ego_missions)
            logger.info(f"offsetting sim_time by: {history_start_time}")

            # Take control of vehicles with corresponding agent_ids
            smarts.switch_ego_agents(agent_interfaces)

            # Finally start the simulation loop...
            logger.info(f"starting simulation loop...")
            observations = smarts.reset(scenario)
            while not all(done for done in dones.values()):
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
                        logger.info(
                            "agent_id={} exited @ sim_time={}".format(
                                agent_id, smarts.elapsed_sim_time
                            )
                        )
                        logger.debug(
                            "   ... with {}".format(observations[agent_id].events)
                        )
                        del observations[agent_id]

    smarts.destroy()


if __name__ == "__main__":
    parser = default_argument_parser("history-vehicles-replacement-example")
    parser.add_argument(
        "-k",
        help="The number vehicles to randomly replace with agents per episode.",
        type=int,
        default=3,
    )
    args = parser.parse_args()

    main(
        script=parser.prog,
        scenarios=args.scenarios,
        headless=args.headless,
        seed=args.seed,
        vehicles_to_replace=args.k,
        episodes=args.episodes,
    )
