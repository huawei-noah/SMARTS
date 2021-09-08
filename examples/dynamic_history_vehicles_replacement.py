import logging
import random
from typing import Any, Callable, Dict, Sequence

from envision.client import Client as Envision
from smarts.core import seed as random_seed
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS

from examples.argument_parser import default_argument_parser

logging.basicConfig(level=logging.INFO)


class BasicAgent(Agent):
    def act(self, obs):
        return "keep_lane"


class Trigger:
    """A basic class that will call a registered callback function when some criteria is met"""

    def __init__(
        self,
        should_trigger: Callable[[Dict[str, Any]], bool],
        on_trigger: Callable[[Dict[str, Any]], None],
        context: Dict[str, Any] = {},
    ) -> None:
        self._fired = False
        self._should_trigger = should_trigger
        self._on_trigger = on_trigger
        self._context = context

    def update(self, **kwargs):
        self._context.update(kwargs)
        if not self._fired and self._should_trigger(self._context):
            self._on_trigger(self._context)
            self._fired = True

    @property
    def context(self):
        return self._context


def main(
    script: str,
    scenarios: Sequence[str],
    headless: bool,
    seed: int,
    vehicles_to_replace_randomly: int,
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

    scenarios_iterator = Scenario.scenario_variations(scenarios, [])
    scenario = next(scenarios_iterator)

    for episode in range(episodes):
        logger.info(f"starting episode {episode}...")

        def should_trigger(ctx: Dict[str, Any]) -> bool:
            return ctx["elapsed_sim_time"] > 2

        def on_trigger(ctx: Dict[str, Any]):
            # Define agent specs to be assigned
            agent_spec = AgentSpec(
                interface=AgentInterface(waypoints=True, action=ActionSpaceType.Lane),
                agent_builder=BasicAgent,
            )
            if ctx["vehicles_to_replace_randomly"] <= 0:
                logger.warning(
                    "default (0) or negative value specified for replacement. Replacing all valid vehicle candidates."
                )
                sample = ctx["vehicle_candidates"]
            else:
                logger.info(
                    "Choosing {} vehicles randomly from {} valid vehicle candidates.".format(
                        ctx["vehicles_to_replace_randomly"],
                        len(ctx["vehicle_candidates"]),
                    )
                )
                sample = random.sample(
                    ctx["vehicle_candidates"], ctx["vehicles_to_replace_randomly"]
                )

            assert len(sample) != 0

            # Select vehicles and map to agent ids & specs
            vehicles_to_trap = {}
            for veh_id in sample:
                agent_id = f"agent-{veh_id}"
                vehicles_to_trap[veh_id] = (agent_id, agent_spec)
                ctx["agents"][agent_id] = agent_spec.build_agent()

            # Create missions for selected vehicles
            veh_missions = scenario.create_dynamic_traffic_history_missions(
                vehicles_to_trap.keys(), ctx["elapsed_sim_time"]
            )

            # Create traps for selected vehicles to be triggered immediately
            smarts.trap_history_vehicles(vehicles_to_trap, veh_missions)

        # Create a table of vehicle trajectory lengths, filtering out non-moving vehicles
        traj_lens = []
        for v_id in scenario.traffic_history.all_vehicle_ids():
            traj = list(scenario.traffic_history.vehicle_trajectory(str(v_id)))
            if [row for row in traj if row.speed != 0]:
                traj_lens.append((v_id, len(traj)))

        # Filter out trajectories with less than 100 timesteps
        vehicle_candidates = [str(t[0]) for t in traj_lens if t[1] > 100]

        assert len(vehicle_candidates) > 0

        k = vehicles_to_replace_randomly
        if k > len(vehicle_candidates):
            logger.warning(
                "vehicles_to_replace_randomly={} is greater than the number of vehicle candidates ({}).".format(
                    vehicles_to_replace_randomly, len(vehicle_candidates)
                )
            )
            k = len(vehicle_candidates)

        # Initialize trigger and define initial context
        context = {
            "agents": {},
            "elapsed_sim_time": 0.0,
            "vehicle_candidates": vehicle_candidates,
            "vehicles_to_replace_randomly": k,
        }
        trigger = Trigger(should_trigger, on_trigger, context=context)

        dones = {}
        observations = smarts.reset(scenario)
        while not dones or not all(done for done in dones.values()):
            # Step trigger with updated context
            trigger.update(elapsed_sim_time=smarts.elapsed_sim_time)

            # Get agents from current context
            agents = trigger.context["agents"]

            # Step simulation
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
    parser = default_argument_parser(
        "history_vehicles_replacement_for_imitation_learning.py"
    )
    parser.add_argument(
        "--random_replacements-per-episode",
        "-k",
        help="The number vehicles to randomly replace with agents per episode.",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    main(
        script=parser.prog,
        scenarios=args.scenarios,
        headless=args.headless,
        seed=args.seed,
        vehicles_to_replace_randomly=args.random_replacements_per_episode,
        episodes=args.episodes,
    )
