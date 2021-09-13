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
    ) -> None:
        self._fired = False
        self._should_trigger = should_trigger
        self._on_trigger = on_trigger

    def update(self, context: Dict[str, Any]):
        if not self._fired and self._should_trigger(context):
            self._on_trigger(context)
            self._fired = True


def main(
    script: str,
    scenarios: Sequence[str],
    headless: bool,
    seed: int,
    vehicles_to_replace_randomly: int,
    positional_radius: int,
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

            # Select a random sample from candidates
            k = ctx.get("k", 0)
            if k <= 0:
                logger.warning(
                    "default (0) or negative value specified for replacement. Replacing all valid vehicle candidates."
                )
                sample = ctx["vehicle_candidates"]
            else:
                logger.info(
                    f"Choosing {k} vehicles randomly from {len(ctx['vehicle_candidates'])} valid vehicle candidates."
                )
                sample = random.sample(ctx["vehicle_candidates"], k)

            assert len(sample) != 0

            # Map selected vehicles to agent ids & specs
            vehicles_to_trap = {}
            for veh_id in sample:
                agent_id = f"agent-{veh_id}"
                vehicles_to_trap[veh_id] = (agent_id, agent_spec)
                ctx["agents"][agent_id] = agent_spec.build_agent()

            # Create missions for selected vehicles
            veh_missions = scenario.create_dynamic_traffic_history_missions(
                sample, ctx["elapsed_sim_time"], positional_radius
            )

            # Create traps for selected vehicles to be triggered immediately
            smarts.trap_history_vehicles(vehicles_to_trap, veh_missions)

        # Create a table of vehicle trajectory lengths, filtering out non-moving vehicles
        vehicle_candidates = []
        for v_id in (str(id) for id in scenario.traffic_history.all_vehicle_ids()):
            traj = list(scenario.traffic_history.vehicle_trajectory(v_id))
            # Find moving vehicles with more than 100 timesteps
            if [row for row in traj if row.speed != 0] and len(traj) >= 100:
                vehicle_candidates.append(v_id)

        assert len(vehicle_candidates) > 0

        k = vehicles_to_replace_randomly
        if k > len(vehicle_candidates):
            logger.warning(
                f"vehicles_to_replace_randomly={k} is greater than the number of vehicle candidates ({len(vehicle_candidates)})."
            )
            k = len(vehicle_candidates)

        # Initialize trigger and define initial context
        context = {
            "agents": {},
            "elapsed_sim_time": 0.0,
            "vehicle_candidates": vehicle_candidates,
            "k": k,
        }
        trigger = Trigger(should_trigger, on_trigger)

        dones = {}
        observations = smarts.reset(scenario)
        while not dones or not all(dones.values()):
            # Update context
            context["elapsed_sim_time"] = smarts.elapsed_sim_time

            # Step trigger to further update context
            trigger.update(context)

            # Get agents from current context
            agents = context["agents"]

            # Step simulation
            actions = {
                agent_id: agents[agent_id].act(agent_obs)
                for agent_id, agent_obs in observations.items()
            }
            logger.debug(
                f"stepping @ sim_time={smarts.elapsed_sim_time} for agents={list(observations.keys())}..."
            )
            observations, rewards, dones, infos = smarts.step(actions)

            for agent_id in agents.keys():
                if dones.get(agent_id, False):
                    if not observations[agent_id].events.reached_goal:
                        logger.warning(
                            f"agent_id={agent_id} exited @ sim_time={smarts.elapsed_sim_time}"
                        )
                        logger.warning(f"   ... with {observations[agent_id].events}")
                    else:
                        logger.info(
                            f"agent_id={agent_id} reached goal @ sim_time={smarts.elapsed_sim_time}"
                        )
                        logger.debug(f"   ... with {observations[agent_id].events}")
                    del observations[agent_id]

    smarts.destroy()


if __name__ == "__main__":
    parser = default_argument_parser("dynamic_history_vehicles_replacement.py")
    parser.add_argument(
        "--random_replacements-per-episode",
        "-k",
        help="The number vehicles to randomly replace with agents per episode.",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--positional_radius",
        "-r",
        help="The maximum radial distance (in metres) from the end position for which the PositionalGoal mission will end.",
        type=int,
        default=3,
    )
    args = parser.parse_args()
    main(
        script=parser.prog,
        scenarios=args.scenarios,
        headless=args.headless,
        seed=args.seed,
        vehicles_to_replace_randomly=args.random_replacements_per_episode,
        positional_radius=args.postional_radius,
        episodes=args.episodes,
    )
