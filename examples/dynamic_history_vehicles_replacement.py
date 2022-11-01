import logging
import random
from typing import Any, Callable, Dict, Sequence

from argument_parser import default_argument_parser

from envision.client import Client as Envision
from smarts.core import seed as random_seed
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.core.plan import PlanningError
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.zoo.agent_spec import AgentSpec

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
    envision_record_data_replay_path: str,
    seed: int,
    vehicles_to_replace_randomly: int,
    min_timestep_count: int,
    positional_radius: int,
    episodes: int,
):
    assert episodes > 0
    logger = logging.getLogger(script)
    logger.setLevel(logging.INFO)
    logger.debug("initializing SMARTS")

    envision_client = None
    if not headless or envision_record_data_replay_path:
        envision_client = Envision(output_dir=envision_record_data_replay_path)

    smarts = SMARTS(
        agent_interfaces={},
        envision=envision_client,
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
            k = ctx.get("vehicles_to_replace_randomly", 0)
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

            for veh_id in sample:
                # Map selected vehicles to agent ids & specs
                agent_id = f"agent-{veh_id}"
                ctx["agents"][agent_id] = agent_spec.build_agent()

                # Create missions based on current state and traffic history
                positional, traverse = scenario.create_dynamic_traffic_history_mission(
                    veh_id, ctx["elapsed_sim_time"], ctx["positional_radius"]
                )

                # Take control of vehicles immediately
                try:
                    # Try to assign a PositionalGoal at the last recorded timestep
                    smarts.add_agent_and_switch_control(
                        veh_id, agent_id, agent_spec.interface, positional
                    )
                except PlanningError:
                    logger.warning(
                        f"Unable to create PositionalGoal for vehicle {veh_id}, falling back to TraverseGoal"
                    )
                    smarts.add_agent_and_switch_control(
                        veh_id, agent_id, agent_spec.interface, traverse
                    )

        # Create a table of vehicle trajectory lengths, filtering out non-moving vehicles
        vehicle_candidates = []
        assert scenario.traffic_history
        for v_id in (str(id) for id in scenario.traffic_history.all_vehicle_ids()):
            traj = list(scenario.traffic_history.vehicle_trajectory(v_id))
            # Find moving vehicles with more than the minimum number of timesteps
            if [row for row in traj if row.speed != 0] and len(
                traj
            ) >= min_timestep_count:
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
            "vehicles_to_replace_randomly": k,
            "positional_radius": positional_radius,
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
        "--min_timestep_count",
        "-t",
        help="The minimum number of timesteps a vehicle must have in its recorded trajectory to become a candidate for selection.",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--positional_radius",
        "-r",
        help="The maximum radial distance (in metres) from the end position for which the PositionalGoal mission will end.",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--envision_record_data_path",
        help="Envisions data replay output directory where the recording will be stored.",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    main(
        script=parser.prog,
        scenarios=args.scenarios,
        headless=args.headless,
        envision_record_data_replay_path=args.envision_record_data_path,
        seed=args.seed,
        vehicles_to_replace_randomly=args.random_replacements_per_episode,
        min_timestep_count=args.min_timestep_count,
        positional_radius=args.positional_radius,
        episodes=args.episodes,
    )
