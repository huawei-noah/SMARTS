#!/usr/bin/env python3
import argparse
import dataclasses
import inspect
import logging
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import gym
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import (
    DoneCriteria, NeighborhoodVehicles, RGB, Waypoints
)
from smarts.core.controllers import ActionSpaceType
import ultra.adapters as adapters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)

_MAX_EPISODE_STEPS = 1000
_SCORES_FILENAME = "scores.txt"
_MAX_EPISODE_STEPS = 1200
_SPEEDING_THRESHOLD = 1.1


@dataclasses.dataclass
class Scores:
    score: float = 0.0
    reached_goal: float = 0.0
    collision: float = 0.0
    off_road: float = 0.0
    off_route: float = 0.0
    wrong_way: float = 0.0
    timed_out: float = 0.0
    speed_violation: float = 0.0
    episode_length: float = 0.0

    def calculate_score(
        self, info_trajectory: List[Dict[str, Any]], expected_steps: int
    ):
        """Calculates the agent's score from information about its episode.

            The score calculation is:
                score = (
                    reached_goal
                    * (speed_violation < 1.0)
                    * expected_steps / episode_length
                )

            Where:
            - reached_goal is 1.0 if the agent has reached the goal, 0.0 otherwise
            - speed_violation is 1.0 if the agent's speed has ever gone 10% over the
                speed limit in the episode, 0.0 otherwise
            - expected_steps is the expected number of steps this scenario should take
                to complete
            - episode_length is the number of steps the agent took to complete the
                scenario

        Args:
            info_trajectory (List[Dict[str, Any]]): A list of dictionaries where each
                dictionary contains information about a step of the agent's episode.
                The dictionaries should contain information according to ULTRA's default
                info adapter (https://github.com/huawei-noah/SMARTS/blob/master/ultra/docs/adapters.md#info-adapters).
            expected_steps (int): The expected number of steps this episode should take
                to complete. This number is used in the calculation of the score.

        Returns: None
        """
        for info in info_trajectory:
            # Length of episode.
            self.episode_length += 1

            # Termination events.
            events = info["logs"]["events"]
            self.reached_goal = 1.0 if events.reached_goal else 0.0
            self.collision = 1.0 if len(events.collisions) > 0 else 0.0
            self.off_road = 1.0 if events.off_road else 0.0
            self.off_route = 1.0 if events.off_route else 0.0
            self.wrong_way = 1.0 if events.wrong_way else 0.0
            self.timed_out = 1.0 if events.reached_max_episode_steps else 0.0

            # Violations.
            speed = info["logs"]["speed"]
            speed_limit = info["logs"]["closest_wp"].speed_limit
            if speed > _SPEEDING_THRESHOLD * speed_limit:
                self.speed_violation = 1.0

        # Calculate the score.
        self.score = (
            (self.reached_goal == 1.0)
            * (self.speed_violation == 0.0)
            * expected_steps / self.episode_length
        )


    def __iadd__(self, other_scores):
        """Support the "+=" operation with another Scores object."""
        assert isinstance(other_scores, Scores)
        self.score += other_scores.score
        self.reached_goal += other_scores.reached_goal
        self.collision += other_scores.collision
        self.off_road += other_scores.off_road
        self.off_route += other_scores.off_route
        self.wrong_way += other_scores.wrong_way
        self.timed_out += other_scores.timed_out
        self.speed_violation += other_scores.speed_violation
        self.episode_length += other_scores.episode_length
        return self

    def __itruediv__(self, n):
        """Support the "/=" operation with an integer or float."""
        assert isinstance(n, int) or isinstance(n, float)
        self.score /= n
        self.reached_goal /= n
        self.collision /= n
        self.off_road /= n
        self.off_route /= n
        self.wrong_way /= n
        self.timed_out /= n
        self.speed_violation /= n
        self.episode_length /= n
        return self

    def to_scores_string(self) -> str:
        """Convert the data in scores to a CodaLab-scores-compatible string."""
        # NOTE: The score string names must be the same as in the competition.yaml.
        return (
            f"score: {self.score}\n"
            f"reached_goal: {self.reached_goal}\n"
            f"collision: {self.collision}\n"
            f"off_road: {self.off_road}\n"
            f"off_route: {self.off_route}\n"
            f"wrong_way: {self.wrong_way}\n"
            f"timed_out: {self.timed_out}\n"
            f"speed_violation: {self.speed_violation}\n"
            f"episode_length: {self.episode_length}\n"
        )


def resolve_codalab_dirs(
    root_path: str, input_dir: str = None, output_dir: str = None
) -> Tuple[str, str, str]:
    """Returns directories needed for the completion of the evaluation submission.

    Args:
        root_path (str): The path to this file.
        input_dir (str): The path containing the "res" and "ref" directories provided by
            CodaLab.
        ouptut_dir (str): The path to output the scores.txt file.

    Returns:
        Tuple[str, str, str]: The submission directory, evaluation scenarios directory,
            and the scores directory respectively. The submission directory contains the
            user's submitted files, the evaluation scenarios directory contains the
            contents of the unzipped evaluation scenarios, and the scores directory is
            the directory in which to write the scores.txt file that is used to update
            the leaderboard.
    """
    logger.info(f"root_path={root_path}")
    logger.info(f"input_dir={input_dir}")
    logger.info(f"output_dir={output_dir}")

    submission_dir = os.path.join(input_dir, "res")
    evaluation_scenarios_dir = os.path.join(input_dir, "ref")
    scores_dir = output_dir

    if not os.path.exists(scores_dir):
        os.makedirs(scores_dir)

    logger.info(f"submission_dir={submission_dir}")
    logger.info(f"evaluation_scenarios_dir={evaluation_scenarios_dir}")
    logger.info(f"scores_dir={scores_dir}")

    if not os.path.isdir(submission_dir):
        logger.warning(f"submission_dir={submission_dir} does not exist.")

    return submission_dir, evaluation_scenarios_dir, scores_dir


def resolve_local_dirs(
    submission_dir: str, evaluation_scenarios_dir: str, scores_dir: str
) -> Tuple[str, str, str]:
    if not os.path.exists(scores_dir):
        os.makedirs(scores_dir)

    logger.info(f"submission_dir={submission_dir}")
    logger.info(f"evaluation_scenarios_dir={evaluation_scenarios_dir}")
    logger.info(f"scores_dir={scores_dir}")

    if not os.path.isdir(submission_dir):
        logger.warning(f"submission_dir={submission_dir} does not exist.")

    return submission_dir, evaluation_scenarios_dir, scores_dir


def _load_agent_spec_submission(submission_dir: str) -> AgentSpec:
    sys.path.append(submission_dir)

    # This will fail with an import error if the submission directory does not exist.
    import agent as agent_submission

    agent_spec: AgentSpec = agent_submission.agent_spec

    # Ensure that the submission uses one of the allowed observation adapters.
    assert (
        inspect.getsource(agent_spec.observation_adapter) in (
            inspect.getsource(adapters.default_observation_image_adapter.adapt),
            inspect.getsource(adapters.default_observation_vector_adapter.adapt),
        )
    ), "Your agent is not using one of the default observation adapters."

    # Ensure that the submission uses the Continuous action space.
    assert (
        agent_spec.interface.action == ActionSpaceType.Continuous
    ), f"Your agent must use the `{ActionSpaceType.Continuous}` action space."

    # Ensure the submission uses the valid RGB sensor.
    assert (
        agent_spec.interface.rgb is False
        or agent_spec.interface.rgb == RGB(width=64, height=64, resolution=(50 / 64))
    ), (
        f"Your agent must use `{RGB(width=64, height=64, resolution=(50 / 64))}`, not "
        f"`{agent_spec.interface.rgb}`."
    )

    # Ensure the submission uses the valid NeighborhoodVehicles sensor.
    assert (
        agent_spec.interface.neighborhood_vehicles is False
        or agent_spec.interface.neighborhood_vehicles == NeighborhoodVehicles(radius=200.0)
    ), (
        f"Your agent must use `{NeighborhoodVehicles(radius=200.0)}`, not "
        f"`{agent_spec.interface.neighborhood_vehicles}`."
    )

    # Ensure the submission uses the valid Waypoints sensor.
    assert (
        agent_spec.interface.waypoints is False
        or agent_spec.interface.waypoints == Waypoints(lookahead=20)
    ), (
        f"Your agent must use `{Waypoints(lookahead=20)}`, not "
        f"`{agent_spec.interface.waypoints}`."
    )

    agent_spec.info_adapter = adapters.default_info_adapter.adapt
    agent_spec.reward_adapter = adapters.default_reward_adapter.adapt
    agent_spec.interface.max_episode_steps = _MAX_EPISODE_STEPS
    agent_spec.interface.done_criteria = DoneCriteria(
        collision=True,
        off_road=True,
        off_route=True,
        on_shoulder=False,
        wrong_way=True,
        not_moving=False,
    )  # Use the default DoneCriteria.

    # Equip the agent's interface with neighborhood vehicle and waypoint sensors (even
    # if they are using the image observations) for use by the info and reward adapter.
    if agent_spec.interface.neighborhood_vehicles is False:
        agent_spec.interface.neighborhood_vehicles = NeighborhoodVehicles(radius=200.0)
    if agent_spec.interface.waypoints is False:
        agent_spec.interface.waypoints = Waypoints(lookahead=20)

    return agent_spec


def write_scores(scores: Scores, output_dir: str):
    with open(os.path.join(output_dir, _SCORES_FILENAME), "w") as output_file:
        output_file.write(scores.to_scores_string())


def _expected_steps_for_scenario(scenario_name: str) -> int:
    """Return the expected number of steps this scenario should take to complete."""
    if "no-traffic" in scenario_name:
        expected_steps = 200
    elif "low-density" in scenario_name:
        expected_steps = 250
    elif "mid-density" in scenario_name:
        expected_steps = 350
    elif "high-density" in scenario_name:
        expected_steps = 450
    else:
        logger.debug(
            f"`{scenario_name}` does not have an expected steps. Using default."
        )
        expected_steps = 200

    logger.debug(f"Expected steps for `{scenario_name}`: {expected_steps}")
    return expected_steps


def _rollout(agent: Agent, agent_id: str, env: gym.Env) -> Scores:
    rollout_scores = Scores()

    start_time = time.time()
    done = False
    info_trajectory = []
    observations = env.reset()
    scenario_name = env._smarts.scenario.name

    logger.debug(f"Rolling out submitted agent in scenario={scenario_name}.")

    expected_steps = _expected_steps_for_scenario(scenario_name)

    while not done:
        action = agent.act(observations[agent_id])
        observations, _, dones, infos = env.step({agent_id: action})
        info_trajectory.append(infos[agent_id])

        done = dones[agent_id]

        logger.debug(
            f"Step complete, elapsed duration={time.time() - start_time}, done={done}."
        )

    rollout_scores.calculate_score(info_trajectory, expected_steps)

    logger.info(f"Rollout complete with {rollout_scores}")

    return rollout_scores


def evaluate_submission(
    submission_dir: str, evaluation_scenarios_dir: str, seed: int = 1
) -> Scores:
    AGENT_ID = "AGENT-007"

    scores = Scores()

    agent_spec = _load_agent_spec_submission(submission_dir)

    scenarios = [
        os.path.join(evaluation_scenarios_dir, scenario_directory)
        for scenario_directory in os.listdir(evaluation_scenarios_dir)
        if os.path.isdir(os.path.join(evaluation_scenarios_dir, scenario_directory))
    ]
    scenarios.sort()

    env = gym.make(
        "ultra.env:ultra-v0",
        agent_specs={AGENT_ID: agent_spec},
        scenario_info=scenarios,
        headless=True,
        seed=seed,
        timestep_sec=0.1,
        ordered_scenarios=True,
    )

    # Evaluate the agent submission in each of the evaluation scenarios.
    for scenario_index in range(len(scenarios)):
        logger.info(f"Evaluating in scenario #{scenario_index + 1}")
        agent = agent_spec.build_agent()
        rollout_scores = _rollout(agent, AGENT_ID, env)
        scores += rollout_scores

    env.close()

    # Average over the scenarios.
    scores /= len(scenarios)

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="codalab-evaluation",
        description=(
            "Evaluation script run by CodaLab that outputs a contestant's leaderboard "
            "score by running their policy in simulation."
        ),
    )
    subparsers = parser.add_subparsers()

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--verbose",
        help="Print extra information regarding the evaluation.",
        action="store_true",
    )

    # Define arguments for evaluation through CodaLab.
    codalab_parser = subparsers.add_parser("codalab", parents=[common_parser])
    codalab_parser.add_argument(
        "--input-dir",
        help=(
            "The path to the directory containing the reference data and user "
            "submission data."
        ),
        required=True,
        type=str,
    )
    codalab_parser.add_argument(
        "--output-dir",
        help=(
            "The path to the directory where the submission's scores.txt file will be "
            "written to."
        ),
        required=True,
        type=str,
    )
    codalab_parser.set_defaults(which="codalab")

    # Define arguments for local evaluation.
    local_parser = subparsers.add_parser("local", parents=[common_parser])
    local_parser.add_argument(
        "--submission-dir",
        help="The path to the your agent.py file (and other relevant files).",
        required=True,
        type=str,
    )
    local_parser.add_argument(
        "--evaluation-scenarios-dir",
        help="The path to the scenarios that will be used for evaluation.",
        required=True,
        type=str,
    )
    local_parser.add_argument(
        "--scores-dir",
        help="The path to the directory where the scores text file will be saved.",
        required=True,
        type=str,
    )
    local_parser.set_defaults(which="local")

    args = parser.parse_args()

    # Obtain directories used to accomplish the evaluation of the submission.
    if args.which == "codalab":
        submission_dir, evaluation_scenarios_dir, scores_dir = resolve_codalab_dirs(
            os.path.dirname(__file__), args.input_dir, args.output_dir
        )
    elif args.which == "local":
        submission_dir, evaluation_scenarios_dir, scores_dir = resolve_local_dirs(
            args.submission_dir, args.evaluation_scenarios_dir, args.scores_dir
        )
    else:
        raise Exception(f"Invalid positional argument: `{args.which}`")

    # Set the verbosity.
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    # Evaluate the given submission.
    scores = evaluate_submission(submission_dir, evaluation_scenarios_dir)

    # Write the scores to the scores.txt file.
    write_scores(scores, scores_dir)

    logger.info("Evaluation complete.")
