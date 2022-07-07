from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional, Union
from enum import Enum
import operator

import numpy as np
from evaluation.costs import COST_FUNCS, Costs
from evaluation.overtake import Overtake

from smarts.core.sensors import Observation


_MAX_STEPS = 800


class Metric:
    def __init__(self, env_name: str, agent_names):
        self._agent_names = agent_names
        self._num_agents = len(agent_names)
        self._env_name = env_name

        self._cost_funcs: Dict[
            str, Dict[str, Callable[[Observation], Dict[str, Union[int, float]]]]
        ]
        self._cost_per_episode: Dict[str, Dict[str, float]]
        self._steps_adjusted_per_episode: int
        """ Total number of `act` steps taken per episode. A crashed episode is 
        assumed to have taken `_MAX_STEPS`.
        """
        self._completion: Dict[str, _Completion]
        """ A dictionary storing the completion state of each agent.
        """
        self._overtake: Dict[str, Optional[Overtake]]
        """An overtake detector, used only in overtake scenarios.
        """
        self._reinit()

        self._counts = Counts()
        self._costs = Costs()

    def _reinit(self):
        self._cost_funcs = {
            agent_name: {
                cost_name: cost_func() for cost_name, cost_func in COST_FUNCS.items()
            }
            for agent_name in self._agent_names
        }
        self._cost_per_episode = {
            agent_name: {key: 0 for key in asdict(Costs()).keys()}
            for agent_name in self._agent_names
        }
        self._steps_adjusted_per_episode = 0
        self._completion = {
            agent_name: _Completion.Crashed for agent_name in self._agent_names
        }
        self._overtake = {
            agent_name: Overtake() if "overtake" in self._env_name else None
            for agent_name in self._agent_names
        }

    def store(self, infos: Dict[str, Any], dones: Dict[str, bool]):
        # Only count steps in which an ego agent was present.
        if len(infos) > 0:
            self._counts.steps += 1
            self._steps_adjusted_per_episode += 1

        for agent_name, agent_info in infos.items():
            agent_obs = agent_info["env_obs"]
            agent_done = dones[agent_name]

            # Compute all cost functions.
            for cost_func in self._cost_funcs[agent_name].values():
                res = cost_func(agent_obs)
                for cost_name, cost_val in res.items():
                    self._cost_per_episode[agent_name][cost_name] += cost_val

            # Compute optional metrics.
            if self._overtake[agent_name]:
                self._overtake[agent_name](agent_obs)

            # If done, categorize agent's completion state.
            if agent_done:
                self._completion[agent_name] = _classify(obs=agent_obs)

        if dones["__all__"] == True:
            # In overtake scenario, if goal was achieved without overtaking, decrement the
            # agent's completion from `Goal` to `Partial`.
            for agent_name, ovrtk in self._overtake.items():
                if (
                    ovrtk is not None
                    and not ovrtk.check()
                    and self._completion[agent_name] == _Completion.Goal
                ):
                    self._completion[agent_name] = _Completion.Partial

            # Update counts.
            self._counts.episodes += 1
            num_crashes = operator.countOf(
                self._completion.values(), _Completion.Crashed
            )
            num_partial = operator.countOf(
                self._completion.values(), _Completion.Partial
            )
            if num_crashes > 0:
                self._counts.crashes += num_crashes / self._num_agents
                self._steps_adjusted_per_episode = _MAX_STEPS
            self._counts.steps_adjusted += self._steps_adjusted_per_episode
            self._counts.partial += num_partial / self._num_agents

            # Transfer episode costs to total costs.
            for agent_name, costs in self._cost_per_episode.items():
                for cost_name, cost_val in costs.items():
                    new_val = getattr(self._costs, cost_name) + cost_val
                    setattr(self._costs, cost_name, new_val)
                self._counts.episodes_adjusted += 1

            # Reset cost functions and episodic costs.
            self._reinit()

    def results(self):
        return self._counts, self._costs


class _Completion(Enum):
    Goal = 0
    """ Agent achieved its goal."""
    Partial = 1
    """ Agent drives to within grace radius of goal location, and becomes
    done, but does not achieve its goal.
    """
    Crashed = 2
    """ Agent becomes done (i) due to traffic rule violation, and (ii) is 
    farther than the grace radius from the goal position, i.e., does not 
    achieve its goal.
    """


def _classify(obs) -> _Completion:
    grace_radius = 15

    rel_pos = (
        obs.ego_vehicle_state.position - obs.ego_vehicle_state.mission.goal.position
    )
    dist = np.linalg.norm(rel_pos[:2])

    if obs.events.reached_goal:
        return _Completion.Goal
    elif dist <= grace_radius and not obs.events.reached_goal:
        return _Completion.Partial
    elif dist > grace_radius and not obs.events.reached_goal:
        assert (
            len(obs.events.collisions) > 0
            or obs.events.off_road
            or obs.events.off_route
            or obs.events.on_shoulder
            or obs.events.wrong_way
            or obs.events.not_moving
            or obs.events.reached_max_episode_steps
        ), f"Unknown agent done reason. Events: {obs.events}."
        return _Completion.Crashed
    else:
        raise Exception(
            f"Unknown agent done reason. Events: {obs.events}. Distance from goal: {dist}."
        )


@dataclass
class Counts:
    crashes: float = 0
    """ Total number of crashed episodes. An episode is considered crashed if
    an agent becomes done (i) due to traffic violation, and (ii) is outside the
    grace radius from the goal position. Fractional values occur when only 
    some agents crashes the episode in a multi-agent case.
    """
    episodes: int = 0
    """ Total number of episodes.
    """
    episodes_adjusted: int = 0
    """ Total number of equivalent episodes. For an n-agent scenario, 
    n-episodes are added to the total.
    """
    partial: float = 0
    """ Total number of partially completed episodes. An episode is considered 
    partially complete if an agent becomes done (i) within the grace radius, 
    (ii) but does not reach the goal position. Fractional values occur when 
    only some agents partially complete the episode in a multi-agent case.
    """
    steps: int = 0
    """ Total number of `act` steps taken.
    """
    steps_adjusted: int = 0
    """ Total number of `act` steps taken. Any crashed episode is assumed to
    have taken `_MAX_STEPS`.
    """
