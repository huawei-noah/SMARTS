from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Union

import numpy as np
from evaluation.costs import COST_FUNCS, Costs

from smarts.core.sensors import Observation

_GRACE_RADIUS = 15
_MAX_STEPS = 800


class Metric:
    def __init__(self, agent_names):
        self._agent_names = agent_names
        self._num_agents = len(agent_names)

        self._cost_funcs: Dict[
            str, Dict[str, Callable[[Observation], Dict[str, Union[int, float]]]]
        ]
        self._cost_per_episode: Dict[str, Dict[str, float]]
        self._crashes_per_episode: int
        """ Total number of crashed agents per episode. An agent is considered
        crashed if it becomes done (i) due to traffic violation, and (ii) is 
        outside the `_GRACE_RADIUS` from the goal position.
        """
        self._steps_adjusted_per_episode: int
        """ Total number of `act` steps taken per episode. A crashed episode is 
        assumed to have taken `_MAX_STEPS`.
        """
        self._incomplete_per_episode: int
        """ Total number of incomplete agents per episode. An agent is 
        considered incomplete if it becomes done (i) within the 
        `_GRACE_RADIUS`, (ii) but does not reach the goal position.    
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

        self._crashes_per_episode = 0
        self._steps_adjusted_per_episode = 0
        self._incomplete_per_episode = 0

    def store(self, infos: Dict[str, Any], dones: Dict[str, bool]):
        # Compute all cost functions
        for agent_name, agent_info in infos.items():
            agent_obs = agent_info["env_obs"]
            for cost_func in self._cost_funcs[agent_name].values():
                res = cost_func(agent_obs)
                for cost_name, cost_val in res.items():
                    self._cost_per_episode[agent_name][cost_name] += cost_val

        # Count only steps where an ego agent was present.
        if len(infos) > 0:
            self._counts.steps += 1
            self._steps_adjusted_per_episode += 1

        # Count number of agents which crashed and which did not complete the map.
        for agent_name, agent_info in infos.items():
            agent_done = dones[agent_name]
            agent_obs = agent_info["env_obs"]
            rel_pos = (
                agent_obs.ego_vehicle_state.position
                - agent_obs.ego_vehicle_state.mission.goal.position
            )
            dist = np.linalg.norm(rel_pos[:2])
            if (
                agent_done
                and dist <= _GRACE_RADIUS
                and not agent_obs.events.reached_goal
            ):
                self._incomplete_per_episode += 1
            elif (
                agent_done
                and dist > _GRACE_RADIUS
                and not agent_obs.events.reached_goal
            ):
                assert (
                    len(agent_obs.events.collisions) > 0
                    or agent_obs.events.off_road
                    or agent_obs.events.off_route
                    or agent_obs.events.on_shoulder
                    or agent_obs.events.wrong_way
                    or agent_obs.events.not_moving
                    or agent_obs.events.reached_max_episode_steps
                ), f"Unknown agent done reason. Events: {agent_obs.events}."
                self._crashes_per_episode += 1

        if dones["__all__"] == True:
            self._counts.episodes += 1

            if self._crashes_per_episode > 0:
                self._counts.crashes += self._crashes_per_episode / self._num_agents
                self._steps_adjusted_per_episode = _MAX_STEPS

            self._counts.steps_adjusted += self._steps_adjusted_per_episode
            self._counts.incomplete += self._incomplete_per_episode / self._num_agents

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


@dataclass
class Counts:
    crashes: float = 0
    """ Total number of crashed episodes. An episode is considered crashed if
    an agent becomes done (i) due to traffic violation, and (ii) is outside the
    `_GRACE_RADIUS` from the goal position. Fractional values occur when only 
    some agents crashes the episode in a multi-agent case.
    """
    episodes: int = 0
    """ Total number of episodes.
    """
    episodes_adjusted: int = 0
    """ Total number of equivalent episodes. For an n-agent scenario, 
    n-episodes are added to the total.
    """
    incomplete: float = 0
    """ Total number of incomplete episodes. An episode is considered 
    incomplete if an agent becomes done (i) within the `_GRACE_RADIUS`, 
    (ii) but does not reach the goal position. Fractional values occur when 
    only some agents do not complete the episode in a multi-agent case.
    """
    steps: int = 0
    """ Total number of `act` steps taken.
    """
    steps_adjusted: int = 0
    """ Total number of `act` steps taken. Any crashed episode is assumed to
    have taken `_MAX_STEPS`.
    """
