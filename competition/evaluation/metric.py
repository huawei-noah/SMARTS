from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Union
from enum import Enum
import operator

from costs import COST_FUNCS, Costs
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
        """ Temporary dictionary to store costs for each agent per episode.
        """
        self._steps_adjusted_per_episode: int
        """ Total number of `act` steps taken per episode. A crashed episode is 
        assumed to have taken `_MAX_STEPS`.
        """
        self._completion: Dict[str, _Completion]
        """ A dictionary storing the completion state of each agent.
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

    def store(self, infos: Dict[str, Any], dones: Dict[str, bool]):
        # Only count steps in which an ego agent was present.
        if len(infos) == 0:
            return 
        self._counts.steps += 1
        self._steps_adjusted_per_episode += 1

        for agent_name, agent_info in infos.items():
            agent_obs = agent_info["env_obs"]
            agent_done = dones[agent_name]

            # Compute all cost functions.
            for cost_func in self._cost_funcs[agent_name].values():
                res = cost_func(agent_obs)
                for cost_name, cost_val in res.items():
                    self._cost_per_episode[agent_name][cost_name] = cost_val

            # If done, categorize agent's completion state.
            if agent_done:
                self._completion[agent_name] = _reason(obs=agent_obs)

        if dones["__all__"] == True:
            # Update counts.
            self._counts.episodes += 1
            num_crashes = operator.countOf(
                self._completion.values(), _Completion.Crashed
            )
            if num_crashes > 0:
                self._counts.crashes += num_crashes / self._num_agents
                self._steps_adjusted_per_episode = _MAX_STEPS
            self._counts.steps_adjusted += self._steps_adjusted_per_episode
            self._counts.episode_agents += self._num_agents

            # Transfer episode costs from all agents into a single running total costs.
            for agent_name, costs in self._cost_per_episode.items():
                for cost_name, cost_val in costs.items():
                    new_val = getattr(self._costs, cost_name) + cost_val
                    setattr(self._costs, cost_name, new_val)

            # Reset cost functions and episodic costs.
            self._reinit()

    def results(self):
        return self._counts, self._costs


class _Completion(Enum):
    Goal = 0
    """Agent achieved its goal."""
    Crashed = 1
    """ Agent becomes done due to collision, driving off road, or reaching max
    episode steps.
    """


def _reason(obs) -> _Completion:
    if obs.events.reached_goal:
        return _Completion.Goal
    elif len(obs.events.collisions) > 0 or \
        obs.events.off_road or \
        obs.events.reached_max_episode_steps:
        return _Completion.Crashed
    else:
        raise Exception(
            f"Unsupported agent done reason. Events: {obs.events}."
        )


@dataclass
class Counts:
    crashes: float = 0
    """ Total number of crashed episodes. An episode is considered crashed if
    an agent becomes done due to collisions, driving off road, or reaching 
    max episode steps. Fractional values occur when only some agents crashes
    the episode in a multi-agent case.
    """
    episodes: int = 0
    """ Total number of episodes.
    """
    episode_agents: int = 0
    """ Total number of equivalent episodes. For an n-agent scenario, 
    n-episodes are added to the total.
    """
    steps: int = 0
    """ Total number of `act` steps taken.
    """
    steps_adjusted: int = 0
    """ Total number of `act` steps taken. Any crashed episode is assumed to
    have taken `_MAX_STEPS`.
    """
