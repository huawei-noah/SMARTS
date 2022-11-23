# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

from smarts.env.wrappers.metric.costs import Costs
from smarts.env.wrappers.metric.counts import Counts
from smarts.env.wrappers.metric.scores import Scores

import gym
import numpy as np

class Metrics(gym.Wrapper):
    """Computes metrics of an agent's performance in a SMARTS environment.
    """

    def __init__(self, env: gym.Env):
        """Sets identical action space, denoted by ``space``, for all agents.
        """
        super().__init__(env)
        self._cur_scen=None
        self._cur_agents=None
        self._records={}
        #     scen_name: {agent_name: metrics},
        #     scen_name: {agent_name: metrics},
        # }

    def action(self, action):
        """Adapts the action input to the wrapped environment.

        Note: Users should not directly call this method.
        """
        wrapped_act = self._wrapper(action)
        return wrapped_act


    def step(self, action:Dict[str,Any]):
        obs, rewards, dones, info = super().step(action)

        # Only count steps in which an ego agent was present.
        if len(obs) == 0:
            return obs, rewards, dones, info

        self._counts.steps += 1
        self._steps_adjusted_per_episode += 1

        for agent_name, agent_obs in obs.items():
            self._records[self._cur_scen][agent_name].counts.steps += 1
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

        return obs, rewards, dones, info

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

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self._cur_scen=(super().scenario_log)["scenario_map"]
        self._cur_agents=super().agent_specs.keys()
        if self._cur_scen not in self._records:
            self._records[self._cur_scen] = {
                agent : _Record(
                    counts=Counts(),
                    costs=Costs(),
                    scores=Scores()
                )
                for agent in self._cur_agents
            }
        return obs

    def score(self)->:
        score = {}
        for scen, record in self._records.items():
            score = {

            }
        score = {
            for self._records
        }
        return self._records


@dataclass
def Score:


@dataclass
class _Record:
    counts: Counts
    costs: Costs
    scores: Scores

