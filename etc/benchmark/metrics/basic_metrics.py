# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import math
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from benchmark.metrics import MetricHandler
from benchmark.utils.episode_log import BasicEpisodeLog


@dataclass
class BehaviorMetric:
    safety: float = 0.0
    agility: float = 0.0
    stability: float = 0.0
    control_diversity: float = 0.0
    cut_in_ratio: float = 0.0

    def __post_init__(self):
        self._features = ["Safety", "Agility", "Stability", "Diversity"]

    @property
    def features(self):
        return self._features

    def _compute_safety_index(self, agent_collisions_seq):
        non_collision = []
        for agent_collisions in agent_collisions_seq:
            sum_collision = sum(agent_collisions.values())
            non_collision.append(int(sum_collision > 0))
        self.safety = np.mean(non_collision)

    def _compute_agility_index(self, agent_speed_seq):
        # compute average speed
        average_speed = []
        for agent_speed in agent_speed_seq:
            # compute ave speed of this group
            mean_speed_list = []
            for speed_list in agent_speed.values():
                mean_speed_list.append(np.mean(speed_list))
            average_speed.append(np.mean(mean_speed_list))
        self.agility = np.mean(average_speed)

    def _compute_stability_index(self, agent_central_dist_seq):
        agent_vars = defaultdict(lambda: [])
        for agent_central_dist in agent_central_dist_seq:
            for agent, central_dist_list in agent_central_dist.items():
                # compute the variance of dist
                agent_vars[agent].append(np.var(central_dist_list))

        # compute stability for each agent
        stability = []
        for agent, vars in agent_vars.items():
            stability.append(np.mean(vars))

        self.stability = np.mean(stability)

    def _compute_diversity_index(self, agent_operation_seq):
        # only work for discrete actions
        agent_action_prob = defaultdict(lambda: defaultdict(lambda: 1e-3))
        for agent_operation in agent_operation_seq:
            for agent, operations in agent_operation.items():
                for ope in operations:
                    agent_action_prob[agent][ope] += 1
        entropy = []
        for agent, operation_weights in agent_action_prob.items():
            total = max(1, sum(operation_weights.values()))
            operation_probs = dict(
                map(lambda kv: (kv[0], kv[1] / total), operation_weights.items())
            )
            # calculate the entropy
            entropy.append(sum([v * math.log(v) for v in operation_probs.values()]))

        self.control_diversity = np.mean(entropy)

    def compute(self, handler: MetricHandler) -> Tuple[Dict[str, Any], List[str]]:
        """Compute behavior analysis metrics

        Parameters
        ----------
        handler
            MetricHandler, the metric handler instance

        Returns
        -------
        metrics
            a dictionary mapping instance
        """

        episode_logs_mapping: Dict[str, List[BasicEpisodeLog]] = handler.logs_mapping
        results = {}

        for algorithm, episode_logs in episode_logs_mapping.items():
            self._compute_agility_index([log.ego_speed for log in episode_logs])
            self._compute_safety_index([log.num_collision for log in episode_logs])
            self._compute_stability_index(
                [log.distance_to_center for log in episode_logs]
            )
            self._compute_diversity_index([log.operations for log in episode_logs])

            results[algorithm] = OrderedDict(
                {
                    "Agility": self.agility,
                    "Safety": self.safety,
                    "Stability": self.stability,
                    "Diversity": self.control_diversity,
                }
            )

        print("results:\n", results)

        _result = list(results.values())[0]
        metric_keys = list(_result.keys())

        return results, metric_keys


@dataclass
class GameTheoryMetric:
    collaborative: Any
    competitive: Any


@dataclass
class PerformanceMetric:
    collision_rate: float
    completion_rate: float
    generalization: Any
