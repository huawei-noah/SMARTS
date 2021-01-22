import math
import numpy as np

from typing import Dict, Any, List
from collections import defaultdict

from dataclasses import dataclass

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

    def compute(self, handler: MetricHandler) -> Dict[str, float]:
        """ Compute behavior analysis metrics

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

            results[algorithm] = {
                "Agility": self.agility,
                "Safety": self.safety,
                "Stability": self.stability,
                "Diversity": self.control_diversity,
            }


@dataclass
class GameTheoryMetric:
    collaborative: Any
    competitive: Any


@dataclass
class PerformanceMetric:
    collision_rate: float
    completion_rate: float
    generalization: Any
