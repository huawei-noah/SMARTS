import unittest, ray
from ultra.baselines.ppo.policy import PPOPolicy
from ultra.env.agent_spec import BaselineAgentSpec
from smarts.core.controllers import ActionSpaceType
import gym

AGENT_ID = "001"
timestep_sec = 0.1
seed = 2
task_id = "00"
task_level = "easy"


class EnvTest(unittest.TestCase):
    def test_scenario_task(self):
        @ray.remote(max_calls=1, num_gpus=0, num_cpus=1)
        def run_experiment():
            agent, env = prepare_test_env_agent()
            task_id1, task_level1 = env.scenario_info
            env.close()
            return task_id1, task_level1

        ray.init(ignore_reinit_error=True)
        task_id1, task_level1 = ray.get(run_experiment.remote())
        ray.shutdown()
        self.assertTrue(task_id1 == task_id)
        self.assertTrue(task_level1 == task_level)

    def test_headless(self):
        @ray.remote(max_calls=1, num_gpus=0, num_cpus=1)
        def run_experiment():
            agent, env = prepare_test_env_agent(headless=False)
            headless1 = env.headless
            env.close()

            agent, env = prepare_test_env_agent(headless=True)
            headless2 = env.headless
            env.close()
            return headless1, headless2

        ray.init(ignore_reinit_error=True)
        headless1, headless2 = ray.get(run_experiment.remote())
        ray.shutdown()
        self.assertFalse(headless1)
        self.assertTrue(headless2)

    def test_timestep_sec(self):
        @ray.remote(max_calls=1, num_gpus=0)
        def run_experiment():
            agent, env = prepare_test_env_agent(headless=False)
            timestep_sec1 = env.timestep_sec
            env.close()
            return timestep_sec1

        ray.init(ignore_reinit_error=True)
        timestep_sec1 = ray.get(run_experiment.remote())
        ray.shutdown()
        self.assertTrue(timestep_sec1 == timestep_sec)


# other attributes are not set
#'action_space', 'close', 'get_task', 'headless', 'info', 'metadata', 'observation_space', 'render', 'reset', 'reward_range', 'scenario_info', 'scenario_log', 'scenarios', 'seed', 'spec', 'step', 'timestep_sec', 'unwrapped'


def prepare_test_env_agent(headless=True):
    timestep_sec = 0.1
    # [throttle, brake, steering]
    policy_class = PPOPolicy
    spec = BaselineAgentSpec(
        action_type=ActionSpaceType.Continuous,
        policy_class=policy_class,
        max_episode_steps=10,
    )
    env = gym.make(
        "ultra.env:ultra-v0",
        agent_specs={AGENT_ID: spec},
        scenario_info=("00", "easy"),
        headless=headless,
        timestep_sec=timestep_sec,
        seed=seed,
    )
    agent = spec.build_agent()
    return agent, env
