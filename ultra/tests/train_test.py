import unittest, ray, os
from ultra.baselines.ppo.policy import PPOPolicy
from ultra.src.train import train


class TrainTest(unittest.TestCase):
    def test_train_cli(self):
        try:
            os.system(
                "python ultra/src/train.py --task 00 --level easy --polic ULTRA_PPO --episodes 1"
            )
        except Exception as err:
            print(err)
            self.assertTrue(False)

    def test_train(self):
        seed = 2
        policy_class = PPOPolicy
        ray.init(ignore_reinit_error=True)

        try:
            ray.get(
                train.remote(
                    task=("00", "easy"),
                    policy_class=policy_class,
                    num_episodes=2,
                    eval_info={
                        "eval_rate": 1000,
                        "eval_episodes": 2,
                        "policy_class": policy_class,
                    },
                    timestep_sec=0.1,
                    headless=True,
                    etag="Test",
                    seed=2,
                )
            )
            self.assertTrue(True)
            ray.shutdown()
        except ray.exceptions.WorkerCrashedError as err:
            print(err)
            self.assertTrue(False)
            ray.shutdown()
