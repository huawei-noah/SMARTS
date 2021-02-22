import unittest, ray, os, sys
from ultra.rllib_train import train
import shutil

AGENT_ID = "001"
seed = 2


class RLlibTrainTest(unittest.TestCase):
    def test_rllib_train_cli(self):
        log_dir = "tests/rllib_results"
        try:
            os.system(
                f"python ultra/rllib_train.py --task 00 --level easy --episodes 1 --max-samples 200 --headless True --log-dir {log_dir}"
            )
        except Exception as err:
            print(err)
            self.assertTrue(False)

        if os.path.exists(log_dir):
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    def test_rllib_train_method(self):
        log_dir = "tests/rllib_results"
        try:
            ray.init()
            train(
                task=("00", "easy"),
                num_episodes=1,
                eval_info={
                    "eval_rate": 2,
                    "eval_episodes": 1,
                },
                timestep_sec=0.1,
                headless=True,
                seed=2,
                max_samples=200,
                log_dir=log_dir,
            )
        except Exception as err:
            print(err)
            self.assertTrue(False)

        if os.path.exists(log_dir):
            self.assertTrue(True)
        else:
            self.assertTrue(False)
