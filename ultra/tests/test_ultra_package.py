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
import os, shutil, ray

from ultra.train import train
from ultra.scenarios.generate_scenarios import build_scenarios

if __name__ == "__main__":
    save_dir = "tests/scenarios/maps/no-traffic/"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    build_scenarios(
        task="task00",
        level_name="eval_test",
        stopwatcher_behavior=None,
        stopwatcher_route=None,
        root_path="tests/scenarios",
        save_dir=save_dir,
    )

    policy_class = "ultra.baselines.sac:sac-v0"

    ray.shutdown()
    try:
        ray.init(ignore_reinit_error=True)
        ray.wait(
            [
                train.remote(
                    scenario_info=("00", "eval_test"),
                    policy_class=policy_class,
                    num_episodes=1,
                    eval_info={"eval_rate": 1000, "eval_episodes": 2,},
                    timestep_sec=0.1,
                    headless=True,
                    seed=2,
                    log_dir="ultra/tests/logs",
                )
            ]
        )
        ray.shutdown()
    except ray.exceptions.WorkerCrashedError as err:
        print(err)
        ray.shutdown()
