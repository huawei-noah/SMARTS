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
try:
    import ray
except Exception as e:
    from examples import RayException

    raise RayException.required_to("stress_sumo.py")

from smarts.core.scenario import Scenario
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation


@ray.remote
def spawn_sumo(worker_idx, batch_id):
    sumo_sim = SumoTrafficSimulation(headless=True)

    scenarios_iterator = Scenario.scenario_variations(
        ["scenarios/loop"],
        ["Agent007"],
    )
    sumo_sim.setup(Scenario.next(scenarios_iterator, f"{batch_id}-{worker_idx}"))
    sumo_sim.teardown()


remotes_per_iteration = 32
ray.init()
for i in range(100):
    ray.wait(
        [spawn_sumo.remote(r, i) for r in range(remotes_per_iteration)],
        num_returns=remotes_per_iteration,
    )
ray.shutdown()
