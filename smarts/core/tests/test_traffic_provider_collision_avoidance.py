# MIT License
#
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
import pytest

# TODO: Rename temp_scenario(...)
from helpers.scenario import temp_scenario

import smarts.sstudio.types as t
from smarts.core.local_traffic_provider import LocalTrafficProvider
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.sstudio import gen_scenario


@pytest.fixture(params=["SUMO", "SMARTS"])
def traffic_sim(request):
    return getattr(request, "param", "SUMO")


@pytest.fixture
def scenarios(traffic_sim):
    with temp_scenario(name="6lane", map="maps/t.net.xml") as scenario_root:
        traffic = t.Traffic(
            engine=traffic_sim,
            flows=[
                t.Flow(
                    route=t.Route(
                        begin=("edge-west-WE", 0, 10),
                        end=("edge-east-WE", 1, "max"),
                    ),
                    repeat_route=True,
                    rate=400,
                    actors={t.TrafficActor("car"): 1},
                ),
                t.Flow(
                    route=t.Route(
                        begin=("edge-south-SN", 1, 10),
                        end=("edge-west-EW", 1, "max"),
                    ),
                    repeat_route=True,
                    rate=400,
                    actors={t.TrafficActor("car"): 1},
                ),
            ],
        )

        gen_scenario(
            t.Scenario(traffic={"all": traffic}),
            output_dir=scenario_root,
        )

        yield Scenario.variations_for_all_scenario_roots([str(scenario_root)], [])


@pytest.fixture
def smarts(traffic_sim):
    traffic_sims = (
        [LocalTrafficProvider()]
        if traffic_sim == "SMARTS"
        else [SumoTrafficSimulation()]
    )
    smarts = SMARTS({}, traffic_sims=traffic_sims)
    yield smarts
    smarts.destroy()


# TODO: Consider a higher-level DSL syntax to fulfill these tests
@pytest.mark.parametrize("traffic_sim", ["SUMO", "SMARTS"], indirect=True)
def test_collision_avoidance_in_intersection(smarts, scenarios, traffic_sim):
    """Ensure that traffic providers can manage basic collision avoidance around an intersection."""
    scenario = next(scenarios)
    smarts.reset(scenario)

    for _ in range(1000):
        smarts.step({})
        assert not smarts._vehicle_collisions
