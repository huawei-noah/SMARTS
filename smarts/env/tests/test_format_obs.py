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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import dataclasses

import gym
import pytest

from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, Waypoints
from smarts.core.controllers import ActionSpaceType
from smarts.env.wrappers.format_obs import FormatObs, get_spaces, intrfc_to_stdobs


def _intrfcs_init():
    return [
        [{"accelerometer": True}, {"accelerometer": False}],
        [{"drivable_area_grid_map": True}] * 2,
        [{"lidar": True}] * 2,
        [{"neighborhood_vehicles": True}] * 2,
        [{"ogm": True}] * 2,
        [{"rgb": True}] * 2,
        [{"waypoints": Waypoints(lookahead=1)}] * 2,
        [{"neighborhood_vehicles": True, "waypoints": Waypoints(lookahead=1)}] * 2,
    ]


def _intrfcs_obs():
    base_intrfc = {
        "accelerometer": True,
        "drivable_area_grid_map": True,
        "lidar": True,
        "neighborhood_vehicles": True,
        "ogm": True,
        "rgb": True,
        "waypoints": Waypoints(lookahead=1),
    }

    return [
        [base_intrfc] * 2,
        [dict(base_intrfc, **{"accelerometer": False})] * 2,
        [dict(base_intrfc, **{"waypoints": Waypoints(lookahead=50)})] * 2,
    ]


def _make_agent_specs(intrfcs):
    base_intrfc = AgentInterface(
        action=ActionSpaceType.Lane,
        accelerometer=False,
        drivable_area_grid_map=False,
        lidar=False,
        neighborhood_vehicles=False,
        ogm=False,
        rgb=False,
        waypoints=False,
    )

    return {
        "AGENT_"
        + agent_id: AgentSpec(
            interface=dataclasses.replace(base_intrfc, **intrfc),
            agent_builder=lambda: Agent.from_function(lambda _: "keep_lane"),
        )
        for agent_id, intrfc in zip(["001", "002"], intrfcs)
    }


@pytest.fixture
def make_env(request):
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=["scenarios/figure_eight"],
        agent_specs=_make_agent_specs(request.param),
        headless=True,
        visdom=False,
    )

    yield env, request.param
    env.close()


@pytest.fixture(scope="module")
def spaces():
    return get_spaces()


@pytest.mark.parametrize("make_env", _intrfcs_init(), indirect=True)
def test_init(make_env, spaces):
    base_env, cur_intrfcs = make_env

    # Test wrapping an env with non-identical agent interfaces
    base_intrfcs = [
        agent_spec.interface for agent_spec in base_env.agent_specs.values()
    ]
    if not all(intrfc == base_intrfcs[0] for intrfc in base_intrfcs):
        with pytest.raises(AssertionError):
            env = FormatObs(env=base_env)
            env.close()
        return

    env = FormatObs(env=base_env)

    # Test whether observation space of wrapped env is created properly
    agent_id = next(iter(base_env.agent_specs.keys()))
    rcv_space = env.observation_space
    rcv_space_keys = set([key for key in rcv_space[agent_id]])

    basic, _ = spaces
    des_space_keys = set(basic.keys())
    opt_space_keys = [
        intrfc_to_stdobs(intrfc)
        for intrfc, val in cur_intrfcs[0].items()
        if val and intrfc_to_stdobs(intrfc)
    ]
    des_space_keys.update(opt_space_keys)
    if "waypoints" in opt_space_keys and "neighbors" in opt_space_keys:
        des_space_keys.update(["ttc"])

    assert rcv_space_keys == des_space_keys

    env.close()


def _check_observation(
    rcv_space,
    obs,
):
    for field1 in obs.__dataclass_fields__:
        val1 = getattr(obs, field1)
        if isinstance(val1, dict):
            for field2, val2 in val1.items():
                assert val2.shape == rcv_space[field1][field2].shape
                assert val2.dtype == rcv_space[field1][field2].dtype
        else:
            assert val1.shape == rcv_space[field1].shape
            assert val1.dtype == rcv_space[field1].dtype

    return


@pytest.mark.parametrize("make_env", _intrfcs_obs(), indirect=True)
def test_observation(make_env):
    base_env, _ = make_env
    env = FormatObs(env=base_env)
    rcv_space = env.observation_space

    # Test whether returned observation matches observation space
    obs = env.reset()
    for agent_id, ob in obs.items():
        _check_observation(rcv_space[agent_id], ob)

    env.close()
