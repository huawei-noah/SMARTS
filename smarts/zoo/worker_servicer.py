# MIT License

# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import cloudpickle
import grpc
import logging
import os
import time

from smarts.core import action as act_util
from smarts.core import observation as obs_util
from smarts.proto import action_pb2, worker_pb2, worker_pb2_grpc

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(f"worker_servicer.py - pid({os.getpid()})")


class WorkerServicer(worker_pb2_grpc.WorkerServicer):
    """Provides methods that implement functionality of Worker Servicer."""

    def __init__(self):
        self._agents = None
        self._agent_specs = None
        self._agent_action_spaces = None

    def build(self, request, context):
        self._agent_specs = cloudpickle.loads(request.payload)
        self._agents = {
            agent_id: agent_spec.build_agent()
            for agent_id, agent_spec in self._agent_specs.items()
        }
        self._agent_action_spaces = {
            agent_id: agent_spec.interface.action
            for agent_id, agent_spec in self._agent_specs.items()
        }

        return worker_pb2.Status()

    def act(self, request, context):
        if self._agents == None or self._agent_specs == None:
            context.set_details(f"Remote agent not built yet.")
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            return action_pb2.Actions()

        obs = obs_util.proto_to_observations(request)
        adapted_action = {
            agent_id: obs_to_act(
                self._agents[agent_id], self._agent_specs[agent_id], agent_obs
            )
            for agent_id, agent_obs, in obs.items()
        }

        return action_pb2.ActionsBoid(
            boids=act_util.actions_to_proto(self._agent_action_spaces, adapted_action)
        )

    def destroy(self):
        log.debug(
            f"worker_servicer - pid({os.getpid()}), shutting down rpc-worker process."
        )


def obs_to_act(agent, agent_spec, obs):
    adapted_obs = agent_spec.observation_adapter(obs)
    action = agent.act(adapted_obs)
    adapted_action = agent_spec.action_adapter(action)
    return adapted_action
