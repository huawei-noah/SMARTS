# MIT License
#
# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
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
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Set, Tuple

from smarts.core.actor_capture_manager import ActorCaptureManager
from smarts.core.condition_state import ConditionState
from smarts.core.plan import NavigationMission
from smarts.sstudio.sstypes import IdEntryTactic

if TYPE_CHECKING:
    from smarts.core import scenario
    from smarts.core.smarts import SMARTS


class IdActorCaptureManager(ActorCaptureManager):
    """The base for managers that handle transition of control of actors."""

    def __init__(self) -> None:
        self._log = logging.getLogger(self.__class__.__name__)
        self._actor_for_agent: Dict[str, Tuple[str, NavigationMission]] = {}

    def step(self, sim: SMARTS):
        if not (
            sim.agent_manager.pending_agent_ids
            | sim.agent_manager.pending_social_agent_ids
        ):
            return

        social_vehicle_ids: Set[str] = {
            v_id
            for v_id in sim.vehicle_index.social_vehicle_ids()
            if not sim.vehicle_index.vehicle_is_hijacked(v_id)
        }

        used_actors = []
        for actor_id, agent_id, mission in (
            (a, b, m)
            for a, (b, m) in self._actor_for_agent.items()
            if m.start_time <= sim.elapsed_sim_time and a in social_vehicle_ids
        ):
            entry_tactic = mission.entry_tactic
            assert isinstance(entry_tactic, IdEntryTactic)
            vehicle = sim.vehicle_index.vehicle_by_id(actor_id)
            condition_kwargs = ActorCaptureManager._gen_all_condition_kwargs(
                agent_id,
                mission,
                sim,
                vehicle.state if vehicle is not None else None,
                entry_tactic.condition.requires,
            )
            condition_result = entry_tactic.condition.evaluate(**condition_kwargs)
            if condition_result == ConditionState.EXPIRED:
                self._log.warning(
                    f"Actor aquisition skipped for `{agent_id}` scheduled to start between "
                    + f"`Condition `{entry_tactic.condition}` has expired with no vehicle. "
                    + f"Missing actor: `{entry_tactic.actor_id}`"
                )
                used_actors.append(actor_id)
                sim.agent_manager.teardown_ego_agents({agent_id})
                continue
            if not condition_result:
                continue
            vehicle = self._take_existing_vehicle(
                sim,
                actor_id,
                agent_id,
                mission,
                social=agent_id in sim.agent_manager.pending_social_agent_ids,
            )
            if vehicle is None:
                continue
            used_actors.append(actor_id)
        for actor_id in used_actors:
            del self._actor_for_agent[actor_id]

    def reset(self, scenario, sim: SMARTS):
        self._actor_for_agent.clear()
        missions: Dict[str, NavigationMission] = scenario.missions
        cancelled_agents = set()
        for agent_id, mission in missions.items():
            if mission is None:
                continue
            entry_tactic = mission.entry_tactic
            if not isinstance(entry_tactic, IdEntryTactic):
                continue
            vehicle = sim.vehicle_index.vehicle_by_id(entry_tactic.actor_id, None)
            condition_kwargs = ActorCaptureManager._gen_all_condition_kwargs(
                agent_id,
                mission,
                sim,
                vehicle.state if vehicle is not None else None,
                entry_tactic.condition.requires,
            )
            condition_result = entry_tactic.condition.evaluate(**condition_kwargs)
            if condition_result == ConditionState.EXPIRED:
                self._log.warning(
                    f"Actor acquisition skipped for `{agent_id}` scheduled to start with"
                    + f"`Condition:{entry_tactic.condition}` because simulation skipped to "
                    f"`simulation time: {sim.elapsed_sim_time}`. Missing actor: `{entry_tactic.actor_id}`"
                )
                cancelled_agents.add(agent_id)
                continue
            self._actor_for_agent[entry_tactic.actor_id] = (agent_id, mission)
        if len(cancelled_agents) > 0:
            sim.agent_manager.teardown_ego_agents(cancelled_agents)

    def teardown(self):
        pass
