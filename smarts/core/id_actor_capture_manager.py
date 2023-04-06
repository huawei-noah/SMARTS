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
import logging
from typing import Dict, Optional, Set, Tuple

from smarts.core.actor_capture_manager import ActorCaptureManager
from smarts.core.plan import Mission
from smarts.core.vehicle import Vehicle
from smarts.sstudio.types import IdEntryTactic


class IdActorCaptureManager(ActorCaptureManager):
    """The base for managers that handle transition of control of actors."""

    def __init__(self) -> None:
        self._log = logging.getLogger(self.__class__.__name__)
        self._actor_for_agent: Dict[str, Tuple[str, Mission]] = {}

    def step(self, sim):
        from smarts.core.smarts import SMARTS

        assert isinstance(sim, SMARTS)

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
            assert isinstance(mission.entry_tactic, IdEntryTactic)
            patience_expiry = mission.start_time + mission.entry_tactic.patience
            if sim.elapsed_sim_time > patience_expiry:
                self._log.error(
                    f"Actor aquisition skipped for `{agent_id}` scheduled to start between "
                    + f"`{mission.start_time}` and `{patience_expiry}` has expired with no vehicle."
                    f"`simulation time: {sim.elapsed_sim_time}`"
                )
                used_actors.append(actor_id)
                sim.agent_manager.teardown_ego_agents({agent_id})
                continue
            vehicle: Optional[Vehicle] = self._take_existing_vehicle(
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

    def reset(self, scenario, sim):
        from smarts.core.smarts import SMARTS

        assert isinstance(sim, SMARTS)
        from smarts.core.scenario import Scenario

        assert isinstance(scenario, Scenario)

        self._actor_for_agent.clear()
        missions: Dict[str, Mission] = scenario.missions
        cancelled_agents = set()
        for agent_id, mission in missions.items():
            if mission is None:
                continue
            if not isinstance(mission.entry_tactic, IdEntryTactic):
                continue
            patience_expiry = mission.start_time + mission.entry_tactic.patience
            if sim.elapsed_sim_time > patience_expiry:
                self._log.error(
                    f"ID actor capture entry tactic failed for `{agent_id}` scheduled to start between "
                    + f"`{mission.start_time}` and `{patience_expiry}` because simulation skipped to "
                    f"`simulation time: {sim.elapsed_sim_time}`"
                )
                cancelled_agents.add(agent_id)
                continue
            self._actor_for_agent[mission.entry_tactic.actor_id] = (agent_id, mission)
        if len(cancelled_agents) > 0:
            sim.agent_manager.teardown_ego_agents(cancelled_agents)

    def teardown(self):
        pass
