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


from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

from smarts.core.colors import Colors
from smarts.sstudio.types.actor.social_agent_actor import SocialAgentActor
from smarts.sstudio.types.bubble import Bubble
from smarts.sstudio.types.dataset import TrafficHistoryDataset
from smarts.sstudio.types.map_spec import MapSpec
from smarts.sstudio.types.mission import EndlessMission, Mission
from smarts.sstudio.types.traffic import Traffic
from smarts.sstudio.types.zone import RoadSurfacePatch


@dataclass(frozen=True)
class ScenarioMetadata:
    """Scenario data that does not have influence on simulation."""

    actor_of_interest_re_filter: str
    """Vehicles with names that match this pattern are vehicles of interest."""
    actor_of_interest_color: Colors
    """The color that the vehicles of interest should have."""


@dataclass(frozen=True)
class Scenario:
    """The sstudio scenario representation."""

    map_spec: Optional[MapSpec] = None
    """Specifies the road map."""
    traffic: Optional[Dict[str, Traffic]] = None
    """Background traffic vehicle specification."""
    ego_missions: Optional[Sequence[Union[Mission, EndlessMission]]] = None
    """Ego agent missions."""
    social_agent_missions: Optional[
        Dict[str, Tuple[Sequence[SocialAgentActor], Sequence[Mission]]]
    ] = None
    """
    Actors must have unique names regardless of which group they are assigned to.
    Every dictionary item ``{group: (actors, missions)}`` gets selected from simultaneously.
    If actors > 1 and missions = 0 or actors = 1 and missions > 0, we cycle
    through them every episode. Otherwise actors must be the same length as 
    missions.
    """
    bubbles: Optional[Sequence[Bubble]] = None
    """Capture bubbles for focused social agent simulation."""
    friction_maps: Optional[Sequence[RoadSurfacePatch]] = None
    """Friction coefficient of patches of road surface."""
    traffic_histories: Optional[Sequence[Union[TrafficHistoryDataset, str]]] = None
    """Traffic vehicles trajectory dataset to be replayed."""
    scenario_metadata: Optional[ScenarioMetadata] = None
    """"Scenario data that does not have influence on simulation."""

    def __post_init__(self):
        def _get_name(item):
            return item.name

        if self.social_agent_missions is not None:
            groups = [k for k in self.social_agent_missions]
            for group, (actors, _) in self.social_agent_missions.items():
                for o_group in groups:
                    if group == o_group:
                        continue
                    if intersection := set.intersection(
                        set(map(_get_name, actors)),
                        map(_get_name, self.social_agent_missions[o_group][0]),
                    ):
                        raise ValueError(
                            f"Social agent mission groups `{group}`|`{o_group}` have overlapping actors {intersection}"
                        )
