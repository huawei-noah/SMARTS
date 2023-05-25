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


import collections
import enum
import warnings
from dataclasses import dataclass
from enum import IntEnum
from functools import cached_property
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple, Type, Union

from smarts.core.colors import Colors
from smarts.sstudio.types.actor.social_agent_actor import SocialAgentActor
from smarts.sstudio.types.bubble import Bubble
from smarts.sstudio.types.dataset import TrafficHistoryDataset
from smarts.sstudio.types.map_spec import MapSpec
from smarts.sstudio.types.mission import EndlessMission, Mission
from smarts.sstudio.types.traffic import Traffic
from smarts.sstudio.types.zone import RoadSurfacePatch


class MetadataFields(IntEnum):
    actor_of_interest_color = enum.auto()
    actor_of_interest_re_filter = enum.auto()
    scenario_difficulty = enum.auto()
    scenario_length = enum.auto()


@dataclass(frozen=True)
class ScenarioMetadata:
    """Deprecated. Scenario data that does not have influence on simulation."""

    actor_of_interest_re_filter: str
    """Vehicles with names that match this pattern are vehicles of interest."""
    actor_of_interest_color: Colors
    """The color that the vehicles of interest should have."""

    def __post_init__(self):
        warnings.warn(f"Use of {ScenarioMetadata.__name__} is deprecated.")


class StandardMetadata(collections.Mapping):
    """Metadata that does not have direct on simulation."""

    def __init__(
        self,
        metadata: Optional[Dict[Union[str, MetadataFields], Any]] = None,
        actor_of_interest_re_filter: Optional[str] = None,
        actor_of_interest_color: Optional[Colors] = Colors.Blue,
        scenario_difficulty: Optional[int] = None,
        scenario_length: Optional[float] = None,
    ) -> None:
        if metadata is None:
            metadata = {}
        basic_standard_metadata = {
            MetadataFields.actor_of_interest_color: actor_of_interest_color,
            MetadataFields.actor_of_interest_re_filter: actor_of_interest_re_filter,
            MetadataFields.scenario_difficulty: scenario_difficulty,
            MetadataFields.scenario_length: scenario_length,
        }
        self._standard_metadata = tuple(
            (
                setting_key.name
                if isinstance(setting_key, MetadataFields)
                else setting_key,
                setting_value,
            )
            for setting_key, setting_value in {
                **metadata,
                **basic_standard_metadata,
            }.items()
            if setting_value is not None
        )

    def __iter__(self) -> Iterator:
        return iter(self._standard_metadata)

    def __len__(self) -> int:
        return len(self._standard_metadata)

    def __getitem__(self, __key: Any) -> Any:
        if isinstance(__key, MetadataFields):
            __key = __key.name
        return self._dict_metadata[__key]

    def get(self, __key, __default=None):
        """Retrieve the value or a default.

        Args:
            __key (Any): The key to find.
            __default (Any, optional): The default if the key does not exist. Defaults to None.

        Returns:
            Optional[Any]: The value or default.
        """
        if isinstance(__key, MetadataFields):
            __key = __key.name
        return self._dict_metadata.get(__key, __default)

    # def __getattr__(self, __name: str) -> Any:
    #     return self._dict_metadata.get(__name)

    def __hash__(self) -> int:
        return self._hash_id

    @cached_property
    def _hash_id(self) -> int:
        return hash(frozenset(self._standard_metadata))

    @cached_property
    def _dict_metadata(self) -> Dict[str, Any]:
        return dict(self._standard_metadata)


@dataclass(frozen=True)
class Scenario:
    """The Scenario Studio (`sstudio`) scenario representation."""

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
    scenario_metadata: Optional[StandardMetadata] = StandardMetadata({})
    """"Scenario data that does not have influence on simulation."""

    def __post_init__(self):
        def _get_name(item):
            return item.name

        if isinstance(self.scenario_metadata, ScenarioMetadata):
            object.__setattr__(
                self,
                "scenario_metadata",
                StandardMetadata(
                    actor_of_interest_color=self.scenario_metadata.actor_of_interest_color,
                    actor_of_interest_re_filter=self.scenario_metadata.actor_of_interest_re_filter,
                ),
            )

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
