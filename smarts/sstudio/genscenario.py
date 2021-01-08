# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
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
"""This script provides a Python interface to generate scenario artifacts. This includes
route files (sumo \\*.rou.xml), missions, and bubbles.
"""

import collections
import itertools
import logging
import os
import pickle
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence, Tuple, Union

from . import types
from .generators import TrafficGenerator

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__file__)


def gen_scenario(
    scenario: types.Scenario, output_dir: Path, seed: int = 42, ovewrite: bool = False,
):
    """This is now the preferred way to generate a scenario. Instead of calling the
    gen_* methods directly, we provide this higher-level abstraction that takes care
    of the sub-calls.
    """
    # XXX: For now this simply coalesces the sub-calls but in the future this allows
    #      us to simplify our serialization between SStudio and SMARTS.

    output_dir = str(output_dir)

    if scenario.traffic:
        for name, traffic in scenario.traffic.items():
            gen_traffic(
                scenario=output_dir,
                traffic=traffic,
                name=name,
                seed=seed,
                overwrite=ovewrite,
            )

    if scenario.ego_missions:
        missions = []
        for mission in scenario.ego_missions:
            if isinstance(mission, types.GroupedLapMission):
                gen_group_laps(
                    scenario=output_dir,
                    begin=mission.route.begin,
                    end=mission.route.end,
                    grid_offset=mission.offset,
                    used_lanes=mission.lanes,
                    vehicle_count=mission.actor_count,
                    num_laps=mission.num_laps,
                    seed=seed,
                    overwrite=ovewrite,
                )
            else:
                missions.append(mission)

        if missions:
            gen_missions(
                scenario=output_dir, missions=missions, seed=seed, overwrite=ovewrite,
            )

    if scenario.social_agent_missions:
        for name, (actors, missions) in scenario.social_agent_missions.items():
            if not (
                isinstance(actors, collections.Sequence)
                and isinstance(missions, collections.Sequence)
            ):
                raise ValueError("Actors and missions must be sequences")

            gen_social_agent_missions(
                name=name,
                scenario=output_dir,
                social_agent_actor=actors,
                missions=missions,
            )

    if scenario.bubbles:
        gen_bubbles(scenario=output_dir, bubbles=scenario.bubbles)

    if scenario.friction_maps:
        gen_friction_map(scenario=output_dir, surface_patches=scenario.friction_maps)

    if scenario.traffic_histories:
        gen_traffic_histories(scenario=output_dir, histories=scenario.traffic_histories)


def gen_traffic(
    scenario: str,
    traffic: types.Traffic,
    name: str = None,
    output_dir: str = None,
    seed: int = 42,
    overwrite: bool = False,
):
    """Generates the traffic routes for the given scenario. If the output directory is
    not provided, the scenario directory is used. If name is not provided the default is
    "routes".
    """
    assert name != "missions", "The name 'missions' is reserved for missions!"

    output_dir = os.path.join(output_dir or scenario, "traffic")
    os.makedirs(output_dir, exist_ok=True)

    generator = TrafficGenerator(scenario, overwrite=overwrite)
    saved_path = generator.plan_and_save(traffic, name, output_dir, seed=seed)

    if saved_path:
        logger.debug(f"Generated traffic for scenario={scenario}")


def gen_social_agent_missions(
    scenario: str,
    missions: Sequence[types.Mission],
    social_agent_actor: Union[types.SocialAgentActor, Sequence[types.SocialAgentActor]],
    name: str,
    seed: int = 42,
    overwrite: bool = False,
):
    """Generates the social agent missions for the given scenario.

    Args:
        scenario:
            The scenario directory
        missions:
            A sequence of missions for social agents to perform
        social_agent_actor(s):
            The actor(s) to use
        name:
            A short name for this grouping of social agents. Is also used as the name
            of the social agent traffic file
        seed:
            The random seed to use when generating behaviour
        overwrite:
            If to forcefully write over the previous existing output file
    """

    # For backwards compatibility we support both a single value and a sequence
    actors = social_agent_actor
    if not isinstance(actors, collections.Sequence):
        actors = [actors]

    # This doesn't support BoidAgentActor. Here we make that explicit
    if any(isinstance(actor, types.BoidAgentActor) for actor in actors):
        raise ValueError(
            "gen_social_agent_missions(...) can't be called with BoidAgentActor, got:"
            f"{actors}"
        )

    actor_names = [a.name for a in actors]
    if len(actor_names) != len(set(actor_names)):
        raise ValueError(f"Actor names={actor_names} must not contain duplicates")

    output_dir = os.path.join(scenario, "social_agents")
    saved = _gen_missions(
        scenario=scenario,
        missions=missions,
        actors=actors,
        name=name,
        output_dir=output_dir,
        seed=seed,
        overwrite=overwrite,
    )

    if saved:
        logger.debug(f"Generated social agent missions for scenario={scenario}")


def gen_missions(
    scenario: str, missions: Sequence, seed: int = 42, overwrite: bool = False,
):
    """Generates a route file to represent missions (a route per mission). Will create
    the output_dir if it doesn't exist already. The ouput file will be named `missions`.

    Args:
        scenario:
            The scenario directory
        missions:
            A sequence of missions for social agents to perform
        seed:
            The random seed to use when generating behaviour
        overwrite:
            If to forcefully write over the previous existing output file
    """

    saved = _gen_missions(
        scenario=scenario,
        missions=missions,
        actors=[types.TrafficActor(name="car")],
        name="missions",
        output_dir=scenario,
        seed=seed,
        overwrite=overwrite,
    )

    if saved:
        logger.debug(f"Generated missions for scenario={scenario}")


def gen_group_laps(
    scenario: str,
    begin: Tuple[str, int, Any],
    end: Tuple[str, int, Any],
    grid_offset: int,
    used_lanes: int,
    vehicle_count: int,
    num_laps: int = 3,
    seed: int = 42,
    overwrite: bool = False,
):
    """Generates missions that start with a grid offset at the startline and do a number
    of laps until finishing.

    Args:
        scenario:
            The scenario directory
        begin:
            The edge and offset of the first vehicle
        end:
            The edge and offset of the finish-line
        grid_offset:
            The F1 starting line staggered with offset disadvantage imposed per vehicle
        used_lanes:
            The number of lanes used for the starting-line from the innermost lane
        vehicle_count:
            The number of vehicles to use
        num_laps:
            The amount of laps before finishing
    """

    start_edge_id, start_lane, start_offset = begin
    end_edge_id, end_lane, end_offset = end

    missions = []
    for i in range(vehicle_count):
        s_lane = (start_lane + i) % used_lanes
        missions.append(
            types.LapMission(
                types.Route(
                    begin=(start_edge_id, s_lane, start_offset - grid_offset * i,),
                    end=(end_edge_id, (end_lane + i) % used_lanes, end_offset),
                ),
                num_laps=num_laps,
                # route_length=route_length,
            )
        )

    saved = gen_missions(
        scenario=scenario, missions=missions, seed=seed, overwrite=overwrite
    )

    if saved:
        logger.debug(f"Generated grouped lap missions for scenario={scenario}")


def gen_bubbles(scenario: str, bubbles: Sequence[types.Bubble]):
    output_path = os.path.join(scenario, "bubbles.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(bubbles, f)


def gen_friction_map(scenario: str, surface_patches: Sequence[types.RoadSurfacePatch]):
    """Generates friction map file according to the surface patches defined in
    scenario file.
    """
    output_path = os.path.join(scenario, "friction_map.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(surface_patches, f)


def _gen_missions(
    scenario: str,
    missions: Sequence[types.Mission],
    actors: Sequence[types.Actor],
    name: str,
    output_dir: str,
    seed: int = 42,
    overwrite: bool = False,
):
    """Generates a route file to represent missions (a route per mission). Will
    create the output_dir if it doesn't exist already.
    """

    generator = TrafficGenerator(scenario)

    def resolve_mission(mission):
        route = getattr(mission, "route", None)
        kwargs = {}
        if route:
            kwargs["route"] = generator.resolve_route(route)

        task = getattr(mission, "task", None)
        if task:
            kwargs["task"] = _resolve_task(task, generator=generator)

        via = getattr(mission, "via", ())
        if via is not ():
            kwargs["via"] = _resolve_vias(via, generator=generator)

        mission = replace(mission, **kwargs)

        return mission

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, name + ".pkl")

    if os.path.exists(output_path) and not overwrite:
        return False

    _validate_missions(missions)

    missions = [
        types._ActorAndMission(actor=actor, mission=resolve_mission(mission))
        for actor, mission in itertools.product(actors, missions)
    ]
    with open(output_path, "wb") as f:
        pickle.dump(missions, f)


def _resolve_task(task, generator):
    if isinstance(task, types.CutIn):
        if isinstance(task.complete_on_edge_id, types.JunctionEdgeIDResolver):
            task = replace(
                task,
                complete_on_edge_id=task.complete_on_edge_id.to_edge(
                    generator.road_network
                ),
            )

    return task


def _resolve_vias(via: Tuple[types.Via], generator):
    vias = [*via]
    for i in range(len(vias)):
        v = vias[i]
        if isinstance(v.edge_id, types.JunctionEdgeIDResolver):
            vias[i] = replace(v, edge_id=v.edge_id.to_edge(generator.road_network))
    return tuple(vias)


def _validate_missions(missions):
    for mission in missions:
        _validate_entry_tactic(mission)


def _validate_entry_tactic(mission):
    if not mission.entry_tactic:
        return

    if isinstance(mission.entry_tactic, types.TrapEntryTactic):
        if not mission.entry_tactic.zone and not isinstance(
            mission.entry_tactic.zone, types.MapZone
        ):
            return

        z_edge, _, _ = mission.entry_tactic.zone.start
        if isinstance(mission, types.EndlessMission):
            edge, _, _ = mission.start
            assert (
                edge == z_edge
            ), f"Zone edge `{z_edge}` is not the same edge as `types.EndlessMission` start edge `{edge}`"

        elif isinstance(mission, (types.Mission, types.LapMission)):
            edge, _, _ = mission.route.begin
            assert (
                edge == z_edge
            ), f"Zone edge `{z_edge}` is not the same edge as `types.Mission` route begin edge `{edge}`"


def gen_traffic_histories(scenario: str, histories):
    output_path = os.path.join(scenario, "traffic_histories.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(histories, f)
