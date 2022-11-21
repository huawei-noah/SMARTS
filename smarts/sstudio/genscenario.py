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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
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
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cloudpickle

from smarts.core.utils.file import pickle_hash
from smarts.core.utils.logging import timeit

from . import types
from .generators import TrafficGenerator

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__file__)
logger.setLevel(logging.WARNING)


def _build_graph(scenario: types.Scenario, base_dir: str) -> Dict[str, Any]:
    graph = collections.defaultdict(list)

    if scenario.map_spec:
        graph["map_spec"] = [os.path.join(base_dir, "map_spec.pkl")]

    if scenario.traffic:
        for name, traffic in scenario.traffic.items():
            ext = "smarts" if traffic.engine == "SMARTS" else "rou"
            artifact_path = os.path.join(base_dir, "traffic", f"{name}.{ext}.xml")
            graph["traffic"].append(artifact_path)

    if scenario.ego_missions:
        graph["ego_missions"] = [os.path.join(base_dir, "missions.pkl")]

    if scenario.social_agent_missions:
        for name in scenario.social_agent_missions.keys():
            artifact_path = os.path.join(base_dir, "social_agents", name)
            graph["social_agent_missions"].append(artifact_path)

    if scenario.bubbles:
        graph["bubbles"] = [os.path.join(base_dir, "bubbles.pkl")]

    if scenario.friction_maps:
        graph["friction_maps"] = [os.path.join(base_dir, "friction_map.pkl")]

    if scenario.traffic_histories:
        for dataset in scenario.traffic_histories:
            artifact_path = os.path.join(base_dir, f"{dataset.name}.shf")
            graph["traffic_histories"].append(artifact_path)

    return graph


def _needs_build(
    db_conn: sqlite3.Connection,
    scenario_obj: Any,
    artifact_paths: List[str],
    obj_hash: str,
) -> bool:
    if scenario_obj is None:
        return False  # There's no object in the DSL, so nothing to build
    if not all([os.path.exists(f) for f in artifact_paths]):
        return True  # Some of the expected output files don't exist

    query = """SELECT scenario_obj_hash FROM Artifacts WHERE artifact_path=?;"""
    for artifact_path in artifact_paths:
        cur = db_conn.cursor()
        cur.execute(query, (artifact_path,))
        row = cur.fetchone()
        cur.close()

        if row is None:
            return True  # Artifact is not in the db

        artifact_hash = row[0]
        if artifact_hash != obj_hash:
            return True  # Hash does not match, out of date
    return False


def _update_artifacts(
    db_conn: sqlite3.Connection, artifact_paths: List[str], hash_val: str
):
    query = """REPLACE INTO Artifacts (artifact_path, scenario_obj_hash)
               VALUES(?, ?);"""
    cur = db_conn.cursor()
    for artifact_path in artifact_paths:
        cur.execute(query, (artifact_path, hash_val))
    db_conn.commit()
    cur.close()


def gen_scenario(
    scenario: types.Scenario,
    output_dir: Union[str, Path],
    seed: int = 42,
    overwrite: bool = False,
):
    """This is now the preferred way to generate a scenario. Instead of calling the
    gen_* methods directly, we provide this higher-level abstraction that takes care
    of the sub-calls.
    """
    # XXX: For now this simply coalesces the sub-calls but in the future this allows
    #      us to simplify our serialization between SStudio and SMARTS.

    scenario_dir = os.path.abspath(str(output_dir))
    build_dir = os.path.join(scenario_dir, "build")
    build_graph = _build_graph(scenario, build_dir)
    os.makedirs(build_dir, exist_ok=True)

    # Create DB for build caching
    db_path = os.path.join(build_dir, "build.db")
    db_conn = sqlite3.connect(db_path)

    cur = db_conn.cursor()
    cur.execute(
        """CREATE TABLE IF NOT EXISTS Artifacts (
                artifact_path     TEXT PRIMARY KEY,
                scenario_obj_hash TEXT
        ) WITHOUT ROWID"""
    )
    db_conn.commit()
    cur.close()

    with timeit("gen_map", logger.info):
        artifact_paths = build_graph["map_spec"]
        obj_hash = pickle_hash(scenario.map_spec, True)
        if _needs_build(db_conn, scenario.map_spec, artifact_paths, obj_hash):
            gen_map(scenario_dir, scenario.map_spec)
            _update_artifacts(db_conn, artifact_paths, obj_hash)
            map_spec = scenario.map_spec
        else:
            map_spec = types.MapSpec(source=scenario_dir)

    with timeit("traffic", logger.info):
        artifact_paths = build_graph["traffic"]
        obj_hash = pickle_hash(scenario.traffic, True)
        if _needs_build(db_conn, scenario.traffic, artifact_paths, obj_hash):
            for name, traffic in scenario.traffic.items():
                gen_traffic(
                    scenario=scenario_dir,
                    traffic=traffic,
                    name=name,
                    seed=seed,
                    overwrite=overwrite,
                    map_spec=map_spec,
                )
            _update_artifacts(db_conn, artifact_paths, obj_hash)

    with timeit("ego_missions", logger.info):
        artifact_paths = build_graph["ego_missions"]
        obj_hash = pickle_hash(scenario.ego_missions, True)
        if _needs_build(db_conn, scenario.ego_missions, artifact_paths, obj_hash):
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
                        overwrite=overwrite,
                        map_spec=map_spec,
                    )
                else:
                    missions.append(mission)

            if missions:
                gen_missions(
                    scenario=output_dir,
                    missions=missions,
                    seed=seed,
                    overwrite=overwrite,
                    map_spec=map_spec,
                )

            _update_artifacts(db_conn, artifact_paths, obj_hash)

    with timeit("social_agent_missions", logger.info):
        artifact_paths = build_graph["social_agent_missions"]
        obj_hash = pickle_hash(scenario.social_agent_missions, True)
        if _needs_build(
            db_conn, scenario.social_agent_missions, artifact_paths, obj_hash
        ):
            for name, (actors, missions) in scenario.social_agent_missions.items():
                if not (
                    isinstance(actors, collections.abc.Sequence)
                    and isinstance(missions, collections.abc.Sequence)
                ):
                    raise ValueError("Actors and missions must be sequences")

                gen_social_agent_missions(
                    name=name,
                    scenario=output_dir,
                    social_agent_actor=actors,
                    missions=missions,
                    seed=seed,
                    map_spec=map_spec,
                )

            _update_artifacts(db_conn, artifact_paths, obj_hash)

    with timeit("bubbles", logger.info):
        artifact_paths = build_graph["bubbles"]
        obj_hash = pickle_hash(scenario.bubbles, True)
        if _needs_build(db_conn, scenario.bubbles, artifact_paths, obj_hash):
            gen_bubbles(scenario=output_dir, bubbles=scenario.bubbles)
            _update_artifacts(db_conn, artifact_paths, obj_hash)

    with timeit("friction_maps", logger.info):
        artifact_paths = build_graph["friction_maps"]
        obj_hash = pickle_hash(scenario.friction_maps, True)
        if _needs_build(db_conn, scenario.friction_maps, artifact_paths, obj_hash):
            gen_friction_map(
                scenario=output_dir, surface_patches=scenario.friction_maps
            )
            _update_artifacts(db_conn, artifact_paths, obj_hash)

    with timeit("traffic_histories", logger.info):
        artifact_paths = build_graph["traffic_histories"]
        obj_hash = pickle_hash(scenario.traffic_histories, True)
        if _needs_build(db_conn, scenario.traffic_histories, artifact_paths, obj_hash):
            gen_traffic_histories(
                scenario=output_dir,
                histories_datasets=scenario.traffic_histories,
                overwrite=overwrite,
                map_spec=map_spec,
            )
            _update_artifacts(db_conn, artifact_paths, obj_hash)


def gen_map(scenario: str, map_spec: types.MapSpec, output_dir: Optional[str] = None):
    """Saves a map spec to file."""
    build_dir = os.path.join(scenario, "build")
    output_dir = os.path.join(output_dir or build_dir, "map")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "map_spec.pkl")
    with open(output_path, "wb") as f:
        # we use cloudpickle here instead of pickle because the
        # map_spec object may contain a reference to a map_builder callable
        cloudpickle.dump(map_spec, f)


def gen_traffic(
    scenario: str,
    traffic: types.Traffic,
    name: str,
    output_dir: Optional[str] = None,
    seed: int = 42,
    overwrite: bool = False,
    map_spec: Optional[types.MapSpec] = None,
):
    """Generates the traffic routes for the given scenario. If the output directory is
    not provided, the scenario directory is used."""
    assert name != "missions", "The name 'missions' is reserved for missions!"

    build_dir = os.path.join(scenario, "build")
    output_dir = os.path.join(output_dir or build_dir, "traffic")
    os.makedirs(output_dir, exist_ok=True)

    generator = TrafficGenerator(scenario, map_spec, overwrite=overwrite)
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
    map_spec: Optional[types.MapSpec] = None,
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
            The random seed to use when generating behavior
        overwrite:
            If to forcefully write over the previous existing output file
        map_spec:
            An optional map specification that takes precedence over scenario directory information.
    """

    # For backwards compatibility we support both a single value and a sequence
    actors = social_agent_actor
    if not isinstance(actors, collections.abc.Sequence):
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

    output_dir = os.path.join(scenario, "build", "social_agents")
    saved = _gen_missions(
        scenario=scenario,
        missions=missions,
        actors=actors,
        name=name,
        output_dir=output_dir,
        seed=seed,
        overwrite=overwrite,
        map_spec=map_spec,
    )

    if saved:
        logger.debug(f"Generated social agent missions for scenario={scenario}")


def gen_missions(
    scenario: str,
    missions: Sequence,
    seed: int = 42,
    overwrite: bool = False,
    map_spec: Optional[types.MapSpec] = None,
):
    """Generates a route file to represent missions (a route per mission). Will create
    the output_dir if it doesn't exist already. The output file will be named `missions`.

    Args:
        scenario:
            The scenario directory
        missions:
            A sequence of missions for social agents to perform
        seed:
            The random seed to use when generating behavior
        overwrite:
            If to forcefully write over the previous existing output file
        map_spec:
            An optional map specification that takes precedence over scenario directory information.
    """

    output_dir = os.path.join(scenario, "build")
    saved = _gen_missions(
        scenario=scenario,
        missions=missions,
        actors=[types.TrafficActor(name="car")],
        name="missions",
        output_dir=output_dir,
        seed=seed,
        overwrite=overwrite,
        map_spec=map_spec,
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
    map_spec: Optional[types.MapSpec] = None,
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

    start_road_id, start_lane, start_offset = begin
    end_road_id, end_lane, end_offset = end

    missions = []
    for i in range(vehicle_count):
        s_lane = (start_lane + i) % used_lanes
        missions.append(
            types.LapMission(
                types.Route(
                    begin=(
                        start_road_id,
                        s_lane,
                        start_offset - grid_offset * i,
                    ),
                    end=(end_road_id, (end_lane + i) % used_lanes, end_offset),
                ),
                num_laps=num_laps,
                # route_length=route_length,
            )
        )

    saved = gen_missions(
        scenario=scenario,
        missions=missions,
        seed=seed,
        overwrite=overwrite,
        map_spec=map_spec,
    )

    if saved:
        logger.debug(f"Generated grouped lap missions for scenario={scenario}")


def gen_bubbles(scenario: str, bubbles: Sequence[types.Bubble]):
    """Generates 'bubbles' in the scenario that capture vehicles for actors.
    Args:
        scenario:
            The scenario directory
        bubbles:
            The bubbles to add to the scenario.
    """
    output_path = os.path.join(scenario, "build", "bubbles.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(bubbles, f)


def gen_friction_map(scenario: str, surface_patches: Sequence[types.RoadSurfacePatch]):
    """Generates friction map file according to the surface patches defined in
    scenario file.
    """
    output_path = os.path.join(scenario, "build", "friction_map.pkl")
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
    map_spec: Optional[types.MapSpec] = None,
):
    """Generates a route file to represent missions (a route per mission). Will
    create the output_dir if it doesn't exist already.
    """

    generator = TrafficGenerator(scenario, map_spec)

    def resolve_mission(mission):
        route = getattr(mission, "route", None)
        kwargs = {}
        if route:
            kwargs["route"] = generator.resolve_route(route, False)

        via = getattr(mission, "via", ())
        if via != ():
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

    return True


def _resolve_vias(via: Tuple[types.Via], generator):
    vias = [*via]
    for i in range(len(vias)):
        v = vias[i]
        if isinstance(v.road_id, types.JunctionEdgeIDResolver):
            vias[i] = replace(v, road_id=v.road_id.to_edge(generator.road_network))
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


def gen_traffic_histories(
    scenario: str,
    histories_datasets: Sequence[Union[types.TrafficHistoryDataset, str]],
    overwrite: bool,
    map_spec: Optional[types.MapSpec] = None,
):
    """Converts traffic history to a format that SMARTS can use.
    Args:
        scenario:
            The scenario directory
        histories_datasets:
            A sequence of traffic history descriptors.
        overwrite:
            If to forcefully write over the previous existing output file
        map_spec:
             An optional map specification that takes precedence over scenario directory information.
    """
    road_map = None  # shared across all history_datasets in scenario
    for hdsr in histories_datasets:
        assert isinstance(hdsr, types.TrafficHistoryDataset)
        if not hdsr.input_path:
            print(f"skipping placeholder dataset spec '{hdsr.name}'.")
            continue

        from smarts.sstudio import genhistories

        map_bbox = None
        if hdsr.filter_off_map or hdsr.flip_y:
            if map_spec:
                if not road_map:
                    road_map, _ = map_spec.builder_fn(map_spec)
                assert road_map
                map_bbox = road_map.bounding_box
            else:
                logger.warn(
                    f"no map_spec supplied, so unable to filter off-map coordinates and/or flip_y for {hdsr.name}"
                )
        output_dir = os.path.join(scenario, "build")
        genhistories.import_dataset(hdsr, output_dir, overwrite, map_bbox)
