from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

# scenario_path is a directory with the following structure:
# /path/to/dataset/{scenario_id}
# ├── log_map_archive_{scenario_id}.json
# └── scenario_{scenario_id}.parquet

scenario_id = "0bf054e3-7698-4b86-9c98-626df2dee9f4"  # e.g. "0000b6ab-e100-4f6b-aee8-b520b57c0530"
scenario_path = None  # e.g. Path("/home/user/argoverse/train/") / scenario_id

traffic_histories = [
    t.TrafficHistoryDataset(
        name=f"argoverse_{scenario_id}",
        source_type="Argoverse",
        input_path=scenario_path,
    )
]
duration = 11
ego_mission = [
    t.Mission(
        route=t.Route(
            begin=("road-331937899", 0, 10.1),
            end=("road-331938057-331937870-331937776", 1, 13.4),
        )
    )
]

gen_scenario(
    t.Scenario(
        ego_missions=ego_mission,
        map_spec=t.MapSpec(source=f"{scenario_path}", lanepoint_spacing=1.0),
        traffic_histories=traffic_histories,
        scenario_metadata=t.ScenarioMetadata(
            scenario_difficulty=0.6,
            scenario_duration=duration,
        ),
    ),
    output_dir=Path(__file__).parent,
)
