from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

# scenario_path is a directory with the following structure:
# /path/to/dataset/{scenario_id}
# ├── log_map_archive_{scenario_id}.json
# └── scenario_{scenario_id}.parquet


scenario_id = "0a576bf1-66ae-495a-9c87-236f3fc2aa01"  # e.g. "0000b6ab-e100-4f6b-aee8-b520b57c0530"
scenario_path = None  # e.g. Path("/home/user/argoverse/train/") / scenario_id

ego_mission = [
    t.Mission(
        t.Route(
            begin=("road-431632924-431632618", 0, 0),
            end=("road-431673448-431673312-431673462", 1, 17.0),
        )
    )
]

traffic_histories = [
    t.TrafficHistoryDataset(
        name=f"argoverse_{scenario_id}",
        source_type="Argoverse",
        input_path=scenario_path,
    )
]
duration = 11
gen_scenario(
    t.Scenario(
        ego_missions=ego_mission,
        map_spec=t.MapSpec(source=f"{scenario_path}", lanepoint_spacing=1.0),
        traffic_histories=traffic_histories,
        scenario_metadata=t.ScenarioMetadata(
            scenario_difficulty=0.3,
            scenario_duration=duration,
        ),
    ),
    output_dir=Path(__file__).parent,
)
