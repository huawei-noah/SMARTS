from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

# scenario_path is a directory with the following structure:
# /path/to/dataset/{scenario_id}
# ├── log_map_archive_{scenario_id}.json
# └── scenario_{scenario_id}.parquet


scenario_id = "0a53dd99-2946-4b4d-ab66-c4d6fef97be2"  # e.g. "0000b6ab-e100-4f6b-aee8-b520b57c0530"
scenario_path = None  # e.g. Path("/home/user/argoverse/train/") / scenario_id

ego_mission = [
    t.Mission(
        t.Route(
            begin=("road-353638670-353638219-353638558", 1, 6.7),
            end=("road-353637909-353637861-353637941-353637727", 1, 4.2),
        ),
        entry_tactic=t.IdEntryTactic(start_time=0.1, actor_id="history-vehicle-5717"),
    )
]

traffic_histories = [
    t.TrafficHistoryDataset(
        name=f"argoverse_{scenario_id}",
        source_type="Argoverse",
        input_path=scenario_path,
    )
]

gen_scenario(
    t.Scenario(
        ego_missions=ego_mission,
        map_spec=t.MapSpec(source=f"{scenario_path}", lanepoint_spacing=1.0),
        traffic_histories=traffic_histories,
    ),
    output_dir=Path(__file__).parent,
)
