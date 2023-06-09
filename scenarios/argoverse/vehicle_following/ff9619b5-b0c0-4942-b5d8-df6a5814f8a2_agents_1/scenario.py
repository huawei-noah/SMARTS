from pathlib import Path

from smarts.core.colors import Colors
from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

scenario_id = "ff9619b5-b0c0-4942-b5d8-df6a5814f8a2"  # e.g. "0000b6ab-e100-4f6b-aee8-b520b57c0530"
scenario_path = None

traffic_histories = [
    t.TrafficHistoryDataset(
        name=f"argoverse_{scenario_id}",
        source_type="Argoverse",
        input_path=scenario_path,
    )
]

duration = 11
ego_mission = [
    t.EndlessMission(begin=("road-358009253-358009468", 1, 0.7), start_time=1)
]

leader_id = "history-vehicle-15518$"
# runtime = 11
gen_scenario(
    t.Scenario(
        ego_missions=ego_mission,
        map_spec=t.MapSpec(source=f"{scenario_path}", lanepoint_spacing=1.0),
        traffic_histories=traffic_histories,
        scenario_metadata=t.ScenarioMetadata(
            actor_of_interest_re_filter=leader_id,
            actor_of_interest_color=Colors.Blue,
            scenario_difficulty=0.3,
            scenario_duration=duration,
        ),
    ),
    output_dir=Path(__file__).parent,
)
