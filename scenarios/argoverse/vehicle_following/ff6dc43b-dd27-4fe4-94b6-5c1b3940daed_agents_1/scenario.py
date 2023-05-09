from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t
from smarts.core.colors import Colors


scenario_id = "ff6dc43b-dd27-4fe4-94b6-5c1b3940daed"
scenario_path = None  # example: Path("path/to/dataset") / scenario_id

traffic_histories = [
    t.TrafficHistoryDataset(
        name=f"argoverse_{scenario_id}",
        source_type="Argoverse",
        input_path=scenario_path,
    )
]

ego_mission = [t.EndlessMission(begin=("road-243772234-243772033", 1, 10.9))]

leader_id = "history-vehicle-40031$"
# runtime = 11
gen_scenario(
    t.Scenario(
        ego_missions=ego_mission,
        map_spec=t.MapSpec(source=f"{scenario_path}", lanepoint_spacing=1.0),
        traffic_histories=traffic_histories,
        scenario_metadata=t.ScenarioMetadata(leader_id, Colors.Blue),
    ),
    output_dir=Path(__file__).parent,
)
