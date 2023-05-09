from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t
from smarts.core.colors import Colors

scenario_id = "ff239c9d-e4ff-4acc-bad5-bd55648c212e"
scenario_path = None  # example: Path("path/to/dataset") / scenario_id

traffic_histories = [
    t.TrafficHistoryDataset(
        name=f"argoverse_{scenario_id}",
        source_type="Argoverse",
        input_path=scenario_path,
    )
]

ego_mission = [t.EndlessMission(begin=("road-202833190-202832889-202833142", 2, 33.9))]

leader_id = "history-vehicle-46408$"
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
