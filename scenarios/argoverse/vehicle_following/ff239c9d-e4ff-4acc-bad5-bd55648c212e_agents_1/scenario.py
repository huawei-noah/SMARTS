from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t
from smarts.core.colors import Colors


PATH = "dataset"
scenario_id = "ff239c9d-e4ff-4acc-bad5-bd55648c212e"
scenario_path = Path(__file__).resolve().parents[3] / PATH / scenario_id

traffic_histories = [
    t.TrafficHistoryDataset(
        name=f"argoverse_{scenario_id}",
        source_type="Argoverse",
        input_path=scenario_path,
    )
]

duration = 11
ego_mission = [t.EndlessMission(begin=("road-202833190-202832889-202833142", 0, "max"))]

leader_id = "history-vehicle-46408$"
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
