from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

dataset_path = None  # e.g. Path(__file__).parent.absolute() / "uncompressed_scenario_training_20s_training_20s.tfrecord-00000-of-01000"
scenario_id = None  # e.g. "4f30f060069bbeb9"

traffic_histories = [
    t.TrafficHistoryDataset(
        name=f"waymo",
        source_type="Waymo",
        input_path=dataset_path,
        scenario_id=scenario_id,
    )
]

gen_scenario(
    t.Scenario(
        map_spec=t.MapSpec(
            source=f"{dataset_path}#{scenario_id}", lanepoint_spacing=1.0
        ),
        traffic_histories=traffic_histories,
    ),
    output_dir=Path(__file__).parent,
)
