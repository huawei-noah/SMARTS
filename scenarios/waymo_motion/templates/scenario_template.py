from pathlib import Path
import yaml
import os

from smarts.sstudio.genscenario import gen_scenario
from smarts.sstudio.types import Scenario, MapSpec

yaml_file = os.path.join(Path(__file__).parent, "waymo.yaml")
with open(yaml_file, "r") as yf:
    dataset_spec = yaml.safe_load(yf)["trajectory_dataset"]

dataset_path = dataset_spec["input_path"]
scenario_id = dataset_spec["scenario_id"]

gen_scenario(
    Scenario(
        map_spec=MapSpec(source=f"{dataset_path}#{scenario_id}", lanepoint_spacing=1.0),
        traffic_histories=["waymo.yaml"],
    ),
    output_dir=str(Path(__file__).parent),
    overwrite=True,
)
