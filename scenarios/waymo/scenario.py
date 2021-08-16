from pathlib import Path

from smarts.sstudio.genscenario import gen_scenario
from smarts.sstudio.types import Scenario

gen_scenario(
    Scenario(traffic_histories=["waymo.yaml"]),
    output_dir=str(Path(__file__).parent),
    overwrite=True,
)
