from pathlib import Path

from smarts.sstudio.genscenario import gen_scenario
from smarts.sstudio.types import Scenario, MapSpec


DATASET_PATH = "uncompressed_scenario_training_20s_training_20s.tfrecord-00000-of-01000"
SCENARIO_ID = "4f30f060069bbeb9"
gen_scenario(
    Scenario(map_spec=MapSpec(source=f"{DATASET_PATH}#{SCENARIO_ID}", lanepoint_spacing=1.0)),
    output_dir=str(Path(__file__).parent),
    overwrite=True,
)
