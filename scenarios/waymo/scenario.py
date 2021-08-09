from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

gen_scenario(
    t.Scenario(traffic_histories=["training_20s.yaml"]),
    output_dir=str(Path(__file__).parent),
    overwrite=True
)
