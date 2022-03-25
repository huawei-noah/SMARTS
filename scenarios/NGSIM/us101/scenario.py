from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

gen_scenario(
    t.Scenario(
        traffic_histories=[
            "us101_0750-0805.yml",
            "us101_0805-0820.yml",
            "us101_0820-0835.yml",
        ],
    ),
    output_dir=Path(__file__).parent,
)
