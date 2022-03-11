from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

gen_scenario(
    t.Scenario(
        traffic_histories=[
            "i80_0400-0415.yml",
            "i80_0500-0515.yml",
            "i80_0515-0530.yml",
        ],
    ),
    output_dir=Path(__file__).parent,
)
