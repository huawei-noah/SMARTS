from pathlib import Path

from smarts.sstudio import types as t
from smarts.sstudio import gen_scenario


gen_scenario(
    t.Scenario(
        traffic_histories=[
            "i80_0400-0415.yaml",
            "i80_0500-0515.yaml",
            "i80_0515-0530.yaml",
        ],
    ),
    output_dir=Path(__file__).parent,
)
