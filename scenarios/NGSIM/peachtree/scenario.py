from pathlib import Path

from smarts.sstudio import types as t
from smarts.sstudio import gen_scenario


gen_scenario(
    t.Scenario(
        traffic_histories=["peach_0400-0415.yml", "peach_1245-0100.yml"],
    ),
    output_dir=Path(__file__).parent,
)
