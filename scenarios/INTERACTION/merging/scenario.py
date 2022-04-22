from pathlib import Path

from smarts.sstudio import types as t
from smarts.sstudio import gen_scenario


gen_scenario(
    t.Scenario(
        traffic_histories=["traffic_history_000.yml", "traffic_history_011.yml"],
    ),
    output_dir=Path(__file__).parent,
)
