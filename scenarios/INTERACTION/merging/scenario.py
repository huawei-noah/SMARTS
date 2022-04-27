import math
from pathlib import Path

from smarts.sstudio import types as t
from smarts.sstudio import gen_scenario


traffic_histories = [
    t.TrafficHistoryDataset(
        name=f"traffic_history_{hd}",
        source_type="INTERACTION",
        input_path=None,  # for example: f"./traffic_history_{hd}.txt"
        speed_limit_mps=15,
        default_heading=0.5 * math.pi,
    )
    for hd in ["000", "011"]
]

gen_scenario(
    t.Scenario(traffic_histories=traffic_histories), output_dir=Path(__file__).parent
)
