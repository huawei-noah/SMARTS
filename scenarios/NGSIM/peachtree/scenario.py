from pathlib import Path

from smarts.sstudio import types as t
from smarts.sstudio import gen_scenario


traffic_histories = [
    t.TrafficHistoryDataset(
        name=f"peach_{hd}",
        source_type="NGSIM_city",
        input_path=None,  # for example: f"./trajectories-{hd}pm.txt"
        speed_limit_mps=28,
        default_heading=0,
    )
    for hd in ["0400-0415", "1245-0100"]
]

gen_scenario(
    t.Scenario(traffic_histories=traffic_histories), output_dir=Path(__file__).parent
)
