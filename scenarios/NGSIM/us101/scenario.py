from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

traffic_histories = [
    t.TrafficHistoryDataset(
        name=f"us101_{hd}",
        source_type="NGSIM",
        input_path=None,  # for example: f"./trajectories-{hd}.txt"
        speed_limit_mps=28,
        default_heading=0,
    )
    for hd in ["0750am-0805am", "0805am-0820am", "0820am-0835am"]
]

gen_scenario(
    t.Scenario(traffic_histories=traffic_histories), output_dir=Path(__file__).parent
)
