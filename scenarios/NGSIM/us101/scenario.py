from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t


traffic_histories = [
    t.TrafficHistoryDataset(
        name=f"us101_{hd}",
        source_type="NGSIM_highway",
        input_path=None,  # for example: f"./trajectories-{hd}am.txt"
        map_width=641.63,
        speed_limit_mps=28,
        default_heading=0,
    )
    for hd in ["0750-0805", "0805-0820", "0820-0835"]
]

gen_scenario(
    t.Scenario(traffic_histories=traffic_histories), output_dir=Path(__file__).parent
)
