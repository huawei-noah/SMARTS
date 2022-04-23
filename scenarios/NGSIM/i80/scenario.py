import math
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

traffic_histories = [
    t.TrafficHistoryDataset(
        name=f"i80_{hd}",
        source_type="NGSIM_highway",
        input_path=None,  # for example: f"./trajectories-{hd}.txt"
        x_margin_px=60,
        swap_xy=True,
        flip_y=True,
        map_max_y=25.02,
        map_width=310.92,
        speed_limit_mps=28,
        heading_inference_window=5,
        max_angular_velocity=4,
        default_heading=1.5 * math.pi,
    )
    for hd in ["0400-0415", "0500-0515", "0515-0530"]
]

gen_scenario(
    t.Scenario(traffic_histories=traffic_histories), output_dir=Path(__file__).parent
)
