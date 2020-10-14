from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio.types import (
    Route,
    Mission,
    Scenario,
)

missions = [
    Mission(Route(begin=("edge-south-SN", 1, 40), end=("edge-west-EW", 0, 60))),
    Mission(Route(begin=("edge-west-WE", 1, 50), end=("edge-east-WE", 0, 60))),
    Mission(Route(begin=("edge-north-NS", 0, 40), end=("edge-south-NS", 1, 40))),
    Mission(Route(begin=("edge-east-EW", 0, 50), end=("edge-west-EW", 1, 40))),
]

gen_scenario(
    Scenario(ego_missions=missions), output_dir=Path(__file__).parent,
)
