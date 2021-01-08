from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio.types import (
    Route,
    Mission,
    Scenario,
)

missions = [
    Mission(Route(begin=("gneE17", 0, 10), end=("gneE5", 1, 100))),
    Mission(Route(begin=("gneE22", 0, 10), end=("gneE5", 0, 100))),
    Mission(Route(begin=("gneE17", 0, 25), end=("gneE5", 0, 100))),
    Mission(Route(begin=("gneE22", 0, 25), end=("gneE5", 1, 100))),
]
gen_scenario(
    Scenario(ego_missions=missions), output_dir=Path(__file__).parent,
)
