from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio.types import (
    Route,
    Mission,
    Scenario,
)

missions = [
    Mission(Route(begin=("-gneE1", 0, 5), end=("-gneE1", 0, 97))),
    Mission(Route(begin=("gneE1", 0, 5), end=("gneE1", 0, 97))),
    Mission(Route(begin=("-gneE1", 0, 20), end=("-gneE1", 0, 97))),
    Mission(Route(begin=("gneE1", 0, 20), end=("gneE1", 0, 97))),
]

gen_scenario(
    Scenario(ego_missions=missions), output_dir=Path(__file__).parent,
)
