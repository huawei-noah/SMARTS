from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio.types import (
    Mission,
    Route,
    Scenario,
)


mission = Mission(
    route=Route(begin=("edge-west-WE", 0, 40), end=("edge-east-WE", 2, 10))
)

gen_scenario(
    scenario=Scenario(ego_missions=[mission],), output_dir=Path(__file__).parent
)
