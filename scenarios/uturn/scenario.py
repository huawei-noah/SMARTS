from pathlib import Path

from smarts.sstudio import gen_missions
from smarts.sstudio.types import (
    Route,
    Mission,
    UTurn,
)

scenario = str(Path(__file__).parent)

gen_missions(
    scenario=scenario,
    missions=[
        Mission(
            route=Route(begin=('edge-west-WE', 0, 0), end=('edge-west-EW', 0, 'max')),
            task=UTurn(),
        ),
    ],
)