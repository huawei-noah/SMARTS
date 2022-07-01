from pathlib import Path

from smarts.sstudio import types as t
from smarts.sstudio.genscenario import gen_scenario

ego_missions = [t.Mission(t.Route(begin=("1_0_R", 1, 5), end=("1_2_R", 1, "max")))]

gen_scenario(
    scenario=t.Scenario(
        ego_missions=ego_missions,
    ),
    output_dir=str(Path(__file__).parent),
    overwrite=True,
)
