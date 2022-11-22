from pathlib import Path

from smarts.sstudio.genscenario import gen_scenario
from smarts.sstudio.types import Scenario

gen_scenario(
    scenario=Scenario(),
    output_dir=Path(__file__).parent,
)
