from pathlib import Path

from smarts.sstudio.genscenario import gen_scenario
from smarts.sstudio.sstypes import Scenario

gen_scenario(
    Scenario(),
    output_dir=str(Path(__file__).parent),
)
