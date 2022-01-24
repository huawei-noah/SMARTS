from pathlib import Path

from src.smarts.sstudio.genscenario import gen_scenario
from src.smarts.sstudio.types import Scenario

gen_scenario(
    Scenario(),
    output_dir=str(Path(__file__).parent),
    overwrite=True,
)
