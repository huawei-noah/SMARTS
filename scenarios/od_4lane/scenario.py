from pathlib import Path
import sys
sys.path.append('./src')
from smarts.sstudio.genscenario import gen_scenario
from smarts.sstudio.types import Scenario

gen_scenario(
    Scenario(),
    output_dir=str(Path(__file__).parent),
    overwrite=True,
)
