import os
from pathlib import Path
import pickle

from smarts.sstudio import gen_social_agent_missions
from smarts.sstudio import types as t
from smarts.sstudio.genscenario import gen_missions


gen_missions(
    scenario = str(Path(__file__).parent),
    missions=[
        t.EndlessMission(begin=("E35-3", 2, 90)),
    ],
)