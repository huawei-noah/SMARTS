from pathlib import Path

from smarts.sstudio import gen_missions
from smarts.sstudio.types import (
    Route,
    Mission,
    TrapEntryTactic,
)

scenario = str(Path(__file__).parent)

gen_missions(
    # Scenario is the path to directory containing the SMARTS scenario resources.
    scenario=scenario,
    # Mission specifies the path the bm agent will take
    missions=[
        Mission(
            # Route will go 
            route=Route(
                # Starting location of bm agent
                begin=("gneE14", 2, 'max'), 
                # End location of bm agent
                end=("gneE7_2", 2, 10)
                # Intermediary locations the bm agent must pass through

            ),
            # The time in simulation seconds before the bm agent vehicle will attempt to enter the simulation.
            start_time=0, #s
            # Specifies additional entry details.
            entry_tactic=TrapEntryTactic(
                # The amount of time to wait to take over a vehicle before just emitting a vehicle. 
                # Recommend to leave at 0 seconds.
                wait_to_hijack_limit_s=0,
                # Speed in m/s the bm vehicle should enter at. 
                # `default_entry_speed=None` defaults to the map speed
                default_entry_speed=None,
            ),
            # Do not specify task. Leave `None`.
            task=None,
        ),
    ],
)
