from pathlib import Path

from smarts.sstudio import gen_missions, gen_traffic
from smarts.sstudio.types import (
    JunctionEdgeIDResolver,
    Route,
    Mission,
    Via,
)
from smarts.sstudio import types as t


scenario = str(Path(__file__).parent)

gen_missions(
    scenario=scenario,
    missions=[
        Mission(
            Route(begin=("gneE14", 1, 1160), end=("gneE20", 0, 10)),
            via=[
                Via(
                    JunctionEdgeIDResolver(
                        start_edge_id="gneE14",
                        start_lane_index=1,
                        end_edge_id="gneE20",
                        end_lane_index=0,
                    ),
                    0,
                    20,
                    20,
                    -1,
                )
            ],
        ),
    ],
)
