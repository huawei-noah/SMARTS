from pathlib import Path

from smarts.sstudio.genscenario import gen_scenario
from smarts.sstudio.types import (
    EndlessMission,
    Flow,
    JunctionEdgeIDResolver,
    Mission,
    RandomRoute,
    Route,
    Scenario,
    Traffic,
    TrafficActor,
    Via,
)

gen_scenario(
    Scenario(traffic_histories=["training_20s.yaml"]),
    output_dir=str(Path(__file__).parent),
    overwrite=True
)

# scenario = Scenario(
#     ego_missions=[
#         Mission(
#             route=Route(begin=("-gneE4", 0, 10),
#             end=("-gneE5", 0, "max"))
#         )
#     ]
# )

# gen_scenario(
#     scenario=scenario,
#     output_dir=Path(__file__).parent,
#     overwrite=True
# )
