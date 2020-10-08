import os
import random

import argparse
from smarts.sstudio import gen_traffic, gen_missions
from smarts.sstudio.types import (
    Traffic,
    Flow,
    Route,
    RandomRoute,
    TrafficActor,
    Mission,
)


def generate_scenario(scenario_root, n_sv, output_dir):

    name = "random_" + str(n_sv)
    traffic = Traffic(
        flows=[
            Flow(
                route=RandomRoute(),
                rate=1,
                begin=0,
                end=1 * 60 * 60,
                actors={TrafficActor(name="car"): 1.0},
            )
            for n in range(n_sv)
        ]
    )

    gen_traffic(scenario_root, traffic, name=name, output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("generate scenario")

    # env setting
    parser.add_argument("--scenario", type=str, default="loop", help="scenario name")
    parser.add_argument("--nv", type=int, default=10, help="social viecle number")
    parser.add_argument("--output_dir", "-o", type=str, default=None, help="output_dir")
    args = parser.parse_args()

    generate_scenario(os.path.abspath(args.scenario), args.nv, args.output_dir)
