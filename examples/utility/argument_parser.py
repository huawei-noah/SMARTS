import argparse
from typing import Optional


def default_argument_parser(program: Optional[str] = None):
    """This factory method returns a vanilla `argparse.ArgumentParser` with the
    minimum subset of arguments that should be supported.

    You can extend it with more `parser.add_argument(...)` calls or obtain the
    arguments via `parser.parse_args()`.
    """
    if not program:
        from pathlib import Path

        program = Path(__file__).stem

    parser = argparse.ArgumentParser(program)
    parser.add_argument(
        "scenarios",
        help="A list of scenarios. Each element can be either the scenario to"
        "run or a directory of scenarios to sample from. See `scenarios/`"
        "folder for some samples you can use.",
        type=str,
        nargs="*",
    )
    parser.add_argument(
        "--episodes",
        help="The number of episodes to run the simulation for.",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--headless", help="Run the simulation in headless mode.", action="store_true"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sim-name",
        help="Simulation name.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--sumo-port", help="Run SUMO with a specified port.", type=int, default=None
    )
    return parser
