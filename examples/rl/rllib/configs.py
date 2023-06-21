import argparse
import multiprocessing
from pathlib import Path


def gen_parser(prog: str, default_result_dir: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog)
    parser.add_argument(
        "scenarios",
        help="A list of scenarios. Each element can be either the scenario to"
        "run or a directory of scenarios to sample from. See `scenarios/`"
        "folder for some samples you can use.",
        type=str,
        nargs="*",
    )
    parser.add_argument(
        "--envision",
        action="store_true",
        help="Run simulation with Envision display.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=2000,
        help="The training batch size. This value must be > 0.",
    )
    parser.add_argument(
        "--time_total_s",
        type=int,
        default=1 * 60 * 60,  # 1 hour
        help="Total time in seconds to run the simulation for. This is a rough end time as it will be checked per training batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The base random seed to use, intended to be mixed with --num_samples",
    )
    parser.add_argument(
        "--num_agents", type=int, default=2, help="Number of agents (one per policy)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=(multiprocessing.cpu_count() // 2 + 1),
        help="Number of workers (defaults to use all system cores)",
    )
    parser.add_argument(
        "--resume_training",
        default=False,
        action="store_true",
        help="Resume an errored or 'ctrl+c' cancelled training. This does not extend a fully run original experiment.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=default_result_dir,
        help="Directory containing results",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="ERROR",
        help="Log level (DEBUG|INFO|WARN|ERROR)",
    )
    parser.add_argument(
        "--checkpoint_freq", type=int, default=3, help="Checkpoint frequency"
    )
    return parser
