import argparse

from ultra.scenarios.generate_scenarios import build_scenarios


if __name__ == "__main__":
    parser = argparse.ArgumentParser("build-scenarios")
    parser.add_argument(
        "--task",
        help="The name of the task used to describe the scenarios.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--level",
        help="The level of the config from which the scenarios will be built.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--save-dir",
        help="Where to save the created scenarios.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--root-dir",
        help="The directory containing the task directories.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--pool-dir",
        help="The directory containing the map files.",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    build_scenarios(
        task=args.task,
        level_name=args.level,
        stopwatcher_behavior=None,
        stopwatcher_route=None,
        save_dir=args.save_dir,
        root_path=args.root_dir,
        pool_dir=args.pool_dir,
    )
