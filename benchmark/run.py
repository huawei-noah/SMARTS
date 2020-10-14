import argparse
import ray

from gym.spaces import Tuple
from pathlib import Path
from ray import tune

from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.scenario import Scenario

from benchmark.agents import load_config
from benchmark.agents.common import SimpleCallbacks


RUN_NAME = Path(__file__).stem
EXPERIMENT_NAME = "{scenario}-{n_agent}"


def main(args):
    if args.cluster:
        ray.init(address="auto", redis_password="5241590000000000")
        print(
            "--------------- Ray startup ------------\n{}".format(
                ray.state.cluster_resources()
            )
        )
    scenario_path = Path(args.scenario).absolute()
    agent_missions_count = Scenario.discover_agent_missions_count(scenario_path)
    if agent_missions_count == 0:
        agent_ids = ["default_policy"]
    else:
        agent_ids = [f"AGENT-{i}" for i in range(agent_missions_count)]

    config = load_config(args.config_file)
    agents = {
        agent_id: AgentSpec(
            **config["agent"], interface=AgentInterface(**config["interface"])
        )
        for agent_id in agent_ids
    }

    config["env_config"].update(
        {
            "seed": 42,
            "scenarios": [str(scenario_path)],
            "headless": args.headless,
            "agent_specs": agents,
        }
    )

    obs_space, act_space = config["policy"][1:3]
    tune_config = config["run"]["config"]

    if args.paradigm == "centralized":
        config["env_config"].update(
            {
                "obs_space": Tuple([obs_space] * agent_mission_num),
                "act_space": Tuple([act_space] * agent_mission_num),
                "groups": {"group": agent_ids},
            }
        )
        tune_config.update(config["policy"][-1])
    else:
        policies = {}

        for k in agents:
            policies[k] = config["policy"][:-1] + (
                {**config["policy"][-1], "agent_id": k},
            )

        tune_config.update(
            {
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": lambda agent_id: agent_id,
                },
            }
        )

    tune_config.update(
        {
            "env_config": config["env_config"],
            "callbacks": SimpleCallbacks,
            "num_workers": args.num_workers,
            "horizon": args.horizon,
        }
    )

    experiment_name = EXPERIMENT_NAME.format(
        scenario=scenario_path.stem, n_agent=len(agents),
    )

    log_dir = Path(args.log_dir).expanduser().absolute() / RUN_NAME
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.restore:
        restore_path = Path(args.restore).expanduser()
        print(f"Loading model from {restore_path}")
    else:
        restore_path = None

    # run experiments
    config["run"].update(
        {
            "run_or_experiment": config["trainer"],
            "name": experiment_name,
            "local_dir": str(log_dir),
            "restore": restore_path,
        }
    )
    analysis = tune.run(**config["run"])

    print(analysis.dataframe().head())


def parse_args():
    parser = argparse.ArgumentParser("Benchmarking learning")
    parser.add_argument(
        "--paradigm",
        type=str,
        default="decentralized",
        help="Algorithm paradigm, decentralized (default) or centralized",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="benchmark/scenarios/intersections/4lane",
        help="Scenario name",
    )
    parser.add_argument(
        "--headless", default=False, action="store_true", help="Turn on headless mode"
    )
    parser.add_argument(
        "--log_dir",
        default="./log/results",
        type=str,
        help="path to store rllib log and checkpoints, default is ./log/results",
    )
    parser.add_argument("--config_file", "-f", type=str)
    parser.add_argument("--restore", type=str)
    parser.add_argument("--address", type=str)
    parser.add_argument("--num_workers", type=int, default=1, help="rllib num workers")
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument(
        "--horizon", type=int, default=1000, help="horizon for a episode"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
