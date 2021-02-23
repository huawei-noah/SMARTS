import argparse
import glob
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import yaml
from rich.console import Console

from smarts.sstudio.types import SocialAgentActor

console = Console()


def make_path(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def latest_dir_or_file(path):
    return max(glob.glob(str(path / "*")), key=os.path.getctime)


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def rm_rf(path):
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def python(command):
    try:
        subprocess.run(
            [sys.executable] + command,
            # shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except Exception as e:
        console.print(e.output.decode("utf-8"))
        exit(e.returncode)


def build_social_agent_actors(original_data):
    agent_names, actors = [], []
    agent_groups = {}
    for group_name, agents in original_data.items():
        agent_groups[group_name] = []
        for agent in agents:
            if "locator" in agent:
                # local file-based agent
                agent_locator = agent["locator"]
                if "name" in agent:
                    agent_name = agent["name"]
                else:
                    agent_name = agent_locator.split(":")[-1]

                actor = SocialAgentActor(
                    name=agent_name,
                    agent_locator=agent_locator,
                    policy_kwargs=agent["params"] if "params" in agent else {},
                )
            else:
                raise ValueError(
                    f"Agent {agent} configuration does not match the expected schema"
                )

            agent_names.append(agent_name)
            actors.append(actor)
            agent_groups[group_name].append(agent_name)

    return agent_names, actors, agent_groups


def build_scenario(agent_name, actor, scenario_path):
    # XXX: These dynamic scenarios are not a good design choice and have simply
    #      been ported from the previous bash-based design. We should adopt a
    #      design supporting better reproducibility and more structured
    #      interfaces.
    with tempfile.NamedTemporaryFile(mode="wb") as f:
        console.print(
            f"  [bold magenta]Agent:[/bold magenta] [bold]{agent_name}[/bold]"
        )
        console.print(
            f'    Building scenario="{scenario_path}" for agent="{agent_name}"'
        )
        pickle.dump(actor, f)
        f.flush()
        agent_path = Path(f.name).absolute()

        try:
            subprocess.run(
                [f"SOCIAL_AGENT_PATH={agent_path} scl scenario build {scenario_path}"],
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except OSError as e:
            console.print(e)
            exit(1)
        except Exception as e:
            console.print(e.output.decode("utf-8"))
            exit(e.returncode)


def evaluate_agents_on_scenario(
    agent_names,
    actors,
    scenario,
    scenarios_root_path,
    scenario_logs_path,
    result_path,
    agent_groups,
):
    console.print(f"[bold magenta]Scenario:[/bold magenta] [bold]{scenario}[/bold]")

    scenario_path = scenarios_root_path / scenario

    log_path = scenario_logs_path / scenario
    log_path.mkdir(parents=True, exist_ok=True)

    for agent_name, actor in zip(agent_names, actors):
        rm_rf(scenario_path / "social_agents")
        build_scenario(agent_name, actor, scenario_path)

        console.print("    Running environment and recording results")
        python(
            [
                str(Path(__file__).parent / "egoless_example.py"),
                scenario_logs_path,
                scenario_path,
            ]
        )

        console.print("    Extracting data")
        data_dir = latest_dir_or_file(log_path / "data_replay")
        step_num = batch_data["scenario_list"][scenario]["step_num"]
        python(
            [
                str(Path(__file__).parent / "metrics" / "data_extraction.py"),
                data_dir,
                log_path,
                agent_name,
                "--step-num",
                str(step_num),
            ]
        )

    console.print("  [bold]Visualizing scenario results[/bold]")
    python(
        [
            str(
                Path(__file__).parent
                / carried_out_path
                / "original_data_visualization.py"
            ),
            log_path,
            result_path,
            str(agent_names),
            str(agent_groups),
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="evaluation",
        description="Evaluation script for agent diversity.",
    )

    parser.add_argument(
        "batch_yaml",
        help="The path to the run batch yaml file.",
        type=str,
    )

    args = parser.parse_args()
    batch_data = load_yaml(args.batch_yaml)
    evaluation_items = batch_data["evaluation_items"].split(" ")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    result_path = make_path(Path(batch_data["result_path"]) / "results" / timestamp)
    scenarios_root_path = make_path(Path(batch_data["scenarios_root"]))
    scenario_logs_path = make_path(result_path / "scenarios_logs")
    other_result_path = make_path(result_path / "metrics_results")
    diversity_result_path = make_path(result_path / "metrics_results" / "diversity")
    carried_out_path = make_path(Path("metrics/diversity"))

    agent_names, actors, agent_groups = build_social_agent_actors(
        batch_data["agent_list"]
    )
    for scenario in batch_data["scenario_list"]:
        evaluate_agents_on_scenario(
            agent_names,
            actors,
            scenario,
            scenarios_root_path,
            scenario_logs_path,
            other_result_path,
            agent_groups,
        )

    console.print("[bold]evaluation results[/bold]")
    python(
        [
            str(Path(__file__).parent / "metrics" / "run_evaluation.py"),
            scenario_logs_path,
            other_result_path,
            diversity_result_path,
            str(evaluation_items),
            str(agent_names),
            str(agent_groups),
        ]
    )

    console.print("[bold]Done![/bold] metric results:")
    with open(result_path / "metrics_results" / "report.csv", "r") as report:
        console.print(report.read())

    console.print(f'Full results can be found at "{result_path}"')
