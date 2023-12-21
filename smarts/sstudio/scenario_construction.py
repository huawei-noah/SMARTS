# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import logging
import multiprocessing
import os
import shutil
import subprocess
import sys
from multiprocessing import Process, Semaphore, synchronize
from pathlib import Path
from typing import Any, Callable, List

logger = logging.getLogger(__name__)
LOG_DEFAULT = logger.info


def build_scenario(
    scenario: str,
    clean: bool = False,
    seed: int = 42,
    log: Callable[[Any], None] = LOG_DEFAULT,
):
    """Build a scenario."""

    log(f"Building: {scenario}")

    if clean:
        clean_scenario(scenario)

    scenario_root = Path(scenario)

    scenario_py = scenario_root / "scenario.py"
    if scenario_py.exists():
        _install_requirements(scenario_root, log)

        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "scenario_builder.py",
                    str(scenario_py.absolute()),
                    str(seed),
                ],
                cwd=Path(__file__).parent,
            )
        except subprocess.CalledProcessError as e:
            raise SystemExit(e)


def build_scenarios(
    scenarios: List[str],
    clean: bool = False,
    seed: int = 42,
    log: Callable[[Any], None] = LOG_DEFAULT,
):
    """Build a list of scenarios."""

    if not scenarios:
        # nargs=-1 in combination with a default value is not supported
        # if scenarios is not given, set /scenarios as default
        scenarios = ["scenarios"]

    concurrency = max(1, multiprocessing.cpu_count() - 1)
    sema = Semaphore(concurrency)
    all_processes = []
    for scenarios_path in scenarios:
        for subdir, _, _ in os.walk(scenarios_path):
            if _is_scenario_folder_to_build(subdir):
                p = Path(subdir)
                scenario = f"{scenarios_path}/{p.relative_to(scenarios_path)}"
                proc = Process(
                    target=_build_scenario_proc,
                    kwargs={
                        "scenario": scenario,
                        "semaphore": sema,
                        "clean": clean,
                        "seed": seed,
                        "log": log,
                    },
                )
                all_processes.append(proc)
                proc.start()

    for proc in all_processes:
        proc.join()


def _build_scenario_proc(
    scenario: str,
    semaphore: synchronize.Semaphore,
    clean: bool,
    seed: int,
    log: Callable[[Any], None] = LOG_DEFAULT,
):

    semaphore.acquire()
    try:
        build_scenario(scenario=scenario, clean=clean, seed=seed, log=log)
    finally:
        semaphore.release()


def _is_scenario_folder_to_build(path: str) -> bool:
    if os.path.exists(os.path.join(path, "waymo.yaml")):
        # for now, don't try to build Waymo scenarios...
        return False
    if os.path.exists(os.path.join(path, "scenario.py")):
        return True
    from smarts.sstudio.sstypes import MapSpec

    map_spec = MapSpec(path)
    road_map, _ = map_spec.builder_fn(map_spec)
    return road_map is not None


def clean_scenario(scenario: str):
    """Remove all cached scenario files in the given scenario directory."""

    to_be_removed = [
        "map.glb",
        "map_spec.pkl",
        "bubbles.pkl",
        "missions.pkl",
        "flamegraph-perf.log",
        "flamegraph.svg",
        "flamegraph.html",
        "*.rou.xml",
        "*.rou.alt.xml",
        "social_agents/*",
        "traffic/*.rou.xml",
        "traffic/*.smarts.xml",
        "history_mission.pkl",
        "*.shf",
        "*-AUTOGEN.net.xml",
        "build.db",
    ]
    p = Path(scenario)

    shutil.rmtree(p / "build", ignore_errors=True)
    shutil.rmtree(p / "traffic", ignore_errors=True)
    shutil.rmtree(p / "social_agents", ignore_errors=True)

    for file_name in to_be_removed:
        for f in p.glob(file_name):
            # Remove file
            f.unlink()


def _install_requirements(scenario_root, log: Callable[[Any], None] = LOG_DEFAULT):
    import os

    requirements_txt = scenario_root / "requirements.txt"
    if requirements_txt.exists():
        import zoo.policies

        path = Path(os.path.dirname(zoo.policies.__file__)).absolute()
        # Serve policies through the static file server, then kill after
        # we've installed scenario requirements
        pip_index_proc = subprocess.Popen(
            ["twistd", "-n", "web", "--path", path],
            # Hide output to keep display simple
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

        pip_install_cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(requirements_txt),
        ]

        log(f"Installing scenario dependencies via '{' '.join(pip_install_cmd)}'")

        try:
            subprocess.check_call(pip_install_cmd, stdout=subprocess.DEVNULL)
        finally:
            pip_index_proc.terminate()
            pip_index_proc.wait()
