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
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional


def build_scenario(
    clean: bool,
    scenario: str,
    seed: Optional[int] = None,
    log: Optional[Callable[[Any], None]] = None,
):
    """Build a scenario."""

    if clean:
        clean_scenario(scenario)

    scenario_root = Path(scenario)

    scenario_py = scenario_root / "scenario.py"
    if scenario_py.exists():
        _install_requirements(scenario_root, log)
        if seed is not None:
            with tempfile.NamedTemporaryFile("w", suffix=".py", dir=scenario_root) as c:
                with open(scenario_py, "r") as o:
                    c.write(
                        f"from smarts.core import seed as smarts_seed; smarts_seed({seed});\n"
                    )
                    c.write(o.read())

                c.flush()
                subprocess.check_call(
                    [sys.executable, Path(c.name).name], cwd=scenario_root
                )
        else:
            subprocess.check_call([sys.executable, "scenario.py"], cwd=scenario_root)


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


def _install_requirements(scenario_root, log: Optional[Callable[[Any], None]] = None):
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

        if log is not None:
            log(f"Installing scenario dependencies via '{' '.join(pip_install_cmd)}'")

        try:
            subprocess.check_call(pip_install_cmd, stdout=subprocess.DEVNULL)
        finally:
            pip_index_proc.terminate()
            pip_index_proc.wait()
