import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Optional


def build_single_scenario(
    clean: bool,
    allow_offset_map: bool,
    scenario: str,
    log: Optional[Callable[[Any], None]] = None,
):
    """Build a scenario."""

    if clean:
        clean_scenario(scenario)

    scenario_root = Path(scenario)
    scenario_root_str = str(scenario_root)

    scenario_py = scenario_root / "scenario.py"
    if scenario_py.exists():
        _install_requirements(scenario_root)
        subprocess.check_call([sys.executable, "scenario.py"], cwd=scenario_root)

    from smarts.core.scenario import Scenario

    traffic_histories = Scenario.discover_traffic_histories(scenario_root_str)
    # don't shift maps for scenarios with traffic histories since history data must line up with map
    shift_to_origin = not allow_offset_map and not bool(traffic_histories)

    map_spec = Scenario.discover_map(scenario_root_str, shift_to_origin=shift_to_origin)
    road_map, _ = map_spec.builder_fn(map_spec)
    if not road_map:
        log(
            "No reference to a RoadNetwork file was found in {}, or one could not be created. "
            "Please make sure the path passed is a valid Scenario with RoadNetwork file required "
            "(or a way to create one) for scenario building.".format(scenario_root_str)
        )
        return

    road_map.to_glb(os.path.join(scenario_root, "map.glb"))


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
    ]
    p = Path(scenario)
    for file_name in to_be_removed:
        for f in p.glob(file_name):
            # Remove file
            f.unlink()


def _install_requirements(scenario_root, log: Optional[Callable[[Any], None]] = None):
    import importlib.resources as pkg_resources

    requirements_txt = scenario_root / "requirements.txt"
    if requirements_txt.exists():
        import zoo.policies

        with pkg_resources.path(zoo.policies, "") as path:
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
