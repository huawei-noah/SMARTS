import subprocess
from typing import List


class RayException(Exception):
    """An exception raised if ray package is required but not available."""

    @classmethod
    def required_to(cls, thing):
        return cls(
            f"""Ray Package is required to simulate {thing}.
               You may not have installed the [train] or [test] dependencies required to run the ray dependent example.
               Install them first using the command `pip install -e .[train, test]` at the source directory to install the package ray[rllib]==1.0.1.post1"""
        )


def build_scenario(scenario: List[str]):
    """Build the given scenarios.

    Args:
        scenario (List[str]): Scenarios to build.
    """
    build_scenario = " ".join(["scl scenario build-all"] + scenario)
    subprocess.call(build_scenario, shell=True)
