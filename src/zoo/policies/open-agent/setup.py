from pathlib import Path

from open_agent.version import VERSION
from setuptools import setup

try:
    import glob
    import shutil
    import subprocess

    from open_agent.agent import compile_solver

    compile_solver("./build")

    python_bindings_src = (
        Path(".") / "build" / "open_agent_solver" / "python_bindings_open_agent_solver"
    ).absolute()
    assert python_bindings_src.is_dir()
    subprocess.call(
        [
            # "sudo",
            "docker",
            "run",
            "--rm",
            "-v",
            f"{str(python_bindings_src.parent)}:/io",
            "--entrypoint",
            "/bin/bash",
            "konstin2/maturin",
            "-c",
            "cd /io/python_bindings_open_agent_solver && maturin build --release -i python3.7",
        ],
        cwd=str(python_bindings_src.parent),
        # ["maturin", "build", "--release", "-i", "python3.7"],
        # cwd=str(python_bindings_src),
    )

    generated_wheels = python_bindings_src / "target" / "wheels"
    assert generated_wheels.is_dir()

    generated_bindings_package = (
        Path(".").absolute().parent / "python-bindings-open-agent-solver"
    )
    generated_bindings_package.mkdir(exist_ok=True)
    for whl in glob.glob(str((generated_wheels / "*.whl"))):
        shutil.copy(whl, generated_bindings_package)

except ImportError:
    print(
        "WARNING: missing depencencies caused us to fail to compile the solver, rerun once the dependencies have installed"
    )

setup(
    name="open-agent",
    description="An Autonomous Vehicle Agent for SMARTS built using OpEn",
    version=VERSION,
    packages=["open_agent"],
    install_requires=[
        "opengen==0.6.4",
        "smarts",
        "wheel",
        "maturin",
        "python-bindings-open-agent-solver",
    ],
    package_data={"open_agent": ["config.json"]},
)
