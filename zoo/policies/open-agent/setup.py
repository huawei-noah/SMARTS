from setuptools import setup
from pathlib import Path

from open_agent.version import VERSION

try:
    from open_agent.policy import compile_solver

    compiled_extensions = compile_solver("./build")
except ImportError:
    compiled_extensions = []

try:
    # See: https://stackoverflow.com/a/45150383
    # We want to let setuptools know that this is an impure package (ie. contains platform specific code)
    # since we are including the compiled extensions from above.
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False


except ImportError:
    bdist_wheel = None

setup(
    name="open-agent",
    description="An Autonomous Vehicle Agent for SMARTS built using OpEn",
    version=VERSION,
    packages=["open_agent"],
    install_requires=["opengen==0.6.4", "smarts"],
    package_data={"open_agent": ["hyper_params.json"]},
    cmdclass={"bdist_wheel": bdist_wheel},
)
