from setuptools import setup

from marl_agent import VERSION

setup(
    name="marl-agent",
    description="marl agent example",
    version=f"0.{VERSION}",
    packages=["marl_agent"],
    include_package_data=True,
    install_requires=["tensorflow==1.15", "smarts"],
)
