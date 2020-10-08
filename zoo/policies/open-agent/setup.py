from setuptools import setup

from open_agent.policy import VERSION

setup(
    name="open-agent",
    description="Trajectory planning agent based on OpEn",
    version=f"0.{VERSION}",
    packages=["open_agent"],
    install_requires=["opengen==0.6.2", "smarts"],
)
