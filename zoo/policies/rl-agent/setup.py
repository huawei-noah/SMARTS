from rl_agent import VERSION
from setuptools import setup

setup(
    name="rl-agent",
    description="lane space rl agent example",
    version=VERSION,
    packages=["rl_agent"],
    include_package_data=True,
    install_requires=["smarts>=0.6.1rc0", "tensorflow==2.4", "ray[rllib]==1.0.1.post1"],
)
