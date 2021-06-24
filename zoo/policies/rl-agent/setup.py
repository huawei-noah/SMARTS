from rl_agent import VERSION
from setuptools import setup

setup(
    name="rl-agent",
    description="lane space rl agent example",
    version=VERSION,
    packages=["rl_agent"],
    include_package_data=True,
    install_requires=["tensorflow==2.2.1", "smarts"],
)
