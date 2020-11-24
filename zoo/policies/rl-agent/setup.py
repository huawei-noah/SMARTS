from setuptools import setup

from rl_agent import VERSION

setup(
    name="rl-agent",
    description="lane space rl agent example",
    version=VERSION,
    packages=["rl_agent"],
    include_package_data=True,
    install_requires=["tensorflow==1.15", "smarts"],
)
