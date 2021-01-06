from setuptools import setup

setup(
    name="dqn",
    description="Ultra DQN agent",
    version="0.1.1",
    packages=["dqn"],
    include_package_data=True,
    install_requires=["tensorflow==1.15", "smarts"],
)
