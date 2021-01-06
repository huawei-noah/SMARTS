from setuptools import setup

setup(
    name="bdqn",
    description="Ultra Behavioral DQN agent",
    version="0.1.1",
    packages=["bdqn"],
    include_package_data=True,
    install_requires=["tensorflow==1.15", "smarts"],
)
