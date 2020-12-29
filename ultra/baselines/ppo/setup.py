from setuptools import setup
setup(
    name="ppo",
    description="Ultra PPO agent",
    version='0.1.1',
    packages=["ppo"],
    include_package_data=True,
    install_requires=["tensorflow==1.15", "smarts"],
)
