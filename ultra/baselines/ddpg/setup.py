from setuptools import setup

setup(
    name="ddpg",
    description="Ultra DDPG agent",
    version="0.1.1",
    packages=["ddpg"],
    include_package_data=True,
    install_requires=["tensorflow==1.15", "smarts"],
)
