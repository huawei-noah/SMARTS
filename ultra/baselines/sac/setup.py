from setuptools import setup

setup(
    name="sac",
    description="Ultra SAC agent",
    version="0.1.1",
    packages=["sac"],
    include_package_data=True,
    install_requires=["tensorflow==1.15", "smarts"],
)
