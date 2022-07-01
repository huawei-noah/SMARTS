class RayException(Exception):
    """An exception raised if ray package is required but not available."""

    @classmethod
    def required_to(cls, thing):
        return cls(
            f"""Ray Package is required to simulate {thing}.
               You may not have installed the [rllib] or [train] dependencies required to run the ray dependent example.
               Install them first using the command `pip install -e .[train, rllib]` at the source directory to install the package ray[rllib]==1.0.1.post1"""
        )
