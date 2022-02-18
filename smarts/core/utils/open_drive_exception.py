class OpenDRIVEException(Exception):
    """An exception raised if opendrive utilities are required but not available."""

    @classmethod
    def required_to(cls, thing):
        return cls(
            f"""OpenDRIVE Package is required to simulate {thing}.
               You may not have installed the [opendrive] dependencies required to run the ray dependent example.
               Install them first using the command `pip install -e .[opendrive]` at the source directory to install the necessary packages"""
        )