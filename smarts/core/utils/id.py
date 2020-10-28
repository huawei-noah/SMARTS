"""A helper to ensure consistent naming of IDs within SMARTS Platform."""
import uuid


class Id(str):
    def __init__(self, dtype: str, identifier: str):
        self._dtype = dtype
        self._identifier = identifier

    def __new__(cls, dtype: str, identifier: str):
        return super(Id, cls).__new__(cls, f"{dtype}-{identifier}")

    def __getnewargs__(self):
        return (self._dtype, self._identifier)

    @classmethod
    def new(cls, dtype: str):
        """E.g. boid-93572825"""
        return Id(dtype=dtype, identifier=str(uuid.uuid4())[:8])

    @classmethod
    def parse(cls, id_: str):
        split = -8 - 1  # should be "-"
        if id_[split] != "-":
            raise ValueError(
                f"id={id_} is invalid, format should be <type>-<8_char_uuid>"
            )

        return Id(dtype=id_[:split], identifier=id_[split + 1 :])

    @property
    def dtype(self):
        return self._dtype


class SocialAgentId(Id):
    """
    >>> SocialAgentId.new("keep-lane", group="all")
    'social-agent-all-keep-lane'
    >>> isinstance(SocialAgentId.new("keep-lane"), str)
    True
    """

    DTYPE = "social-agent"

    @classmethod
    def new(cls, name: str, group: str = None):
        identifier = "-".join([group, name]) if group is not None else name
        return SocialAgentId(dtype=SocialAgentId.DTYPE, identifier=identifier)
