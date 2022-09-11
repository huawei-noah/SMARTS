from dataclasses import dataclass


@dataclass
class Counts:
    crashes: float = 0
    """ Total number of crashed episodes. An episode is considered crashed if
    an agent becomes done due to collisions, driving off road, or reaching 
    max episode steps. Fractional values occur when only some agents crashes
    the episode in a multi-agent case.
    """
    episodes: int = 0
    """ Total number of episodes.
    """
    episode_agents: int = 0
    """ Total number of equivalent episodes. For an n-agent scenario, 
    n-episodes are added to the total.
    """
    steps: int = 0
    """ Total number of `act` steps taken.
    """
    steps_adjusted: int = 0
    """ Total number of `act` steps taken. Any crashed episode is assumed to
    have taken `_MAX_STEPS`.
    """
