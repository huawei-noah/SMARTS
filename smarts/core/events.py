from typing import NamedTuple


class Events(NamedTuple):
    collisions: bool
    off_route: bool
    reached_goal: bool
    reached_max_episode_steps: bool
    off_road: bool
    wrong_way: bool
    not_moving: bool
