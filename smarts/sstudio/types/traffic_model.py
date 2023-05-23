# MIT License
#
# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import collections.abc as collections_abc
from enum import IntEnum
from typing import Callable


class _SUMO_PARAMS_MODE(IntEnum):
    TITLE_CASE = 0
    KEEP_SNAKE_CASE = 1


class _SumoParams(collections_abc.Mapping):
    """For some Sumo params (e.g. LaneChangingModel) the arguments are in title case
    with a given prefix. Subclassing this class allows for an automatic way to map
    between PEP8-compatible naming and Sumo's.
    """

    def __init__(
        self, prefix, whitelist=[], mode=_SUMO_PARAMS_MODE.TITLE_CASE, **kwargs
    ):
        def snake_to_title(word):
            return "".join(x.capitalize() or "_" for x in word.split("_"))

        def keep_snake_case(word: str):
            w = word[0].upper() + word[1:]
            return "".join(x or "_" for x in w.split("_"))

        func: Callable[[str], str] = snake_to_title
        if mode == _SUMO_PARAMS_MODE.TITLE_CASE:
            pass
        elif mode == _SUMO_PARAMS_MODE.KEEP_SNAKE_CASE:
            func = keep_snake_case

        # XXX: On rare occasions sumo doesn't respect their own conventions
        #      (e.x. junction model's impatience).
        self._params = {key: kwargs.pop(key) for key in whitelist if key in kwargs}

        for key, value in kwargs.items():
            self._params[f"{prefix}{func(key)}"] = value

    def __iter__(self):
        return iter(self._params)

    def __getitem__(self, key):
        return self._params[key]

    def __len__(self):
        return len(self._params)

    def __hash__(self):
        return hash(frozenset(self._params.items()))

    def __eq__(self, other):
        return self.__class__ == other.__class__ and hash(self) == hash(other)


class LaneChangingModel(_SumoParams):
    """Models how the actor acts with respect to lane changes."""

    # For SUMO-specific attributes, see:
    # https://sumo.dlr.de/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html#lane-changing_models

    def __init__(self, **kwargs):
        super().__init__("lc", whitelist=["minGapLat"], **kwargs)


class JunctionModel(_SumoParams):
    """Models how the actor acts with respect to waiting at junctions."""

    def __init__(self, **kwargs):
        super().__init__("jm", whitelist=["impatience"], **kwargs)


class SmartsLaneChangingModel(LaneChangingModel):
    """Implements the simple lane-changing model built-into SMARTS.

    Args:
        cutin_prob (float, optional): Float value [0, 1] that
            determines the probability this vehicle will "arbitrarily" cut in
            front of an adjacent agent vehicle when it has a chance, even if
            there would otherwise be no reason to change lanes at that point.
            Higher values risk a situation where this vehicle ends up in a lane
            where it cannot maintain its planned route. If that happens, this
            vehicle will perform whatever its default behavior is when it
            completes its route. Defaults to 0.0.
        assertive (float, optional): Willingness to accept lower front and rear
            gaps in the target lane. The required gap is divided by this value.
            Attempts to match the semantics of the attribute in SUMO's default
            lane-changing model, see: ``https://sumo.dlr.de/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html#lane-changing_models``.
            Range: positive reals. Defaults to 1.0.
        dogmatic (bool, optional): If True, will cut-in when a suitable
            opportunity presents itself based on the above parameters, even if
            it means the risk of not not completing the assigned route;
            otherwise, will forego the chance. Defaults to True.
        hold_period (float, optional): The minimum amount of time (seconds) to
            remain in the agent's lane after cutting into it (including the
            time it takes within the lane to complete the maneuver). Must be
            non-negative. Defaults to 3.0.
        slow_down_after (float, optional): Target speed during the hold_period
            will be scaled by this value. Must be non-negative. Defaults to 1.0.
        multi_lane_cutin (bool, optional): If True, this vehicle will consider
            changing across multiple lanes at once in order to cut-in upon an
            agent vehicle when there's an opportunity. Defaults to False.
    """

    def __init__(
        self,
        cutin_prob: float = 0.0,
        assertive: float = 1.0,
        dogmatic: bool = True,
        hold_period: float = 3.0,
        slow_down_after: float = 1.0,
        multi_lane_cutin: bool = False,
    ):
        super().__init__(
            cutin_prob=cutin_prob,
            assertive=assertive,
            dogmatic=dogmatic,
            hold_period=hold_period,
            slow_down_after=slow_down_after,
            multi_lane_cutin=multi_lane_cutin,
        )


class SmartsJunctionModel(JunctionModel):
    """Implements the simple junction model built-into SMARTS.

    Args:
        yield_to_agents (str, optional): Defaults to "normal". 3 options are
            available, namely: (1) "always" - Traffic actors will yield to Ego
            and Social agents within junctions. (2) "never" - Traffic actors
            will never yield to Ego or Social agents within junctions.
            (3) "normal" - Traffic actors will attempt to honor normal
            right-of-way conventions, only yielding when an agent has the
            right-of-way. Examples of such conventions include (a) vehicles
            going straight have the right-of-way over turning vehicles;
            (b) vehicles on roads with more lanes have the right-of-way
            relative to vehicles on intersecting roads with less lanes;
            (c) all other things being equal, the vehicle to the right
            in a counter-clockwise sense has the right-of-way.
        wait_to_restart (float, optional): The amount of time in seconds
            after stopping at a signal or stop sign before this vehicle
            will start to go again. Defaults to 0.0.
    """

    def __init__(self, yield_to_agents: str = "normal", wait_to_restart: float = 0.0):
        super().__init__(
            yield_to_agents=yield_to_agents, wait_to_restart=wait_to_restart
        )
