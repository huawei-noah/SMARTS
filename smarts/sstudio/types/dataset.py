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
import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TrafficHistoryDataset:
    """Describes a dataset containing trajectories (time-stamped positions)
    for a set of vehicles.  Often these have been collected by third parties
    from real-world observations, hence the name 'history'.  When used
    with a SMARTS scenario, traffic vehicles will move on the map according
    to their trajectories as specified in the dataset.  These can be mixed
    with other types of traffic (such as would be specified by an object of
    the Traffic type in this DSL).  In order to use this efficiently, SMARTS
    will pre-process ('import') the dataset when the scenario is built."""

    name: str
    """a unique name for the dataset"""
    source_type: str
    """the type of the dataset; supports values in (``NGSIM``, ``INTERACTION``, ``Waymo``, ``Argoverse``)"""
    input_path: Optional[str] = None
    """a relative or absolute path to the dataset; if omitted, dataset will not be imported"""
    scenario_id: Optional[str] = None
    """a unique ID for a Waymo scenario. For other datasets, this field will be None."""
    x_margin_px: float = 0.0
    """x offset of the map from the data (in pixels)"""
    y_margin_px: float = 0.0
    """y offset of the map from the data (in pixels)"""
    swap_xy: bool = False
    """if True, the x and y axes the dataset coordinate system will be swapped"""
    flip_y: bool = False
    """if True, the dataset will be mirrored around the x-axis"""
    filter_off_map: bool = False
    """if True, then any vehicle whose coordinates on a time step fall outside of the map's bounding box will be removed for that time step"""

    map_lane_width: float = 3.7
    """This is used to figure out the map scale, which is map_lane_width / real_lane_width_m.  (So use `real_lane_width_m` here for 1:1 scale - the default.)  It's also used in SMARTS for detecting off-road, etc."""
    real_lane_width_m: float = 3.7
    """Average width in meters of the dataset's lanes in the real world.  US highway lanes are about 12 feet (or ~3.7m, the default) wide."""
    speed_limit_mps: Optional[float] = None
    """used by SMARTS for the initial speed of new agents being added to the scenario"""

    heading_inference_window: int = 2
    """When inferring headings from positions, a sliding window (moving average) of this size will be used to smooth inferred headings and reduce their dependency on any individual position changes.  Defaults to 2 if not specified."""
    heading_inference_min_speed: float = 2.2
    """Speed threshold below which a vehicle's heading is assumed not to change.  This is useful to prevent abnormal heading changes that may arise from noise in position estimates in a trajectory dataset dominating real position changes in situations where the real position changes are very small.  Defaults to 2.2 m/s if not specified."""
    max_angular_velocity: Optional[float] = None
    """When inferring headings from positions, each vehicle's angular velocity will be limited to be at most this amount (in rad/sec) to prevent lateral-coordinate noise in the dataset from causing near-instantaneous heading changes."""
    default_heading: float = 1.5 * math.pi
    """A heading in radians to be used by default for vehicles if the headings are not present in the dataset and cannot be inferred from position changes (such as on the first time step)."""
