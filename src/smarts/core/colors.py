# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
from enum import Enum


# Color channel order: RGBA
class Colors(Enum):
    Red = (210 / 255, 30 / 255, 30 / 255, 1)
    Rose = (196 / 255, 0, 84 / 255, 1)
    Burgundy = (127 / 255, 0, 1 / 255, 1)
    Orange = (237 / 255, 109 / 255, 0, 1)
    Yellow = (255 / 255, 190 / 255, 40 / 255, 1)
    GreenTransparent = (98 / 255, 178 / 255, 48 / 255, 0.3)
    Silver = (192 / 255, 192 / 255, 192 / 255, 1)
    Black = (0, 0, 0, 1)

    DarkBlue = (5 / 255, 5 / 255, 70 / 255, 1)
    Blue = (0, 153 / 255, 1, 1)
    LightBlue = (173 / 255, 216 / 255, 230 / 255, 1)
    BlueTransparent = (60 / 255, 170 / 255, 200 / 255, 0.6)

    DarkCyan = (47 / 255, 79 / 255, 79 / 255, 1)
    CyanTransparent = (48 / 255, 181 / 255, 197 / 255, 0.5)

    DarkPurple = (50 / 255, 30 / 255, 50 / 255, 1)
    Purple = (127 / 255, 0, 127 / 255, 1)

    DarkGrey = (80 / 255, 80 / 255, 80 / 255, 1)
    Grey = (119 / 255, 136 / 255, 153 / 255, 1)
    LightGreyTransparent = (221 / 255, 221 / 255, 221 / 255, 0.1)

    OffWhite = (200 / 255, 200 / 255, 200 / 255, 1)
    White = (1, 1, 1, 1)


class SceneColors(Enum):
    Agent = Colors.Red.value
    SocialAgent = Colors.Silver.value
    SocialVehicle = Colors.Silver.value

    Road = Colors.DarkGrey.value
    EgoWaypoint = Colors.CyanTransparent.value
    EgoDrivenPath = Colors.CyanTransparent.value
    BubbleLine = Colors.LightGreyTransparent.value
    MissionRoute = Colors.GreenTransparent.value
    LaneDivider = Colors.OffWhite.value
    EdgeDivider = Colors.Yellow.value

    EnvisionColors = {
        "agent": Agent,
        "social_agent": SocialAgent,
        "social_vehicle": SocialVehicle,
        "road": Road,
        "ego_waypoint": EgoWaypoint,
        "ego_driven_path": EgoDrivenPath,
        "bubble_line": BubbleLine,
        "mission_route": MissionRoute,
        "lane_divider": LaneDivider,
        "edge_divider": EdgeDivider,
    }
