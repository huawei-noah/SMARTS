from collections import defaultdict, namedtuple

from shapely.geometry import LineString
from shapely.ops import split

# TODO: Move these into SMARTS and reuse from tests
BubbleGeometry = namedtuple(
    "BubbleGeometry", ["bubble", "airlock_entry", "airlock_exit", "airlock"],
)


def bubble_geometry(bubble, road_network):
    bubble_geometry_ = bubble.zone.to_geometry(road_network)
    airlock_geometry = bubble_geometry_.buffer(bubble.margin)
    split_x, split_y = airlock_geometry.centroid.coords[0]
    divider = LineString([(split_x, -999), (split_x, split_y + 999)])
    airlock_entry_geometry, airlock_exit_geometry = split(airlock_geometry, divider)
    return BubbleGeometry(
        bubble=bubble_geometry_,
        airlock_entry=airlock_entry_geometry,
        airlock_exit=airlock_exit_geometry,
        airlock=airlock_geometry,
    )
