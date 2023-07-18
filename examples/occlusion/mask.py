import math
from itertools import islice, cycle
from typing import Dict, List, Tuple, Union


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from shapely.affinity import rotate as shapely_rotate
from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.geometry import Polygon
from shapely.geometry import box as shapely_box


def generate_vehicle_polygon(position, length, width, heading) -> Polygon:
    """Returns a bounding box around the vehicle."""
    half_len = 0.5 * length
    half_width = 0.5 * width
    poly = shapely_box(
        position[0] - half_width,
        position[1] - half_len,
        position[0] + half_width,
        position[1] + half_len,
    )
    return shapely_rotate(poly, heading, use_radians=True)

def find_point_past_target(center, target_point, distance: float):
    direction_vector = np.array((target_point[0] - center[0], target_point[1] - center[1]))
    ab_len = math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)
    unit_vector = direction_vector / ab_len

    return np.array((center[0] + unit_vector[0] * distance, center[1] + unit_vector[1] * distance))

def perpendicular_slope(slope):
    if slope == 0:  # Special case for horizontal line
        return float('inf')
    return -1 / slope

def get_midpoint(point1, point2):
    return (np.array(point1) + np.array(point2)) / 2 

def is_edge_facing_away(edge_start, edge_end, reference_point):
    edge_vector = np.array(edge_end) - np.array(edge_start)
    perpendicular_vector = np.array([edge_vector[1], -edge_vector[0]])
    ref_vector = np.array(reference_point) - np.array(edge_end)

    dot_product = np.dot(perpendicular_vector, ref_vector)

    if dot_product > 0:
        return True
    elif dot_product < 0:
        return False
    else:
        return None  # Reference point lies on the edge


def generate_shadow_mask_polygons(center, object_geometry: Polygon, radius: float) -> List[Polygon]:
    """From a center point cast a mask away from the object

    Args:
        center (_type_): The source point of the cast.
        object_geometry (_type_): The object to cast the mask from. Assumes counterclockwise coordinates.
        radius (float): The distance from the source point to extent of the cast.

    Returns:
        List[Polygon]: A series of masks from the sides of the geometry facing away from the center origin.
    """
    out_shapes = []

    for point_a, point_b in zip(object_geometry.exterior.coords, islice(cycle(object_geometry.exterior.coords), 1, 5)):
        if is_edge_facing_away(point_a, point_b, center):
            continue
        midpoint = get_midpoint(point_a, point_b)
        mid_a = find_point_past_target(center, midpoint, radius)
        tagent_slope = perpendicular_slope(vector_to_slope(np.array((center[0] - mid_a[0], center[1] - mid_a[1]))))

        a_slope = vector_to_slope(np.array((center[0] - point_a[0], center[1] - point_a[1])))
        b_slope = vector_to_slope(np.array((center[0] - point_b[0], center[1] - point_b[1])))

        a2edge_intersection = find_line_intersection(a_slope, tagent_slope, point1=center, point2=mid_a)
        b2edge_intersection = find_line_intersection(b_slope, tagent_slope, point1=center, point2=mid_a)


        shell = [point_a, point_b, (b2edge_intersection[0], b2edge_intersection[1]), (a2edge_intersection[0], a2edge_intersection[1])]
        out_shapes.append(Polygon(shell))

    return out_shapes

def generate_circle_polygon(center, radius) -> Polygon:
    return Point(*center).buffer(radius)

def vector_to_slope(direction_vector):
    return direction_vector[1] / (direction_vector[0] or 1e-10)

def find_line_intersection(slope1, slope2, point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    # Calculate the y-intercepts
    b1 = y1 - slope1 * x1
    b2 = y2 - slope2 * x2
    
    # Check if slopes are parallel (same slope with different intercepts)
    if slope1 == slope2 and b1 != b2:
        return None  # No intersection (parallel lines)
    
    # Check if slopes are identical (same slope with same intercept)
    if slope1 == slope2 and b1 == b2:
        return float('inf'), float('inf')  # Infinite intersection points

    # Calculate the x-coordinate of the intersection point
    x = (b2 - b1) / (slope1 - slope2)

    # Calculate the y-coordinate of the intersection point
    y = slope1 * x + b1

    return x, y

def gen_circle_mask(center, radius):
    circle_area = generate_circle_polygon(center, radius)
    return circle_area

def gen_shadow_masks(center, vehicle_states, radius):
    masks: List[Polygon] = []
    for vehicle_state in vehicle_states:
        vehicle_shape = generate_vehicle_polygon(
            vehicle_state["position"],
            *vehicle_state["dimensions"][:2], # length, width
            vehicle_state["heading"]
        )
        new_masks = generate_shadow_mask_polygons(center, vehicle_shape, radius)
        masks.extend(new_masks)
    return masks

def apply_masks(center, vehicle_states, radius) -> List[Dict[str, Union[float, Tuple[float, float]]]]:
    # Test that vehicles are within visible range
    observation_area = gen_circle_mask(center, radius)
    remaining_vehicle_states = []
    for vehicle_state in vehicle_states:
        position_point = Point(vehicle_state["position"])
        if not observation_area.contains(position_point):
            continue
        remaining_vehicle_states.append(vehicle_state)

    # Test that vehicles are not occluded
    occlusion_masks = gen_shadow_masks(center, remaining_vehicle_states, radius)
    final_vehicle_states = []
    for vehicle_state in vehicle_states:
        position_point = Point(vehicle_state["position"])
        # discard any vehicle state that is not included
        for shadow_polygon in occlusion_masks:
            if shadow_polygon.contains(position_point):
                break # state is masked
        else:
            # if not masked
            final_vehicle_states.append(vehicle_state)
    return final_vehicle_states


_vehicle_states: List[Dict[str, Union[float, Tuple[float, float]]]] = [
    {
        "position": (1, 1),
        "dimensions": (2, 1),
        "heading": math.pi / 4
    },
    {
        "position": (-1, -1),
        "dimensions": (1.5, 0.8),
        "heading": math.pi / 2
    },
    {
        "position": (3, 2),
        "dimensions": (1.7, 0.9),
        "heading": 2 * math.pi / 3
    },
    {
        "position": (-2, 3),
        "dimensions": (1.3, 0.7),
        "heading": 7 * math.pi / 6
    },
    {
        "position": (0.5, -2),
        "dimensions": (1.8, 0.6),
        "heading": 5 * math.pi / 6
    },
    {
        "position": (-3, -2.5),
        "dimensions": (1.4, 0.7),
        "heading": math.pi
    },
    {
        "position": (2.5, -1),
        "dimensions": (1.6, 0.8),
        "heading": math.pi / 6
    },
    {
        "position": (-2, 1.5),
        "dimensions": (1.5, 0.8),
        "heading": 3 * math.pi / 4
    },
    {
        "position": (4, 3),
        "dimensions": (1.7, 0.9),
        "heading": math.pi / 3
    },
    {
        "position": (-3.5, -3),
        "dimensions": (1.3, 0.7),
        "heading": math.pi
    },
    {
        "position": (-3.5, 8),
        "dimensions": (1.3, 0.7),
        "heading": math.pi
    }
]

if __name__ == "__main__":
    _center = np.array((-1, 5))
    _radius = 10

    center_geom = generate_circle_polygon(_center, 0.1)
    x, y = center_geom.exterior.xy
    plt.plot(x, y)

    circle_mask = generate_circle_polygon(_center, _radius)
    x, y = circle_mask.exterior.xy
    plt.plot(x, y)

    # draw vehicle center points
    for vs in _vehicle_states:
        vehicle_center = generate_circle_polygon(vs["position"], 0.1)
        x, y = vehicle_center.exterior.xy
        plt.plot(x, y)

    _vehicle_states = apply_masks(_center, _vehicle_states, _radius)
    for vs in _vehicle_states:
        vehicle = generate_vehicle_polygon(vs["position"], vs["dimensions"][0], vs["dimensions"][1], vs["heading"])
        x, y = vehicle.exterior.xy
        plt.plot(x, y)

        masks = generate_shadow_mask_polygons(_center, vehicle, _radius)
        for mask in masks:
            x, y = mask.exterior.xy
            plt.plot(x, y)


    plt.axis('equal')
    plt.savefig('occlusion.jpg')