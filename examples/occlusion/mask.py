import enum
import math
import os
import random
import subprocess
from collections import defaultdict, deque
from dataclasses import replace
from enum import IntEnum
from functools import cached_property, lru_cache, partial
from itertools import cycle, islice
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib import transforms
from PIL import Image
from shapely.affinity import rotate as shapely_rotate
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union

from smarts.core.agent import Agent
from smarts.core.agent_interface import (
    OGM,
    DoneCriteria,
    DrivableAreaGridMap,
    RoadWaypoints,
    Waypoints,
)
from smarts.core.observations import Observation, VehicleObservation
from smarts.core.road_map import Waypoint, interpolate_waypoints
from smarts.core.utils.math import squared_dist, vec_to_slope
from smarts.sstudio.graphics.bytemap2edge import (
    far_kernel,
    generate_edge_from_heightfield,
)
from smarts.sstudio.graphics.heightfield import CoordinateSampleMode, HeightField

T = TypeVar("T")


class Mode(IntEnum):
    UNFORMATTED = enum.auto()
    FORMATTED = enum.auto()
    DEFAULT = UNFORMATTED


class PointGenerator:
    @classmethod
    @lru_cache(maxsize=1000)
    def generate(cls, x, y, *args):
        return Point(x, y)


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
    direction_vector = np.array(
        (target_point[0] - center[0], target_point[1] - center[1])
    )
    ab_len = math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)
    unit_vector = direction_vector / ab_len

    return np.array(
        (center[0] + unit_vector[0] * distance, center[1] + unit_vector[1] * distance)
    )


def perpendicular_slope(slope):
    if slope == 0:  # Special case for horizontal line
        return float("inf")
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


def generate_shadow_mask_polygons(
    center, object_geometry: Polygon, radius: float
) -> List[Polygon]:
    """From a center point cast a mask away from the object

    Args:
        center (_type_): The source point of the cast.
        object_geometry (_type_): The object to cast the mask from. Assumes counterclockwise coordinates.
        radius (float): The distance from the source point to extent of the cast.

    Returns:
        List[Polygon]: A series of masks from the sides of the geometry facing away from the center origin.
    """
    out_shapes = []

    facing_away_edges = deque()
    for points in zip(
        object_geometry.exterior.coords,
        islice(cycle(object_geometry.exterior.coords), 1, 5),
    ):
        if is_edge_facing_away(points[0], points[1], center):
            continue

        facing_away_edges.append(points)

    # Edges need to be rotated so that the coordinates are in strict sequential order rather than cycled
    if len(facing_away_edges) > 0:
        for i in range(len(facing_away_edges) - 1):
            if facing_away_edges[i][1] == facing_away_edges[i + 1][0]:
                continue
            facing_away_edges.rotate(len(facing_away_edges) - (i + 1))
            break

    last_tangent_intersection = None
    intersections = []
    for point_a, point_b in reversed(facing_away_edges):
        ## If the intention is to generate a quadrilateral shadow cast towards the edge of the circle,
        # the center point of the line must cast through to generate a tangential line at the circle edge to guarentee
        # that each of the intersections from the bounding points of the original line to the tangent fall outside of the circle.
        # Otherwise, the nearest point would need to be used.
        midpoint = get_midpoint(point_a, point_b)
        point_on_tangent = find_point_past_target(center, midpoint, radius)
        tangent_slope = perpendicular_slope(
            vec_to_slope(
                np.array(
                    (center[0] - point_on_tangent[0], center[1] - point_on_tangent[1])
                )
            )
        )
        b2edge_intersection = find_tangent_intersection(
            center, radius, point_b, point_on_tangent, tangent_slope
        )
        intersection = b2edge_intersection
        if last_tangent_intersection is not None and squared_dist(
            center, b2edge_intersection
        ) < squared_dist(center, last_tangent_intersection):
            intersection = last_tangent_intersection
        intersections.append(intersection[:2])
        a2edge_intersection = find_tangent_intersection(
            center, radius, point_a, point_on_tangent, tangent_slope
        )
        last_tangent_intersection = a2edge_intersection

    if len(facing_away_edges) > 0:
        midpoint = get_midpoint(point_a, point_b)
        point_on_tangent = find_point_past_target(center, midpoint, radius)
        tangent_slope = perpendicular_slope(
            vec_to_slope(
                np.array(
                    (center[0] - point_on_tangent[0], center[1] - point_on_tangent[1])
                )
            )
        )
        point_a, _ = facing_away_edges[0]
        a2edge_intersection = find_tangent_intersection(
            center, radius, point_a, point_on_tangent, tangent_slope
        )
        intersections.append(a2edge_intersection[:2])

    shell = [
        *(a for a, _ in facing_away_edges),
        facing_away_edges[-1][-1],
        *intersections,
    ]
    poly = Polygon(shell)
    if poly.is_valid:
        if isinstance(poly, (MultiPolygon, GeometryCollection)):
            for g in poly.geoms:
                if not hasattr(g, "exterior"):
                    continue
                out_shapes.append(poly)
        else:
            out_shapes.append(poly)
    else:
        pass
        # breakpoint()

    return out_shapes


def find_tangent_intersection(center, radius, point_a, point_on_tangent, tangent_slope):
    if squared_dist(point_a, center) > radius**2:
        a2edge_intersection = point_on_tangent
    else:
        # This is the slope from the center to the near bounding points
        a_slope = vec_to_slope(
            np.array((center[0] - point_a[0], center[1] - point_a[1]))
        )
        # This is where the slope intersects with the circle tangent.
        a2edge_intersection = find_line_intersection(
            a_slope, tangent_slope, point1=center, point2=point_on_tangent
        )

    return a2edge_intersection


def generate_circle_polygon(center, radius) -> Polygon:
    return PointGenerator.generate(*center[:2]).buffer(radius)


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
        return float("inf"), float("inf")  # Infinite intersection points

    # Calculate the x-coordinate of the intersection point
    x = (b2 - b1) / (slope1 - slope2)

    # Calculate the y-coordinate of the intersection point
    y = slope1 * x + b1

    return x, y


def gen_circle_mask(center, radius):
    circle_area = generate_circle_polygon(center, radius)
    return circle_area


def gen_shadow_masks(center, vehicle_states, radius, mode=Mode.FORMATTED):
    if mode == Mode.FORMATTED:
        accessor = lambda v: (
            v["position"],
            *v["dimensions"][:2],  # length, width
            v["heading"],
        )
    else:
        accessor = lambda v: (
            v.position[:2],
            *v.bounding_box.as_lwh[:2],
            v.heading,
        )
    masks: List[Polygon] = []
    for vehicle_state in vehicle_states:
        vehicle_shape = generate_vehicle_polygon(*accessor(vehicle_state))
        new_masks = generate_shadow_mask_polygons(center, vehicle_shape, radius)
        masks.append(unary_union(new_masks))
    return masks


def apply_masks(center, vehicle_states: T, radius, mode=Mode.FORMATTED) -> T:
    # Test that vehicles are within visible range
    observation_area = gen_circle_mask(center, radius)
    remaining_vehicle_states = []
    remaining_vehicle_points = []

    if mode == Mode.FORMATTED:
        gen = lambda vs: PointGenerator.generate(vs["position"])
    else:
        gen = lambda vs: PointGenerator.generate(vs.position)

    for vehicle_state in vehicle_states:
        position_point = gen(vehicle_state)
        if not observation_area.contains(position_point):
            continue
        remaining_vehicle_states.append(vehicle_state)
        remaining_vehicle_points.append(position_point)

    # Test that vehicles are not occluded
    occlusion_masks = gen_shadow_masks(center, remaining_vehicle_states, radius, mode)
    final_vehicle_states = []
    for vehicle_state, position_point in zip(
        remaining_vehicle_states, remaining_vehicle_points
    ):
        # discard any vehicle state that is not included
        for shadow_polygon in occlusion_masks:
            if shadow_polygon.contains(position_point):
                break  # state is masked
        else:
            # if not masked
            final_vehicle_states.append(vehicle_state)
    return final_vehicle_states


def one_by_r_attentuation(*, dist2: float, **_):
    """The crude formula for sound pressure attenuation.

    Args:
        dist2 (float): The squared distance to attenuate.
    """

    return 1 / max(1, math.sqrt(abs(dist2)))


def one_by_r2_attentuation(*, dist2: float, **_):
    """The crude formula for sound intensity attenuation.

    Args:
        dist2 (float): The squared distance to attenuate.
    """

    return 1 / max(1, abs(dist2))


def clamp(x, mn, mx):
    return min(max(x, mn), mx)


def smoothstep(*, dist2: float, max_edge2: float, min_edge2: float = 0):
    dist = max_edge2 - min_edge2
    x = clamp((dist2 - min_edge2) / dist, 0, 1)

    return x**2 * (3.0 - 2 * x)


def smootherstep(*, dist2: float, max_edge2: float, min_edge2: float = 0):
    dist = max_edge2 - min_edge2
    x = clamp((dist2 - min_edge2) / dist, 0, 1)

    return x**3 * (3.0 * x * (2 * x - 5.0) + 10.0)


def one_minus_smoothstep(*, dist2: float, max_edge2: float, min_edge2: float = 0):
    return 1 - smoothstep(dist2=dist2, max_edge2=max_edge2, min_edge2=min_edge2)


def one_minus_smootherstep(*, dist2: float, max_edge2: float, min_edge2: float = 0):
    return 1 - smootherstep(dist2=dist2, max_edge2=max_edge2, min_edge2=min_edge2)


def gauss_noise(base, mu=0, theta=0.15, sigma=0.3):
    noise = theta * (mu - base) + sigma * np.random.randn(1)
    return (base + noise)[0]


def gauss_noise2(base, mu=0, sigma=0.078):
    return base + random.gauss(mu, sigma)


def sample_weighted_binary_probability(normalized_bar):
    r = random.randint(0, 1e9)

    return r * 1e-9 > normalized_bar


def certainty_displace(
    center_point: Tuple[float, float],
    target_point: Tuple[float, float],
    perturb_target: Any,
    certainty_attenuation_fn=one_by_r2_attentuation,
    uncertainty_noise_fn=gauss_noise2,
    coin_fn=sample_weighted_binary_probability,
    max_sigma=0.073,
    max_observable_radius: float = 10,
) -> Tuple[float, float, float]:
    c_x, c_y = center_point
    t_x, t_y = target_point

    d_x2 = (t_x - c_x) ** 2
    d_y2 = (t_y - c_y) ** 2

    distance_attenuation = certainty_attenuation_fn(
        dist2=d_x2 + d_y2, max_edge2=max_observable_radius**2
    )

    # Coinflip chance of displacement
    should_displace = coin_fn(normalized_bar=distance_attenuation)

    meta = dict(should_displace=should_displace, attenuation=distance_attenuation)

    if not should_displace:
        return perturb_target, meta

    # Generate displacement
    sigma = max_sigma * (1 - distance_attenuation)
    meta["sigma"] = sigma

    if isinstance(perturb_target, (list, np.ndarray, Sequence)):
        return (
            [uncertainty_noise_fn(pt_v, sigma=sigma) for pt_v in perturb_target],
            meta,
        )
    else:
        return uncertainty_noise_fn(perturb_target, sigma=sigma), meta


_vehicle_states: List[Dict[str, Union[float, Tuple[float, float]]]] = [
    {"position": (1, 1), "dimensions": (2, 1), "heading": math.pi / 4},
    {"position": (-1, -1), "dimensions": (1.5, 0.8), "heading": math.pi / 2},
    {"position": (3, 2), "dimensions": (1.7, 0.9), "heading": 2 * math.pi / 3},
    {"position": (-2, 3), "dimensions": (1.3, 0.7), "heading": 7 * math.pi / 6},
    {"position": (0.5, -2), "dimensions": (1.8, 0.6), "heading": 5 * math.pi / 6},
    {"position": (-3, -2.5), "dimensions": (1.4, 0.7), "heading": math.pi},
    {"position": (2.5, -1), "dimensions": (1.6, 0.8), "heading": math.pi / 6},
    {"position": (-2, 1.5), "dimensions": (1.5, 0.8), "heading": 3 * math.pi / 4},
    {"position": (4, 3), "dimensions": (1.7, 0.9), "heading": math.pi / 3},
    {"position": (-3.5, -3), "dimensions": (1.3, 0.7), "heading": math.pi},
    {"position": (-3.5, 8), "dimensions": (1.3, 0.7), "heading": math.pi},
    {"position": (-13, 5), "dimensions": (1.3, 0.7), "heading": math.pi},
]


def downgrade_waypoints(
    center: Tuple[float, float],
    waypoints: List[List[Waypoint]],
    wp_space_resolution: float,
    max_observable_radius: float,
    waypoint_displacement_factor: float = 1e-5,
    waypoint_heading_factor: float = math.pi * 0.125,
) -> List[List[Waypoint]]:
    assert wp_space_resolution > 0, "Resolution must be a real number."

    center = center[:2]

    out_waypoints = defaultdict(list)

    for i, lane in enumerate(waypoints):
        current_delta = 0
        for waypoint, n_waypoint in zip(lane[:-1], lane[1:]):
            if waypoint.lane_offset - n_waypoint.lane_offset == 0:
                continue
            wps, used = interpolate_waypoints(
                waypoint,
                n_waypoint,
                wp_space_resolution - current_delta % wp_space_resolution,
                wp_space_resolution,
            )
            current_delta += used
            out_waypoints[i].extend(wps)

    return [
        [
            replace(
                wp,
                pos=certainty_displace(
                    center_point=center,
                    target_point=wp.pos,
                    perturb_target=wp.pos,
                    max_sigma=waypoint_displacement_factor,
                    certainty_attenuation_fn=one_minus_smootherstep,
                    max_observable_radius=max_observable_radius,
                )[0],
                heading=certainty_displace(
                    center_point=center,
                    target_point=wp.pos,
                    perturb_target=wp.heading,
                    max_sigma=waypoint_heading_factor,
                    max_observable_radius=max_observable_radius,
                )[0],
            )
            for wp in l
        ]
        for l in out_waypoints.values()
    ]


def downgrade_vehicles(
    center: Tuple[float, float],
    neighborhood_vehicle_states: List[VehicleObservation],
    mode=Mode.FORMATTED,
):
    if mode:
        pos_accessor = lambda o: o.position
        heading_accessor = lambda o: o.heading
    else:
        pos_accessor = lambda o: o["position"]
        heading_accessor = lambda o: o["heading"]
    center = center[:2]
    return [
        v._replace(
            position=certainty_displace(
                center_point=center,
                target_point=pos_accessor(v)[:2],
                perturb_target=pos_accessor(v),
                max_sigma=2e-1,
            )[0],
            heading=certainty_displace(
                center_point=center,
                target_point=pos_accessor(v)[:2],
                perturb_target=heading_accessor(v),
                max_sigma=math.pi / 16,
            )[0],
        )
        for v in neighborhood_vehicle_states
    ]


record_dir = Path("./vaw/vaw")


class VectorAgentWrapper(Agent):
    def __init__(self, inner_agent, mode, observation_radius=40) -> None:
        self._inner_agent = inner_agent
        self._observation_radius = observation_radius
        self._mode = mode
        os.makedirs(record_dir, exist_ok=True)
        super().__init__()

    @lru_cache(1)
    def _get_perlin(
        self,
        width,
        height,
        smooth_iterations,
        seed,
        table_dim,
        shift,
        amplitude=5,
        granularity=0.02,
    ):
        # from smarts.sstudio.graphics.perlin_bytemap import generate_perlin

        # return generate_perlin(width, height, smooth_iterations, seed, table_dim, shift)
        from smarts.sstudio.graphics.perlin_bytemap import generate_simplex

        return generate_simplex(
            width,
            height,
            seed,
            shift,
            octaves=2,
            amplitude=amplitude,
            granularity=granularity,
        )

    def _rotate_image(self, heightfield: HeightField, heading: float):
        image = Image.fromarray(heightfield.data, "L")
        rotated_image = image.rotate(math.degrees(heading))

        return HeightField(
            data=np.asarray(rotated_image).reshape(heightfield.resolution),
            size=heightfield.size,
        )

    def act(self, obs: Optional[Observation], **configs):
        img_width, img_height = (
            obs.drivable_area_grid_map.metadata.width,
            obs.drivable_area_grid_map.metadata.height,
        )

        if self._mode == Mode.FORMATTED:
            pos_accessor = lambda o: o["position"]
            len_accessor = lambda o: o["box"][0]
            width_accessor = lambda o: o["box"][1]
            heading_accessor = lambda o: o["heading"]
            ego_accessor = lambda o: o["ego_vehicle_state"]
            nvs_accessor = lambda o: o["neighborhood_vehicle_states"]
            wpp_accessor = lambda o: o["waypoint_paths"]
        else:
            pos_accessor = lambda o: o.position
            len_accessor = lambda o: o.bounding_box.length
            width_accessor = lambda o: o.bounding_box.width
            heading_accessor = lambda o: o.heading
            ego_accessor = lambda o: o.ego_vehicle_state
            nvs_accessor = lambda o: o.neighborhood_vehicle_states
            wpp_accessor = lambda o: o.waypoint_paths

        fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
        ax: plt.Axes
        if obs is None:
            return None
        ego_state = ego_accessor(obs)
        ego_heading = heading_accessor(ego_state)
        _observation_center = pos_accessor(ego_state)[:2]

        vehicle_hf = HeightField.from_rgb(obs.occupancy_grid_map.data)
        height_scaling = 5
        drivable_hf = HeightField(
            obs.drivable_area_grid_map.data, (img_width, img_height)
        )
        edge_hf = generate_edge_from_heightfield(
            drivable_hf,
            far_kernel(),
            0 * height_scaling,
            0.2 * height_scaling,
        )
        perlin_hf = self._get_perlin(
            img_width,
            img_height,
            0,
            42,
            2048,
            (
                _observation_center[0] / self._observation_radius,
                _observation_center[1] / -self._observation_radius,
            ),
            amplitude=int(1 * height_scaling),
        )
        perlin_hf = self._rotate_image(perlin_hf, -ego_heading)
        offroad_perlin = self._get_perlin(
            img_width,
            img_height,
            0,
            42,
            2048,
            (
                _observation_center[0] / self._observation_radius,
                _observation_center[1] / -self._observation_radius,
            ),
            amplitude=int(2 * height_scaling),
            granularity=0.5,
        )
        offroad_hf = self._rotate_image(offroad_perlin, -ego_heading).scale_by(
            drivable_hf.inverted()
        )
        hf = edge_hf.add(perlin_hf).max(vehicle_hf).max(offroad_hf)

        los = hf.to_line_of_sight(
            (0, 0),
            1 * height_scaling,
            0.3,
            coordinate_sample_mode=CoordinateSampleMode.POINT,
        )
        extent = [
            -img_width * 0.5,
            img_width * 0.5,
            -img_height * 0.5,
            img_height * 0.5,
        ]
        tr = (
            transforms.Affine2D()
            .rotate_deg(math.degrees(ego_heading))
            .translate(*_observation_center)
        )

        image_data = obs.drivable_area_grid_map.data
        image_data = hf.data * 10
        image_data = los.data
        # image_data = offroad_hf.data
        ax.imshow(
            image_data,
            cmap="gray",
            vmin=0,
            vmax=255,
            transform=tr + ax.transData,
            extent=extent,
        )

        _observation_radius = self._observation_radius
        observation_inverse_mask: Polygon = generate_circle_polygon(
            _observation_center, _observation_radius
        )

        v_geom = generate_vehicle_polygon(
            pos_accessor(ego_state),
            len_accessor(ego_state),
            width_accessor(ego_state),
            heading_accessor(ego_state),
        )
        ax.plot(*v_geom.exterior.xy, color="y")

        # draw vehicle center points
        for vs in nvs_accessor(obs):
            ax.plot(*PointGenerator.generate(*pos_accessor(vs)).xy, "y+")

        vehicles_to_downgrade: List[VehicleObservation] = [
            v
            for v in nvs_accessor(obs)
            if observation_inverse_mask.contains(
                PointGenerator.generate(*pos_accessor(v))
            )
        ]
        occlusion_masks: List[Polygon] = gen_shadow_masks(
            _observation_center, vehicles_to_downgrade, _observation_radius, self._mode
        )

        for poly in occlusion_masks:
            # if not hasattr(poly, "exterior"):
            #     continue
            observation_inverse_mask = observation_inverse_mask.difference(poly)

        if isinstance(observation_inverse_mask, (MultiPolygon, GeometryCollection)):
            for g in observation_inverse_mask.geoms:
                if not hasattr(g, "exterior"):
                    continue
                ax.plot(*g.exterior.xy)
        else:
            ax.plot(*observation_inverse_mask.exterior.xy)

        final_vehicle_states = []
        for vehicle_state, position_point in zip(
            vehicles_to_downgrade,
            (PointGenerator.generate(*pos_accessor(v)) for v in vehicles_to_downgrade),
        ):
            # discard any vehicle state that is not included
            for shadow_polygon in occlusion_masks:
                if shadow_polygon.contains(position_point):
                    break  # state is masked
            else:
                # if not masked
                final_vehicle_states.append(vehicle_state)

        downgraded_vehicles = downgrade_vehicles(
            pos_accessor(ego_state), vehicles_to_downgrade, mode=self._mode
        )

        wp_downgrading_fn = partial(
            downgrade_waypoints,
            center=pos_accessor(ego_state),
            wp_space_resolution=2,
            max_observable_radius=_observation_radius * 0.5,
            waypoint_displacement_factor=0.6,
        )
        waypoints_to_downgrade = [
            [
                wp
                for wp in l
                if observation_inverse_mask.contains(
                    PointGenerator.generate(*wp.position)
                )
            ]
            for l in wpp_accessor(obs)
        ]
        downgraded_waypoints = wp_downgrading_fn(waypoints=waypoints_to_downgrade)

        road_waypoints_to_downgrade = [
            [
                wp
                for wp in path
                if observation_inverse_mask.contains(
                    PointGenerator.generate(*wp.position)
                )
            ]
            for paths in obs.road_waypoints.lanes.values()
            for path in paths
        ]
        downgraded_road_waypoints = wp_downgrading_fn(
            waypoints=road_waypoints_to_downgrade
        )

        for vehicle in nvs_accessor(obs):
            v_geom = generate_vehicle_polygon(
                pos_accessor(vehicle),
                len_accessor(vehicle),
                width_accessor(vehicle),
                heading_accessor(vehicle),
            )
            ax.plot(*v_geom.exterior.xy, color="b")
        for vehicle in downgraded_vehicles:
            v_geom = generate_vehicle_polygon(
                pos_accessor(vehicle),
                len_accessor(vehicle),
                width_accessor(vehicle),
                heading_accessor(vehicle),
            )
            ax.plot(*v_geom.exterior.xy, color="r")

        self.draw_waypoints(road_waypoints_to_downgrade, pos_accessor, ax, color="y")
        self.draw_waypoints(downgraded_road_waypoints, pos_accessor, ax, color="g")
        self.draw_waypoints(wpp_accessor(obs), pos_accessor, ax, color="b")
        self.draw_waypoints(downgraded_waypoints, pos_accessor, ax, color="r")

        # Ego vehicle center
        ax.plot(*PointGenerator.generate(*_observation_center).xy, "r+")
        xlim = _observation_radius + 15
        ylim = xlim
        ax.set_xlim(-xlim + _observation_center[0], xlim + _observation_center[0])
        ax.set_ylim(-ylim + _observation_center[1], ylim + _observation_center[1])

        if not obs.steps_completed % 1:
            ax.axis("off")

            plt.savefig(record_dir / f"A1_{obs.steps_completed}.jpg")
        plt.close("all")
        plt.cla()
        dowgraded_obs = obs._replace(
            neighborhood_vehicle_states=_vehicle_states,
            waypoint_paths=downgraded_waypoints,
        )
        return self._inner_agent.act(dowgraded_obs, **configs)

    def draw_waypoints(self, waypoints, pos_accessor, ax, color):
        wp_line_strings: List[shapely.LineString] = [
            shapely.LineString([pos_accessor(wp)[:2] for wp in l])
            for l in waypoints
            if len(l) > 1
        ]
        for ls in wp_line_strings:
            ax.plot(*ls.coords.xy, color=color)


def obscure_main(vehicle_states, mode=Mode.FORMATTED):
    _observation_center = np.array((-1, 5))
    _observation_radius = 10

    # This mask will include any geometry that intersects it (exclude geometry that does not intersect)
    observation_inverse_mask = generate_circle_polygon(
        _observation_center, _observation_radius
    )
    x, y = observation_inverse_mask.exterior.xy
    plt.plot(x, y)

    if mode == Mode.FORMATTED:
        gen = lambda vs: PointGenerator.generate(*vs["position"])
    else:
        gen = lambda vs: PointGenerator.generate(*vs.position)

    # draw vehicle center points
    for vs in vehicle_states:
        vehicle_center = gen(vs)
        plt.plot(*vehicle_center.xy, "ro")

    vehicle_states = apply_masks(
        _observation_center, vehicle_states, _observation_radius, mode=Mode.FORMATTED
    )
    for vs in vehicle_states:
        v_geom = generate_vehicle_polygon(
            vs["position"], vs["dimensions"][0], vs["dimensions"][1], vs["heading"]
        )
        plt.plot(*v_geom.exterior.xy, color="r")

        masks = generate_shadow_mask_polygons(
            _observation_center, v_geom, _observation_radius
        )
        for mask in masks:
            plt.plot(*mask.exterior.xy, color="g")

    # Ego vehicle center
    plt.plot([_observation_center[0]], [_observation_center[1]], "r+")

    plt.axis("equal")
    plt.savefig("occlusion.jpg")


def consume(video_source_pattern="A1_%d.jpg", video_name="sd_obs.mp4"):
    os.chdir(record_dir)
    subprocess.call(
        [
            "ffmpeg",
            "-framerate",
            "8",
            "-y",
            "-i",
            video_source_pattern,
            "-r",
            "30",
            "-pix_fmt",
            "yuv420p",
            video_name,
        ]
    )
    subprocess.call(["mv", video_name, f"../{video_name}"])
    subprocess.call(
        [
            "rm",
            f"./{video_source_pattern[0]}*",
        ]
    )
    os.chdir("../..")


def dummy_main(output_file):
    import gymnasium as gym

    from smarts.env.gymnasium.hiway_env_v1 import HiWayEnvV1
    from smarts.zoo.registry import make

    agent_spec = make("zoo.policies:keep-lane-agent-v0")
    observation_radius = 40
    agents = {
        "A1": VectorAgentWrapper(
            agent_spec.build_agent(),
            Mode.UNFORMATTED,
            observation_radius=observation_radius,
        )
    }

    env = HiWayEnvV1(
        scenarios=[
            # "./scenarios/sumo/intersections/4lane_t",
            # "./smarts/diagnostic/n_sumo_actors/200_actors",
            # "./scenarios/argoverse/straight/00a445fb-7293-4be6-adbc-e30c949b6cf7_agents_1/",
            "./scenarios/argoverse/turn/0a60b442-56b0-46c3-be45-cf166a182b67_agents_1/",
            # "./scenarios/argoverse/turn/0a764a82-b44e-481e-97e7-05e1f1f925f6_agents_1/",
            # "./scenarios/argoverse/turn/0bf054e3-7698-4b86-9c98-626df2dee9f4_agents_1/",
        ],
        observation_options="unformatted",
        action_options="unformatted",
        agent_interfaces={
            "A1": replace(
                agent_spec.interface,
                neighborhood_vehicle_states=True,
                drivable_area_grid_map=DrivableAreaGridMap(
                    width=observation_radius * 2,
                    height=observation_radius * 2,
                    resolution=1,
                ),
                occupancy_grid_map=OGM(
                    width=observation_radius * 2,
                    height=observation_radius * 2,
                    resolution=1,
                ),
                road_waypoints=RoadWaypoints(horizon=50),
                waypoint_paths=Waypoints(lookahead=50),
                done_criteria=DoneCriteria(
                    collision=False, off_road=False, off_route=False
                ),
            )
        },
    )

    terms = {"__all__": False}
    obs, info = env.reset()
    with env:
        for _ in range(50):
            if terms["__all__"]:
                break
            acts = {a_id: a.act(obs.get(a_id)) for a_id, a in agents.items()}

            obs, rewards, terms, truncs, infos = env.step(acts)

    consume(video_name=output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Downgrader")
    parser.add_argument("mode", nargs=1, default=0, type=int)
    args = parser.parse_args()

    if args.mode[0] == 0:
        dummy_main("trial.mp4")
    elif args.mode[0] == 1:
        obscure_main(_vehicle_states)
