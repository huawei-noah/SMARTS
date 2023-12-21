import importlib.resources as pkg_resources
import math
import os
import random
import subprocess
from abc import ABCMeta
from collections import defaultdict, deque
from dataclasses import replace
from functools import lru_cache, partial
from itertools import cycle, islice
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple, TypeVar, Union

import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib import transforms
from PIL import Image
from shapely import prepared
from shapely.affinity import rotate as shapely_rotate
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union
from shapely.prepared import PreparedGeometry

from smarts.core import glsl
from smarts.core.agent import Agent
from smarts.core.agent_interface import (
    OGM,
    RGB,
    CameraSensorName,
    CustomRender,
    CustomRenderCameraDependency,
    DoneCriteria,
    DrivableAreaGridMap,
    OcclusionMap,
    RoadWaypoints,
    Waypoints,
)
from smarts.core.colors import Colors
from smarts.core.observations import Observation, VehicleObservation
from smarts.core.road_map import Waypoint, interpolate_waypoints
from smarts.core.utils.core_math import squared_dist, vec_to_slope
from smarts.core.utils.observations import points_to_pixels
from smarts.env.utils.observation_conversion import ObservationOptions
from smarts.sstudio.graphics.bytemap2edge import (
    far_kernel,
    generate_edge_from_heightfield,
)
from smarts.sstudio.graphics.heightfield import CoordinateSampleMode, HeightField

T = TypeVar("T")
VIDEO_PREFIX: Final[str] = "A1_"
IMAGE_SUFFIX: Final[str] = "jpg"


OUTPUT_DIR: Final[Path] = Path("./vaw/vaw")


class PropertyAccessorUtil:
    def __init__(self, mode: ObservationOptions) -> None:
        if mode in (ObservationOptions.multi_agent, ObservationOptions.full):
            self.position_accessor = lambda o: o["position"]
            self.len_accessor = lambda o: o["box"][0]
            self.width_accessor = lambda o: o["box"][1]
            self.heading_accessor = lambda o: o["heading"]
            self.ego_accessor = lambda o: o["ego_vehicle_state"]
            self.nvs_accessor = lambda o: o["neighborhood_vehicle_states"]
            self.wpp_accessor = lambda o: o["waypoint_paths"]
            self.waypoint_position_accessor = lambda o: self.position_accessor(
                self.wpp_accessor(o)
            )
        else:
            self.position_accessor = lambda o: o.position
            self.len_accessor = lambda o: o.bounding_box.length
            self.width_accessor = lambda o: o.bounding_box.width
            self.heading_accessor = lambda o: o.heading
            self.ego_accessor = lambda o: o.ego_vehicle_state
            self.nvs_accessor = lambda o: o.neighborhood_vehicle_states
            self.wpp_accessor = lambda o: o.waypoint_paths

            def _pad(waypoints):
                max_lane_wps = max([len(lane) for lane in waypoints])
                base = np.zeros((len(waypoints), max_lane_wps, 3), dtype=np.float64)

                for i, lane in enumerate(waypoints):
                    for j, wp in enumerate(lane):
                        base[i, j, :2] = self.position_accessor(wp)
                return base

            self.waypoint_position_accessor = lambda o: _pad(self.wpp_accessor(o))


class PointGenerator:
    @classmethod
    @lru_cache(maxsize=1000)
    def cache_generate(cls, x, y, *_):
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
    point_a, point_b = None, None
    for point_a, point_b in reversed(facing_away_edges):
        ## If the intention is to generate a quadrilateral shadow cast towards the edge of the circle,
        # the center point of the line must cast through to generate a tangential line at the circle edge to guarantee
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

    if point_a is not None and point_b is not None:
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
        from shapely.validation import explain_validity

        print(explain_validity(poly))

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
    return PointGenerator.cache_generate(*center[:2]).buffer(radius)


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


def gen_shadow_masks(
    center, vehicle_states, radius, mode=ObservationOptions.multi_agent
):
    if mode in (ObservationOptions.multi_agent, ObservationOptions.full):
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


def apply_masks(
    center, vehicle_states: T, radius, mode=ObservationOptions.multi_agent
) -> T:
    # Test that vehicles are within visible range
    observation_area = gen_circle_mask(center, radius)
    remaining_vehicle_states = []
    remaining_vehicle_points = []

    if mode in (ObservationOptions.multi_agent, ObservationOptions.full):
        gen = lambda vs: PointGenerator.cache_generate(*vs["position"])
    else:
        gen = lambda vs: PointGenerator.cache_generate(*vs.position)

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


def one_by_r_attenuation(*, dist2: float, **_):
    """The crude formula for sound pressure attenuation.

    Args:
        dist2 (float): The squared distance to attenuate.
    """

    return 1 / max(1, math.sqrt(abs(dist2)))


def one_by_r2_attenuation(*, dist2: float, **_):
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


def gaussian_noise(base, mu=0, theta=0.15, sigma=0.3):
    noise = theta * (mu - base) + sigma * np.random.randn(1)
    return (base + noise)[0]


def gaussian_noise2(base, mu=0, sigma=0.078):
    return base + random.gauss(mu, sigma)


def sample_weighted_binary_probability(normalized_bar):
    r = random.randint(0, 1e9)

    return r * 1e-9 > normalized_bar


def certainty_displace(
    center_point: Tuple[float, float],
    target_point: Tuple[float, float],
    perturb_target: Any,
    certainty_attenuation_fn=one_by_r2_attenuation,
    uncertainty_noise_fn=gaussian_noise2,
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
    mode=ObservationOptions.multi_agent,
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


class AugmentationWrapper(Agent, metaclass=ABCMeta):
    def __init__(
        self, mode, observation_radius, output_dir, agent_name, record: bool
    ) -> None:
        self._observation_radius = observation_radius
        self._output_dir = output_dir
        self._agent_name = agent_name
        self._mode = mode
        self._pa = PropertyAccessorUtil(self._mode)
        self._record = record
        super().__init__()

    def export_video_image(self, ax, obs: Observation):
        ego_state = self._pa.ego_accessor(obs)
        _observation_center = self._pa.position_accessor(ego_state)[:2]
        ax.plot(*PointGenerator.cache_generate(*_observation_center).xy, "r+")
        xlim = self._observation_radius + 15
        ylim = xlim
        ax.set_xlim(-xlim + _observation_center[0], xlim + _observation_center[0])
        ax.set_ylim(-ylim + _observation_center[1], ylim + _observation_center[1])
        if obs.steps_completed % 1 == 0:
            ax.axis("off")

            plt.savefig(
                self._output_dir
                / f"{self._agent_name}_{obs.steps_completed}.{IMAGE_SUFFIX}"
            )
        plt.close("all")
        plt.cla()

    def generate_video(self, video_source_pattern="%d"):
        if not self._record:
            return
        full_video_source_pattern = (
            f"{self._agent_name}_{video_source_pattern}.{IMAGE_SUFFIX}"
        )
        video_name = f"{self._agent_name}.mp4"
        os.chdir(self._output_dir)
        subprocess.call(
            [
                "ffmpeg",
                "-framerate",
                "8",
                "-y",
                "-i",
                full_video_source_pattern,
                "-r",
                "30",
                "-pix_fmt",
                "yuv420p",
                video_name,
            ],
            stderr=subprocess.DEVNULL,
        )
        subprocess.call(
            [
                "rm",
                "-f",
                f"{VIDEO_PREFIX}*.jpg",
            ]
        )
        subprocess.call(["mv", video_name, f"../{video_name}"])
        os.chdir("../..")


class VectorAgentWrapper(AugmentationWrapper):
    def __init__(
        self,
        inner_agent,
        mode,
        agent_name,
        observation_radius=40,
        record: bool = True,
        output_dir=OUTPUT_DIR,
    ) -> None:
        self._inner_agent = inner_agent
        os.makedirs(output_dir, exist_ok=True)
        super().__init__(
            mode=mode,
            output_dir=output_dir,
            agent_name=agent_name,
            observation_radius=observation_radius,
            record=record,
        )

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

        fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
        ax: plt.Axes
        if obs is None:
            return None
        ego_state = self._pa.ego_accessor(obs)
        ego_heading = self._pa.heading_accessor(ego_state)
        _observation_center = self._pa.position_accessor(ego_state)[:2]

        vehicle_hf = HeightField(obs.occupancy_grid_map.data, (img_width, img_height))
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
        vehicle_hf = HeightField(obs.obfuscation_grid_map.data, (img_width, img_height))
        image_data = vehicle_hf.data
        ax.imshow(
            image_data,
            cmap="gray",
            vmin=0,
            vmax=255,
            transform=tr + ax.transData,
            extent=extent,
        )

        observation_inverse_mask: Polygon = generate_circle_polygon(
            _observation_center, self._observation_radius
        )
        prep_observation_inverse_mask: PreparedGeometry = prepared.prep(
            observation_inverse_mask
        )

        v_geom = generate_vehicle_polygon(
            self._pa.position_accessor(ego_state),
            self._pa.len_accessor(ego_state),
            self._pa.width_accessor(ego_state),
            self._pa.heading_accessor(ego_state),
        )
        ax.plot(*v_geom.exterior.xy, color="y")

        # draw vehicle center points
        for v in self._pa.nvs_accessor(obs):
            ax.plot(
                *PointGenerator.cache_generate(*self._pa.position_accessor(v)).xy, "y+"
            )

        vehicles = [v for v in self._pa.nvs_accessor(obs)]
        vehicles_to_downgrade: List[VehicleObservation] = [
            v
            for v in vehicles
            if prep_observation_inverse_mask.contains(
                PointGenerator.cache_generate(*self._pa.position_accessor(v))
            )
        ]
        occlusion_masks: List[Polygon] = gen_shadow_masks(
            _observation_center,
            vehicles_to_downgrade,
            self._observation_radius,
            self._mode,
        )

        for poly in occlusion_masks:
            # if not hasattr(poly, "exterior"):
            #     continue
            observation_inverse_mask = observation_inverse_mask.difference(poly)
        prep_observation_inverse_mask = prepared.prep(observation_inverse_mask)

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
            (
                PointGenerator.cache_generate(*self._pa.position_accessor(v))
                for v in vehicles_to_downgrade
            ),
        ):
            # discard any vehicle state that is not included
            for shadow_polygon in occlusion_masks:
                if shadow_polygon.contains(position_point):
                    break  # state is masked
            else:
                # if not masked
                final_vehicle_states.append(vehicle_state)

        downgraded_vehicles = downgrade_vehicles(
            self._pa.position_accessor(ego_state),
            vehicles_to_downgrade,
            mode=self._mode,
        )

        wp_downgrading_fn = partial(
            downgrade_waypoints,
            center=self._pa.position_accessor(ego_state),
            wp_space_resolution=2,
            max_observable_radius=self._observation_radius * 0.5,
            waypoint_displacement_factor=0.6,
        )
        waypoints_to_downgrade = [
            [
                wp
                for wp in l
                if prep_observation_inverse_mask.contains(
                    PointGenerator.cache_generate(*wp.position)
                )
            ]
            for l in self._pa.wpp_accessor(obs)
        ]
        downgraded_waypoints = wp_downgrading_fn(waypoints=waypoints_to_downgrade)

        road_waypoints_to_downgrade = [
            [
                wp
                for wp in path
                if prep_observation_inverse_mask.contains(
                    PointGenerator.cache_generate(*wp.position)
                )
            ]
            for paths in obs.road_waypoints.lanes.values()
            for path in paths
        ]
        downgraded_road_waypoints = wp_downgrading_fn(
            waypoints=road_waypoints_to_downgrade
        )

        for vehicle in self._pa.nvs_accessor(obs):
            v_geom = generate_vehicle_polygon(
                self._pa.position_accessor(vehicle),
                self._pa.len_accessor(vehicle),
                self._pa.width_accessor(vehicle),
                self._pa.heading_accessor(vehicle),
            )
            ax.plot(*v_geom.exterior.xy, color="b")
        for vehicle in downgraded_vehicles:
            v_geom = generate_vehicle_polygon(
                self._pa.position_accessor(vehicle),
                self._pa.len_accessor(vehicle),
                self._pa.width_accessor(vehicle),
                self._pa.heading_accessor(vehicle),
            )
            ax.plot(*v_geom.exterior.xy, color="r")

        self.draw_waypoints(
            road_waypoints_to_downgrade, self._pa.position_accessor, ax, color="y"
        )
        self.draw_waypoints(
            downgraded_road_waypoints, self._pa.position_accessor, ax, color="g"
        )
        self.draw_waypoints(
            self._pa.wpp_accessor(obs), self._pa.position_accessor, ax, color="b"
        )
        self.draw_waypoints(
            downgraded_waypoints, self._pa.position_accessor, ax, color="r"
        )

        self.export_video_image(ax, obs)
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


class OcclusionAgentWrapper(AugmentationWrapper):
    def __init__(
        self,
        inner_agent,
        mode: ObservationOptions,
        agent_name,
        observation_radius: float = 40.0,
        scale: float = 1.0,
        record: bool = True,
        output_dir: Path = OUTPUT_DIR,
    ) -> None:
        self._inner_agent = inner_agent
        self._wps_color = np.array(Colors.Green.value[:3]) * 255
        self._no_color = np.zeros(shape=(3,))
        self._inverse_scale = 1.0 / scale
        os.makedirs(output_dir, exist_ok=True)
        super().__init__(
            mode=mode,
            agent_name=agent_name,
            output_dir=output_dir,
            observation_radius=observation_radius,
            record=record,
        )

    def act(self, obs: Optional[Observation], **configs):

        img_width, img_height = (
            obs.drivable_area_grid_map.metadata.width,
            obs.drivable_area_grid_map.metadata.height,
        )

        fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
        ax: plt.Axes
        if obs is None:
            return None
        ego_state = self._pa.ego_accessor(obs)
        ego_heading = self._pa.heading_accessor(ego_state)
        _observation_center = self._pa.position_accessor(ego_state)[:2]
        extent = [
            -img_width * 0.5,
            img_width * 0.5,
            -img_height * 0.5,
            img_height * 0.5,
        ]

        vehicle_hf = HeightField(obs.obfuscation_grid_map.data, (img_width, img_height))

        rgb_ego = obs.custom_renders[0].data.copy()
        waypoint_paths = np.array(self._pa.waypoint_position_accessor(obs))

        position = self._pa.position_accessor(ego_state)
        heading = self._pa.heading_accessor(ego_state)
        for path in waypoint_paths[0:11, 3:35, 0:3]:
            wps_valid = points_to_pixels(
                points=path,
                center_position=position,
                heading=heading,
                width=obs.obfuscation_grid_map.metadata.width,
                height=obs.obfuscation_grid_map.metadata.height,
                resolution=obs.obfuscation_grid_map.metadata.resolution,
            )
            for point in wps_valid:
                img_x, img_y = point[0], point[1]
                if all(rgb_ego[img_y, img_x, :] != self._no_color):
                    rgb_ego[img_y, img_x, :] = self._wps_color
                else:
                    break

        if self._record:
            tr = (
                transforms.Affine2D()
                .rotate_deg(math.degrees(ego_heading))
                .translate(*_observation_center)
            )
            image_data = vehicle_hf.data
            ax.imshow(
                rgb_ego,
                # cmap="gray",
                vmin=0,
                vmax=255,
                transform=tr + ax.transData,
                extent=extent,
            )
            # ax.imshow(
            #     image_data,
            #     cmap="gray",
            #     vmin=0,
            #     vmax=255,
            #     transform=tr + ax.transData,
            #     extent=extent,
            # )

            for vehicle in self._pa.nvs_accessor(obs):
                vehicle_pos = np.array(self._pa.position_accessor(vehicle))[:2]
                v_geom = generate_vehicle_polygon(
                    _observation_center
                    + (vehicle_pos - _observation_center) * self._inverse_scale,
                    self._pa.len_accessor(vehicle) * self._inverse_scale,
                    self._pa.width_accessor(vehicle) * self._inverse_scale,
                    self._pa.heading_accessor(vehicle),
                )
                ax.plot(*v_geom.exterior.xy, color="b")

            self.export_video_image(ax, obs)

        dowgraded_obs = obs._replace()
        return self._inner_agent.act(dowgraded_obs, **configs)


def occlusion_main():
    from smarts.env.gymnasium.hiway_env_v1 import HiWayEnvV1
    from smarts.zoo.registry import make

    observation_formatting = ObservationOptions.unformatted

    agent_spec = make("zoo.policies:keep-lane-agent-v0")
    observation_radius = 60
    resolution = 1
    w, h = observation_radius * 2, observation_radius * 2
    agent_count = 1
    agents: Dict[str, OcclusionAgentWrapper] = {
        f"A{c}": OcclusionAgentWrapper(
            agent_spec.build_agent(),
            observation_formatting,
            observation_radius=observation_radius,
            scale=resolution,
            agent_name=f"A{c}",
        )
        for c in range(1, agent_count + 1)
    }

    with pkg_resources.path(glsl, "map_values.frag") as frag_shader:
        agent_interface = replace(
            agent_spec.interface,
            neighborhood_vehicle_states=True,
            drivable_area_grid_map=DrivableAreaGridMap(
                width=w,
                height=h,
                resolution=resolution,
            ),
            occupancy_grid_map=OGM(
                width=w,
                height=h,
                resolution=resolution,
            ),
            top_down_rgb=RGB(
                width=w,
                height=h,
                resolution=resolution,
            ),
            occlusion_map=OcclusionMap(
                width=w,
                height=h,
                resolution=resolution,
                surface_noise=True,
            ),
            road_waypoints=RoadWaypoints(horizon=50),
            waypoint_paths=Waypoints(lookahead=50),
            done_criteria=DoneCriteria(
                collision=False, off_road=False, off_route=False
            ),
            custom_renders=(
                CustomRender(
                    name="noc",
                    fragment_shader_path=frag_shader,
                    dependencies=(
                        CustomRenderCameraDependency(
                            camera_dependency_name=CameraSensorName.OCCLUSION,
                            variable_name="iChannel0",
                        ),
                        CustomRenderCameraDependency(
                            camera_dependency_name=CameraSensorName.TOP_DOWN_RGB,
                            variable_name="iChannel1",
                        ),
                    ),
                    width=w,
                    height=h,
                    resolution=resolution,
                ),
            ),
        )

    with HiWayEnvV1(
        scenarios=[
            # "./scenarios/sumo/intersections/4lane_t",
            # "./smarts/diagnostic/n_sumo_actors/200_actors",
            # "./scenarios/argoverse/straight/00a445fb-7293-4be6-adbc-e30c949b6cf7_agents_1/",
            "./scenarios/argoverse/turn/0a60b442-56b0-46c3-be45-cf166a182b67_agents_1/",
            # "./scenarios/argoverse/turn/0a764a82-b44e-481e-97e7-05e1f1f925f6_agents_1/",
            # "./scenarios/argoverse/turn/0bf054e3-7698-4b86-9c98-626df2dee9f4_agents_1/",
        ],
        observation_options=observation_formatting,
        action_options="unformatted",
        agent_interfaces={"A1": agent_interface},
        fixed_timestep_sec=0.1,
    ) as env:
        terms = {"__all__": False}
        obs, info = env.reset()
        for _ in range(70):
            if terms["__all__"]:
                break
            acts = {a_id: a.act(obs.get(a_id)) for a_id, a in agents.items()}

            obs, rewards, terms, truncs, infos = env.step(acts)

    for _, a in agents.items():
        a.generate_video()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Downgrader")
    args = parser.parse_args()

    occlusion_main()
