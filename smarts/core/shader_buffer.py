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
from enum import Enum


class CameraSensorID(Enum):
    """Describes default names for camera configuration."""

    DRIVABLE_AREA_GRID_MAP = "drivable_area_grid_map"
    TOP_DOWN_RGB = "top_down_rgb"
    OCCUPANCY_GRID_MAP = "ogm"
    OCCLUSION = "occlusion_map"


class BufferID(Enum):
    """The names of the different buffers available for camera rendering."""

    DELTA_TIME = "dt"
    STEP_COUNT = "step_count"
    STEPS_COMPLETED = "steps_completed"
    ELAPSED_SIM_TIME = "elapsed_sim_time"

    EVENTS_COLLISIONS = "events_collisions"
    EVENTS_OFF_ROAD = "events_off_road"
    EVENTS_OFF_ROUTE = "events_off_route"
    EVENTS_ON_SHOULDER = "events_on_shoulder"
    EVENTS_WRONG_WAY = "events_wrong_way"
    EVENTS_NOT_MOVING = "events_not_moving"
    EVENTS_REACHED_GOAL = "events_reached_goal"
    EVENTS_REACHED_MAX_EPISODE_STEPS = "events_reached_max_episode_steps"
    EVENTS_AGENTS_ALIVE_DONE = "events_agents_done_alive"
    EVENTS_INTEREST_DONE = "events_interest_done"

    EGO_VEHICLE_STATE_POSITION = "ego_vehicle_state_position"
    EGO_VEHICLE_STATE_BOUNDING_BOX = "ego_vehicle_state_bounding_box"
    EGO_VEHICLE_STATE_HEADING = "ego_vehicle_state_heading"
    EGO_VEHICLE_STATE_SPEED = "ego_vehicle_state_speed"
    EGO_VEHICLE_STATE_STEERING = "ego_vehicle_state_steering"
    EGO_VEHICLE_STATE_YAW_RATE = "ego_vehicle_state_yaw_rate"
    EGO_VEHICLE_STATE_ROAD_ID = "ego_vehicle_state_road_id"
    EGO_VEHICLE_STATE_LANE_ID = "ego_vehicle_state_lane_id"
    EGO_VEHICLE_STATE_LANE_INDEX = "ego_vehicle_state_lane_index"
    EGO_VEHICLE_STATE_LINEAR_VELOCITY = "ego_vehicle_state_linear_velocity"
    EGO_VEHICLE_STATE_ANGULAR_VELOCITY = "ego_vehicle_state_angular_velocity"
    EGO_VEHICLE_STATE_LINEAR_ACCELERATION = "ego_vehicle_state_linear_acceleration"
    EGO_VEHICLE_STATE_ANGULAR_ACCELERATION = "ego_vehicle_state_angular_acceleration"
    EGO_VEHICLE_STATE_LINEAR_JERK = "ego_vehicle_state_linear_jerk"
    EGO_VEHICLE_STATE_ANGULAR_JERK = "ego_vehicle_state_angular_jerk"
    EGO_VEHICLE_STATE_LANE_POSITION = "ego_vehicle_state_lane_position"

    UNDER_THIS_VEHICLE_CONTROL = "under_this_vehicle_control"

    NEIGHBORHOOD_VEHICLE_STATES_POSITION = "neighborhood_vehicle_states_position"
    NEIGHBORHOOD_VEHICLE_STATES_BOUNDING_BOX = (
        "neighborhood_vehicle_states_bounding_box"
    )
    NEIGHBORHOOD_VEHICLE_STATES_HEADING = "neighborhood_vehicle_states_heading"
    NEIGHBORHOOD_VEHICLE_STATES_SPEED = "neighborhood_vehicle_states_speed"
    NEIGHBORHOOD_VEHICLE_STATES_ROAD_ID = "neighborhood_vehicle_states_road_id"
    NEIGHBORHOOD_VEHICLE_STATES_LANE_ID = "neighborhood_vehicle_states_lane_id"
    NEIGHBORHOOD_VEHICLE_STATES_LANE_INDEX = "neighborhood_vehicle_states_lane_index"
    NEIGHBORHOOD_VEHICLE_STATES_LANE_POSITION = (
        "neighborhood_vehicle_states_lane_position"
    )
    NEIGHBORHOOD_VEHICLE_STATES_INTEREST = "neighborhood_vehicle_states_interest"

    WAYPOINT_PATHS_POSITION = "waypoint_paths_pos"
    WAYPOINT_PATHS_HEADING = "waypoint_paths_heading"
    WAYPOINT_PATHS_LANE_ID = "waypoint_paths_lane_id"
    WAYPOINT_PATHS_LANE_WIDTH = "waypoint_paths_lane_width"
    WAYPOINT_PATHS_SPEED_LIMIT = "waypoint_paths_speed_limit"
    WAYPOINT_PATHS_LANE_INDEX = "waypoint_paths_lane_index"
    WAYPOINT_PATHS_LANE_OFFSET = "waypoint_paths_lane_offset"

    DISTANCE_TRAVELLED = "distance_travelled"

    ROAD_WAYPOINTS_POSITION = "road_waypoints_lanes_pos"
    ROAD_WAYPOINTS_HEADING = "road_waypoints_lanes_heading"
    ROAD_WAYPOINTS_LANE_ID = "road_waypoints_lanes_lane_id"
    ROAD_WAYPOINTS_LANE_WIDTH = "road_waypoints_lanes_width"
    ROAD_WAYPOINTS_SPEED_LIMIT = "road_waypoints_lanes_speed_limit"
    ROAD_WAYPOINTS_LANE_INDEX = "road_waypoints_lanes_lane_index"
    ROAD_WAYPOINTS_LANE_OFFSET = "road_waypoints_lanes_lane_offset"

    VIA_DATA_NEAR_VIA_POINTS_POSITION = "via_data_near_via_points_position"
    VIA_DATA_NEAR_VIA_POINTS_LANE_INDEX = "via_data_near_via_points_lane_index"
    VIA_DATA_NEAR_VIA_POINTS_ROAD_ID = "via_data_near_via_points_road_id"
    VIA_DATA_NEAR_VIA_POINTS_REQUIRED_SPEED = "via_data_near_via_points_required_speed"
    VIA_DATA_NEAR_VIA_POINTS_HIT = "via_data_near_via_points_hit"

    LIDAR_POINT_CLOUD_POINTS = "lidar_point_cloud_points"
    LIDAR_POINT_CLOUD_HITS = "lidar_point_cloud_hits"
    LIDAR_POINT_CLOUD_ORIGIN = "lidar_point_cloud_origin"
    LIDAR_POINT_CLOUD_DIRECTION = "lidar_point_cloud_direction"

    VEHICLE_TYPE = "vehicle_type"

    SIGNALS_LIGHT_STATE = "signals_light_state"
    SIGNALS_STOP_POINT = "signals_stop_point"
    # SIGNALS_CONTROLLED_LANES = "signals_controlled_lanes"
    SIGNALS_LAST_CHANGED = "signals_last_changed"
