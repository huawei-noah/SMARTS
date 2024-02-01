#version 330 core
// This script is intended to test that all of the observation buffers work in association with
//  the test script `test_renderers.py`.

uniform int step_count;
uniform int steps_completed;
uniform int events_collisions;
uniform int events_off_road;
uniform int events_off_route;
uniform int events_on_shoulder;
uniform int events_wrong_way;
uniform int events_not_moving;
uniform int events_reached_goal;
uniform int events_reached_max_episode_steps;
uniform int events_agents_done_alive;
uniform int events_interest_done;
uniform int ego_vehicle_state_road_id;
uniform int ego_vehicle_state_lane_id;
uniform int ego_vehicle_state_lane_index;
uniform int under_this_vehicle_control;
uniform int vehicle_type;

uniform float dt;
uniform float ego_vehicle_state_heading;
uniform float ego_vehicle_state_speed;
uniform float ego_vehicle_state_steering;
uniform float ego_vehicle_state_yaw_rate;
uniform float elapsed_sim_time;
uniform float distance_travelled;

uniform vec3 ego_vehicle_state_position;
uniform vec3 ego_vehicle_state_bounding_box;
uniform vec3 ego_vehicle_state_lane_position;

uniform int neighborhood_vehicle_states_road_id[10];
uniform int neighborhood_vehicle_states_lane_id[10];
uniform int neighborhood_vehicle_states_lane_index[10];
uniform int neighborhood_vehicle_states_interest[10];
uniform int waypoint_paths_lane_id[10];
uniform int waypoint_paths_lane_index[10];
uniform int road_waypoints_lanes_lane_id[10];
uniform int road_waypoints_lanes_lane_index[10];
uniform int via_data_near_via_points_lane_index[10];
uniform int via_data_near_via_points_road_id[10];
uniform int via_data_near_via_points_hit[10];
uniform int lidar_point_cloud_hits[100];
uniform int signals_light_state[10];

uniform float neighborhood_vehicle_states_heading[10];
uniform float neighborhood_vehicle_states_speed[10];
uniform float waypoint_paths_heading[10];
uniform float waypoint_paths_lane_width[10];
uniform float waypoint_paths_speed_limit[10];
uniform float waypoint_paths_lane_offset[10];
uniform float road_waypoints_lanes_heading[10];
uniform float road_waypoints_lanes_width[10];
uniform float road_waypoints_lanes_speed_limit[10];
uniform float road_waypoints_lanes_lane_offset[10];
uniform float via_data_near_via_points_required_speed[10];
uniform float signals_last_changed[10];

uniform vec2 ego_vehicle_state_linear_velocity[10];
uniform vec2 ego_vehicle_state_angular_velocity[10];
uniform vec2 ego_vehicle_state_linear_acceleration[10];
uniform vec2 ego_vehicle_state_angular_acceleration[10];
uniform vec2 ego_vehicle_state_linear_jerk[10];
uniform vec2 ego_vehicle_state_angular_jerk[10];
uniform vec2 waypoint_paths_pos[10];
uniform vec2 road_waypoints_lanes_pos[10];
uniform vec2 via_data_near_via_points_position[10];
uniform vec2 signals_stop_point[10];

uniform vec3 neighborhood_vehicle_states_position[10];
uniform vec3 neighborhood_vehicle_states_bounding_box[10];
uniform vec3 neighborhood_vehicle_states_lane_position[10];
uniform vec3 lidar_point_cloud_points[100];
uniform vec3 lidar_point_cloud_origin[100];
uniform vec3 lidar_point_cloud_direction[100];
// SIGNALS_CONTROLLED_LANES = "signals_controlled_lanes"

// Output color
out vec4 p3d_Color;

uniform vec2 iResolution;


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 rec_res = 1.0 / iResolution.xy;
    vec2 p = fragCoord.xy * rec_res;

    fragColor = vec4(0.0, 0.0, 0.0, 0.0);
}

void main(){
    int a = step_count + steps_completed + events_collisions + events_off_road + events_off_route 
        + events_on_shoulder + events_wrong_way + events_not_moving 
        + events_reached_goal + events_reached_max_episode_steps 
        + events_agents_done_alive + events_interest_done + ego_vehicle_state_road_id 
        + ego_vehicle_state_lane_id + ego_vehicle_state_lane_index + under_this_vehicle_control + vehicle_type;
    float b = dt + ego_vehicle_state_heading + ego_vehicle_state_speed 
        + ego_vehicle_state_steering + ego_vehicle_state_yaw_rate + elapsed_sim_time 
        + distance_travelled;
    vec3 c = ego_vehicle_state_position + ego_vehicle_state_bounding_box + ego_vehicle_state_lane_position;
    int d = neighborhood_vehicle_states_road_id[0] + neighborhood_vehicle_states_lane_id[0]
        + neighborhood_vehicle_states_lane_index[0] + neighborhood_vehicle_states_interest[0]
        + waypoint_paths_lane_id[0] + waypoint_paths_lane_index[0] + road_waypoints_lanes_lane_id[0]
        + road_waypoints_lanes_lane_index[0] + via_data_near_via_points_lane_index[0]
        + via_data_near_via_points_road_id[0] + via_data_near_via_points_hit[0]
        + lidar_point_cloud_hits[0] + signals_light_state[0];
    float e = neighborhood_vehicle_states_heading[0] + neighborhood_vehicle_states_speed[0]
        + waypoint_paths_heading[0] + waypoint_paths_lane_width[0] + waypoint_paths_speed_limit[0]
        + waypoint_paths_lane_offset[0] + road_waypoints_lanes_heading[0] + road_waypoints_lanes_width[0]
        + road_waypoints_lanes_speed_limit[0] + road_waypoints_lanes_lane_offset[0]
        + via_data_near_via_points_required_speed[0] + signals_last_changed[0];
    vec2 f = ego_vehicle_state_linear_velocity[0] + ego_vehicle_state_angular_velocity[0]
        + ego_vehicle_state_linear_acceleration[0] + ego_vehicle_state_angular_acceleration[0]
        + ego_vehicle_state_linear_jerk[0] + ego_vehicle_state_angular_jerk[0]
        + waypoint_paths_pos[0] + road_waypoints_lanes_pos[0] + via_data_near_via_points_position[0]
        + signals_stop_point[0];
    vec3 g = neighborhood_vehicle_states_position[0] + neighborhood_vehicle_states_bounding_box[0]
        + neighborhood_vehicle_states_lane_position[0] + lidar_point_cloud_points[0]
        + lidar_point_cloud_origin[0] + lidar_point_cloud_direction[0];
    
    mainImage( p3d_Color, gl_FragCoord.xy );
}