
string   agent_id      # must be unique in the simulation

string   agent_type    # can be used as lookup into zoo registry
string   agent_ver     # optional version (if not specified, defaults to "latest")

string   params_json   # optional JSON string containing config parameters

uint8 VEHICLE_TYPE_UNSPECIFIED = 0
uint8 VEHICLE_TYPE_CAR = 1
uint8 VEHICLE_TYPE_TRUCK = 2
uint8 VEHICLE_TYPE_TRAILER = 3
uint8 VEHICLE_TYPE_BUS = 4
uint8 VEHICLE_TYPE_COACH = 5
uint8 VEHICLE_TYPE_PEDESTRIAN = 6
uint8 VEHICLE_TYPE_MOTORCYCLE = 7
# more to be added...

uint8    veh_type      # Value sould come from above (only CAR is currently supported)
float64  veh_length    # in meters
float64  veh_width     # in meters
float64  veh_height    # in meters (can be 0)


geometry_msgs/Pose     start_pose
float64                start_speed    # linear speed in m/s (in direction of initial heading)

geometry_msgs/Pose[]   via_points     # optional points to try to hit on the way to end_pose (NYI)

geometry_msgs/Pose     end_pose       # optional, if not specified, vehicle will randomly cruise after task

