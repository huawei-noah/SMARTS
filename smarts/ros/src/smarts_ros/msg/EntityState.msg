string entity_id

uint8 ENTITY_TYPE_UNSPECIFIED = 0
uint8 ENTITY_TYPE_CAR = 1
uint8 ENTITY_TYPE_TRUCK = 2
uint8 ENTITY_TYPE_TRAILER = 3
uint8 ENTITY_TYPE_BUS = 4
uint8 ENTITY_TYPE_COACH = 5
uint8 ENTITY_TYPE_PEDESTRIAN = 6
uint8 ENTITY_TYPE_MOTORCYCLE = 7
# more to be added...

uint8 entity_type     # Value sould come from above

geometry_msgs/Pose  pose
geometry_msgs/Twist velocity
geometry_msgs/Accel acceleration

float64 length        # in meters
float64 width         # in meters
float64 height        # in meters (can be 0)

