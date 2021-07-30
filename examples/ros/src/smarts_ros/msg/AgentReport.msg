
string   agent_id

geometry_msgs/Pose  pose

float64  speed                 # in m/s

float64  distance_travelled    # in m

bool     is_done               # no more updates expected
bool     reached_goal

bool     did_collide

bool     is_wrong_way
bool     is_off_route
bool     is_off_road
bool     is_on_shoulder

bool     is_not_moving
