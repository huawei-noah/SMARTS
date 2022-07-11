import math

def goal_region_reward(threshold, goal_x, goal_y, cur_x, cur_y):
    eucl_distance = math.sqrt((goal_x - cur_x)**2 + (goal_y - cur_y)**2)

    if eucl_distance <= threshold:
        return 10
    else:
        return 0


