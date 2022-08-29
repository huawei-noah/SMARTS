import math
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Set


def goal_region_reward(threshold, goal_x, goal_y, cur_x, cur_y):
    eucl_distance = math.sqrt((goal_x - cur_x) ** 2 + (goal_y - cur_y) ** 2)

    if eucl_distance <= threshold:
        return 10
    else:
        return 0


def inside_coor_to_pixel(goal_x, goal_y, cur_x, cur_y):
    ratio = 256 / 50  # 256 pixels corresonds to 50 meters
    x_diff = abs(goal_x - cur_x)
    y_diff = abs(goal_y - cur_y)

    # find true condition of first quadrant
    if goal_x > cur_x and goal_y > cur_y:
        x_pixel_loc = min(
            128 + round(x_diff * ratio), 255
        )  # cap on 256 which is the right edge
        y_pixel_loc = max(
            127 - round(y_diff * ratio), 0
        )  # cap on 0 which is the upper edge

    # find second quadrant
    elif goal_x < cur_x and goal_y > cur_y:
        x_pixel_loc = max(
            127 - round(x_diff * ratio), 0
        )  # cap on 0 which is the left edge
        y_pixel_loc = max(
            127 - round(y_diff * ratio), 0
        )  # cap on 0 which is the upper edge

    # To find third quadrant
    elif goal_x < cur_x and goal_y < cur_y:
        x_pixel_loc = max(
            127 - round(x_diff * ratio), 0
        )  # cap on 0 which is the left edge
        y_pixel_loc = min(
            128 + round(y_diff * ratio), 255
        )  # cap on 256 which is the bottom edge

    # To find Fourth quadrant
    elif goal_x > cur_x and goal_y < cur_y:
        x_pixel_loc = min(
            128 + round(x_diff * ratio), 255
        )  # cap on 256 which is the right edge
        y_pixel_loc = min(
            128 + round(y_diff * ratio), 255
        )  # cap on 256 which is the bottom edge

    # To find if goal is at cur (do not change to elif)
    if (abs(cur_x) - 0.05 <= abs(goal_x) <= abs(cur_x) + 0.05) and (
        abs(cur_y) - 0.05 <= abs(goal_y) <= abs(cur_y) + 0.05
    ):
        x_pixel_loc = 128
        y_pixel_loc = 128

    # On x-axis
    elif (abs(cur_y) - 0.05 <= abs(goal_y) <= abs(cur_y) + 0.05) and goal_x != cur_x:
        if goal_x >= cur_x:
            x_pixel_loc = min(128 + round(x_diff * ratio), 255)
        else:
            x_pixel_loc = max(127 - round(x_diff * ratio), 0)
        y_pixel_loc = min(128 + round(y_diff * ratio), 255)

    # On y-axis
    elif (abs(cur_x) - 0.05 <= abs(goal_x) <= abs(cur_x) + 0.05) and goal_y != cur_y:
        if goal_y >= cur_y:
            y_pixel_loc = max(127 - round(y_diff * ratio), 0)
        else:
            y_pixel_loc = min(128 + round(y_diff * ratio), 255)
        x_pixel_loc = min(128 + round(x_diff * ratio), 255)

    goal_obs = np.zeros((1, 256, 256))
    goal_obs[0, y_pixel_loc, x_pixel_loc] = 255
    return goal_obs


def outside_coor_to_pixel(goal_x, goal_y, cur_x, cur_y):
    ratio = 256 / 50  # 256 pixels corresonds to 25 meters
    x_diff = abs(goal_x - cur_x)
    y_diff = abs(goal_y - cur_y)

    # find true condition of first quadrant
    if goal_x > cur_x and goal_y > cur_y:
        theta = math.atan(y_diff / x_diff)
        if 0 < theta < (math.pi / 4):
            x_pixel_loc = 255
            y_pixel_loc = max(127 - round((25 * (y_diff / x_diff)) * ratio), 0)
        elif (math.pi / 4) < theta < (math.pi / 2):
            x_pixel_loc = min(128 + round((25 / (y_diff / x_diff)) * ratio), 255)
            y_pixel_loc = 0
        elif theta == (math.pi / 4):
            x_pixel_loc = 255
            y_pixel_loc = 0

    # find second quadrant
    elif goal_x < cur_x and goal_y > cur_y:
        theta = math.atan(y_diff / x_diff)
        if 0 < theta < (math.pi / 4):
            x_pixel_loc = 0
            y_pixel_loc = max(127 - round((25 * (y_diff / x_diff)) * ratio), 0)
        elif (math.pi / 4) < theta < (math.pi / 2):
            x_pixel_loc = max(127 - round((25 / (y_diff / x_diff)) * ratio), 0)
            y_pixel_loc = 0
        elif theta == (math.pi / 4):
            x_pixel_loc = 0
            y_pixel_loc = 0

    # To find third quadrant
    elif goal_x < cur_x and goal_y < cur_y:
        theta = math.atan(y_diff / x_diff)
        if 0 < theta < (math.pi / 4):
            x_pixel_loc = 0
            y_pixel_loc = min(128 + round((25 * (y_diff / x_diff)) * ratio), 255)
        elif (math.pi / 4) < theta < (math.pi / 2):
            x_pixel_loc = max(127 - round((25 / (y_diff / x_diff)) * ratio), 0)
            y_pixel_loc = 255
        elif theta == (math.pi / 4):
            x_pixel_loc = 0
            y_pixel_loc = 255

    # To find Fourth quadrant
    elif goal_x > cur_x and goal_y < cur_y:
        theta = math.atan(y_diff / x_diff)
        if 0 < theta < (math.pi / 4):
            x_pixel_loc = 255
            y_pixel_loc = min(128 + round((25 * (y_diff / x_diff)) * ratio), 255)
        elif (math.pi / 4) < theta < (math.pi / 2):
            x_pixel_loc = min(128 + round((25 / (y_diff / x_diff)) * ratio), 255)
            y_pixel_loc = 255
        elif theta == (math.pi / 4):
            x_pixel_loc = 255
            y_pixel_loc = 255

    # On x-axis (do not change to elif)
    if (abs(cur_y) - 0.05 <= abs(goal_y) <= abs(cur_y) + 0.05) and goal_x != cur_x:
        if goal_x >= cur_x:
            x_pixel_loc = 255
        else:
            x_pixel_loc = 0
        y_pixel_loc = 128

    # On y-axis
    elif (abs(cur_x) - 0.05 <= abs(goal_x) <= abs(cur_x) + 0.05) and goal_y != cur_y:
        if goal_y >= cur_y:
            y_pixel_loc = 0
        else:
            y_pixel_loc = 255
        x_pixel_loc = 128

    goal_obs = np.zeros((1, 256, 256))
    goal_obs[0, y_pixel_loc, x_pixel_loc] = 255
    return goal_obs


def get_trans_coor(goal_x, goal_y, cur_x, cur_y, cur_heading):

    if 0 < cur_heading < math.pi:  # Facing Left Half
        theta = cur_heading

    elif -(math.pi) < cur_heading < 0:  # Facing Right Half
        theta = 2 * math.pi + cur_heading

    elif cur_heading == 0:  # Facing up North
        theta = 0

    elif (cur_heading == math.pi) or (cur_heading == -(math.pi)):  # Facing South
        theta = 2 * math.pi + cur_heading

    trans_matrix = np.array(
        [[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]]
    )
    cur_pos = np.array([[cur_x], [cur_y]])
    goal_pos = np.array([[goal_x], [goal_y]])
    trans_cur = np.round(np.matmul(trans_matrix, cur_pos), 5)
    trans_goal = np.round(np.matmul(trans_matrix, goal_pos), 5)

    return [trans_cur, trans_goal]


# mark goal position with integer 256, other entries are all 0
def get_goal_layer(goal_x, goal_y, cur_x, cur_y, cur_heading):

    trans_coor = get_trans_coor(goal_x, goal_y, cur_x, cur_y, cur_heading)
    trans_cur = trans_coor[0]
    trans_goal = trans_coor[1]

    if (trans_cur[0, 0] - 25) <= trans_goal[0, 0] <= (trans_cur[0, 0] + 25):
        if (trans_cur[1, 0] - 25) <= trans_goal[1, 0] <= (trans_cur[1, 0] + 25):
            inside = True
        else:
            inside = False
    else:
        inside = False

    if inside:
        goal_obs = inside_coor_to_pixel(
            trans_goal[0, 0], trans_goal[1, 0], trans_cur[0, 0], trans_cur[1, 0]
        )
    else:
        goal_obs = outside_coor_to_pixel(
            trans_goal[0, 0], trans_goal[1, 0], trans_cur[0, 0], trans_cur[1, 0]
        )

    return goal_obs


def global_target_pose(action, agent_obs):

    cur_x = agent_obs["ego"]["pos"][0]
    cur_y = agent_obs["ego"]["pos"][1]
    cur_heading = agent_obs["ego"]["heading"]

    if 0 < cur_heading < math.pi:  # Facing Left Half
        theta = cur_heading

    elif -(math.pi) < cur_heading < 0:  # Facing Right Half
        theta = 2 * math.pi + cur_heading

    elif cur_heading == 0:  # Facing up North
        theta = 0

    elif (cur_heading == math.pi) or (cur_heading == -(math.pi)):  # Facing South
        theta = 2 * math.pi + cur_heading

    trans_matrix = np.array(
        [[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]]
    )
    action_bev = np.array([[action[0]], [action[1]]])
    action_global = np.matmul(np.transpose(trans_matrix), action_bev)
    target_pose = np.array(
        [
            cur_x + action_global[0],
            cur_y + action_global[1],
            action[2] + cur_heading,
            0.1,
        ],
        dtype=object,
    )

    return target_pose
