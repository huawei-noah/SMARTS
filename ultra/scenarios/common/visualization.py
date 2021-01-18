# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import cv2, os, re
import numpy as np
from matplotlib import pyplot as plt
import imageio
from collections import defaultdict
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from .social_vehicle_definitions import get_social_vehicle_color

# import geometry as geometry


def convert_to_gif(images, save_dir, name):
    print("Processing images...")
    imageio.mimsave(f"{save_dir}/{name}.gif", images)
    print(f"Saved {save_dir}/{name}.gif")


def convert_to_mov(images, save_dir):
    height, width, layers = images[0].shape
    video_name = f"{save_dir}/movie.mov"
    video = cv2.VideoWriter(video_name, 0, 1, (width, height))
    for image in images:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video.write(bgr)


def fig2data(fig_canvas):
    # Retrieve a view on the renderer buffer
    fig_canvas.draw()
    buf = fig_canvas.buffer_rgba()
    # convert to a NumPy array
    image = np.asarray(buf)
    return image


def profile_vehicles(vehicle_states, save_dir):
    def plot(ax, x_label, y_label, state):
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.plot(range(len(state[y_label])), state[y_label])
        if "in_junction" in state:
            ax.axvspan(
                state["in_junction"][0], state["in_junction"][1], color="red", alpha=0.5
            )

    for v_id, state in vehicle_states.items():
        if not state["teleported"]:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
            plot(ax[0], "time", "speed", state)
            plot(ax[1], "time", "accel", state)
            behavior_key, behavior_color = get_social_vehicle_color(state["behavior"])
            temp_save_dir = f"{save_dir}/{behavior_key}"
            if not os.path.exists(temp_save_dir):
                os.makedirs(temp_save_dir)
            plt.savefig(f"{temp_save_dir}/{state['route']}_{v_id}.png")

    plt.close("all")


def draw_intersection(
    ego,
    goal_path,
    all_waypoints,
    step,
    goal,
    start,
    social_vehicle_states,
    finished_vehicles,
    lookaheads,
    intersection_tag="t",
):
    fig_w = 812
    fig_h = 812
    canvas = np.zeros((fig_w, fig_h, 3), np.uint8)

    if intersection_tag == "c":
        fig_offset_y = 320
        fig_offset_x = 100
    else:
        fig_offset_y = 700
        fig_offset_x = 300
    fig_offset_y = 100
    fig_offset_x = 100
    fig_mul = 2
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(20, 10))
    for point in all_waypoints:
        canvas = cv2.circle(
            canvas,
            (
                fig_offset_x + int(point[0] * fig_mul),
                fig_offset_y - int(point[1] * fig_mul),
            ),
            radius=0,
            color=(255, 255, 255),
            thickness=1,
        )
    for point in lookaheads:
        canvas = cv2.circle(
            canvas,
            (
                fig_offset_x + int(point[0] * fig_mul),
                fig_offset_y - int(point[1] * fig_mul),
            ),
            radius=0,
            color=(255, 255, 0),
            thickness=1,
        )
    ego_color = (0, 200, 0)
    colors_legend = set()
    colors_legend.add(("ego", ego_color))
    font = cv2.FONT_HERSHEY_DUPLEX

    for state in social_vehicle_states:  # .items():
        # print(state['behavior'])
        # if v_id not in finished_vehicles:
        # behavior_key, behavior_color = get_social_vehicle_color(state["behavior"])
        pos_x = fig_offset_x + int(state.position[0] * fig_mul)
        pos_y = fig_offset_y - int(state.position[1] * fig_mul)
        canvas = cv2.circle(
            canvas, (pos_x, pos_y,), radius=1, color=(10, 10, 30), thickness=2,
        )
        # canvas = cv2.putText(
        #     canvas, str(v_id), (pos_x + 4, pos_y + 4), font, 0.3, behavior_color, 1,
        # )
        # colors_legend.add((behavior_key, behavior_color))
    # print(ego)
    # if ego:
    canvas = cv2.circle(
        canvas,
        (fig_offset_x + int(ego[0] * fig_mul), fig_offset_y - int(ego[1] * fig_mul),),
        radius=1,
        color=ego_color,
        thickness=2,
    )

    canvas = cv2.putText(
        canvas,
        "x",
        (
            fig_offset_x + int(goal[0] * fig_mul) - 4,
            fig_offset_y - int(goal[1] * fig_mul) + 8,
        ),
        font,
        0.5,
        (200, 150, 255),
        1,
    )
    canvas = cv2.putText(
        canvas,
        "x",
        (
            fig_offset_x + int(start[0] * fig_mul) - 4,
            fig_offset_y - int(start[1] * fig_mul) + 8,
        ),
        font,
        0.5,
        (1, 255, 255),
        1,
    )

    # color_offset = 20
    # legend_position = (30, 50)
    #
    # for color_key, color in colors_legend:
    #     canvas = cv2.putText(
    #         canvas,
    #         color_key,
    #         (legend_position[0], legend_position[1] + color_offset),
    #         font,
    #         0.5,
    #         color,
    #         1,
    #     )
    #     color_offset += 20
    cv2.imwrite(f"temp/{step}.png", canvas)
    plt.close("all")
    return canvas


# def visualize_social_safety(ax, all_waypoints, nighbor_bounding_box):
#     all_waypoints = np.asarray(all_waypoints)

#     ax.clear()
#     ax.set_axis_off()
#     ax.view_init(60, 60)
#     ax.scatter(
#         all_waypoints[:, 0],
#         all_waypoints[:, 1],
#         -5.0,
#         s=100.0,
#         marker=".",
#         color="blue",
#         alpha=0.1,
#     )

#     ax = geometry.visualize_boxes(
#         ax,
#         nighbor_bounding_box,
#         ["o" for box in nighbor_bounding_box],
#         [
#             "r" if i == 0 else "g" if i == 1 else "b"
#             for i, box in enumerate(nighbor_bounding_box)
#         ],
#         [
#             "cyan" if i == 0 or i == 1 else "pink"
#             for i, box in enumerate(nighbor_bounding_box)
#         ],
#     )
#     plt.draw()
#     plt.pause(0.00001)
