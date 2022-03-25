# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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
import base64
import glob
from typing import Dict
import cv2
import numpy as np
from pathlib import Path
from itertools import groupby

from collections import defaultdict
from PIL import Image
from pathlib import Path

from smarts.core.utils.logging import isnotebook


def flatten_obs(sim_obs):
    obs = []
    for agent_id, agent_obs in sim_obs.items():
        if agent_obs is None:
            continue
        elif isinstance(agent_obs, dict):  # is_boid_agent
            for vehicle_id, vehicle_obs in agent_obs.items():
                obs.append((vehicle_id, vehicle_obs))
        else:
            obs.append((agent_id, agent_obs))
    return obs


def vis_sim_obs(sim_obs) -> Dict[str, np.ndarray]:
    vis_images = defaultdict(list)

    for agent_id, agent_obs in flatten_obs(sim_obs):
        # Visdom image format: Channel x Height x Width
        drivable_area = getattr(agent_obs, "drivable_area_grid_map", None)
        if drivable_area is not None:
            image = drivable_area.data
            image = image[:, :, [0, 0, 0]]
            image = image.astype(np.uint8)
            vis_images[f"{agent_id}-DrivableAreaGridMap"].append(image)

        ogm = getattr(agent_obs, "occupancy_grid_map", None)
        if ogm is not None:
            image: np.ndarray = ogm.data
            image = image[:, :, [0, 0, 0]]
            image = image.astype(np.uint8)
            vis_images[f"{agent_id}-OGM"].append(image)

        rgb = getattr(agent_obs, "top_down_rgb", None)
        if rgb is not None:
            image = rgb.data
            image = image.astype(np.uint8)
            vis_images[f"{agent_id}-Top-Down-RGB"].append(image)

    return {key: np.array(images) for key, images in vis_images.items()}


def write_image(sim_obs, frame_folder, tag_id):
    im_obs = vis_sim_obs(sim_obs)
    for im_id, im in im_obs.items():
        out = np.array(im)
        image_name = f"{frame_folder}/{im_id}_{tag_id}.JPG"
        cv2.imwrite(image_name, out[0])


def make_gif(frame_folder):
    group_key = lambda f: f.split("-")[-1].split("_")[0]
    image_file_groups = groupby(
        sorted(glob.glob(f"{frame_folder}/*.JPG"), key=group_key), key=group_key
    )
    image_file_groups = [list(imf) for g, imf in image_file_groups]
    sort_key = lambda im: int(Path(im).name.split("_")[-1].split(".")[0])
    gif_num = 0
    for image_files in image_file_groups:
        frames = []
        for image in sorted(image_files, key=sort_key):
            frames.append(Image.open(image))
        frame_one = frames[0]
        frame_one.save(
            f"{frame_folder}/im_{gif_num}.gif",
            format="GIF",
            append_images=frames,
            save_all=True,
            duration=100,
            loop=0,
        )
        gif_num += 1


def show_notebook_videos(path="videos", height="400px", split_html=""):
    if not isnotebook():
        return
    from IPython import display as ipythondisplay

    html = []
    for mp4 in Path(path).glob("*.mp4"):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append(
            """<video alt="{}" autoplay
                      loop controls style="height: {};">
                      <source src="data:video/mp4;base64,{}" type="video/mp4" />
                 </video>""".format(
                mp4, height, video_b64.decode("ascii")
            )
        )
    ipythondisplay.display(ipythondisplay.HTML(data=split_html.join(html)))
