# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
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
import json
import logging
import multiprocessing
import threading
from collections import defaultdict

import numpy as np

try:
    import visdom
except ImportError:
    raise ImportError("Visdom is required for visualizations.")


class VisdomClient:
    def __init__(self, hostname="http://localhost", port=8097):
        self._log = logging.getLogger(self.__class__.__name__)
        self._port = port
        self._hostname = hostname
        self._visdom_obs_queue = self._build_visdom_watcher_queue()

    def send(self, obs):
        try:
            self._visdom_obs_queue.put(obs, block=False)
        except Exception:
            self._log.debug("Dropped visdom frame instead of blocking")

    def teardown(self):
        pass

    def _build_visdom_watcher_queue(self):
        # Set queue size to 1 so that we don't hang on too old observations
        obs_queue = multiprocessing.Queue(1)

        queue_watcher = threading.Thread(
            target=self._watcher_loop,
            args=(obs_queue,),
            daemon=True,  # If False, the proc will not terminate until this thread stops
        )

        queue_watcher.start()
        return obs_queue

    def _watcher_loop(self, obs_queue):
        vis = visdom.Visdom(port=self._port, server=self._hostname)

        while True:
            obs = obs_queue.get()

            for key, images in self._vis_sim_obs(obs).items():
                title = json.dumps({"Type:": key})
                images = np.stack(images, axis=0)
                vis.images(images, win=key, opts={"title": title})
                vis.images(images[0], win=key, opts={"title": title})

            for key, readings in self._vis_sim_text(obs).items():
                title = json.dumps({"Type:": key})
                vis.text(readings, win=key, opts={"title": title})

    def _vis_sim_obs(self, sim_obs):
        vis_images = defaultdict(list)

        for agent_id, agent_obs in self._flatten_obs(sim_obs):
            # Visdom image format: Channel x Height x Width
            drivable_area = getattr(agent_obs, "drivable_area_grid_map", None)
            if drivable_area is not None:
                image = drivable_area.data
                image = image.transpose(2, 0, 1)
                image = image.astype(np.float32)
                vis_images[f"{agent_id}-DrivableAreaGridMap"].append(image)

            ogm = getattr(agent_obs, "occupancy_grid_map", None)
            if ogm is not None:
                image = ogm.data
                image = image.reshape(image.shape[0], image.shape[1])
                image = np.expand_dims(image, axis=0)
                image = (image.astype(np.float32) / 100) * 255
                vis_images[f"{agent_id}-OGM"].append(image)

            rgb = getattr(agent_obs, "top_down_rgb", None)
            if rgb is not None:
                image = rgb.data
                image = image.transpose(2, 0, 1)
                image = image.astype(np.float32)
                vis_images[f"{agent_id}-Top-Down RGB"].append(image)

        return vis_images

    def _vis_sim_text(self, sim_obs):
        vis_texts = defaultdict(list)

        for agent_id, agent_obs in self._flatten_obs(sim_obs):
            vehicle_state = getattr(agent_obs, "ego_vehicle_state", None)
            throttle = getattr(vehicle_state, "throttle", None)
            if throttle is not None:
                vis_texts[f"{agent_id}-throttle"].append(throttle)

            vehicle_state = getattr(agent_obs, "ego_vehicle_state", None)
            brake = getattr(vehicle_state, "brake", None)
            if brake is not None:
                vis_texts[f"{agent_id}-brake"].append(brake)

            vehicle_state = getattr(agent_obs, "ego_vehicle_state", None)
            steering = getattr(vehicle_state, "steering", None)
            if steering is not None:
                vis_texts[f"{agent_id}-steering"].append(steering)

        return vis_texts

    def _flatten_obs(self, sim_obs):
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
