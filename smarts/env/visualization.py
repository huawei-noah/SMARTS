import json
import multiprocessing
import threading
from collections import defaultdict

import numpy as np

try:
    import visdom
except ImportError:
    raise ImportError("Visdom is required for visualizations.")


def build_visdom_watcher_queue():
    # Set queue size to 1 so that we don't hang on too old observations
    obs_queue = multiprocessing.Queue(1)

    queue_watcher = threading.Thread(
        target=_watcher_loop,
        args=(obs_queue,),
        daemon=True,  # If False, the proc will not terminate until this thread stops
    )

    queue_watcher.start()

    return obs_queue


def _watcher_loop(obs_queue):
    def _vis_sim_obs(sim_obs):
        vis_images = defaultdict(list)
        for agent_id, agent_obs in sim_obs.items():
            if agent_obs is None:
                continue

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

    def _vis_sim_text(sim_obs):
        vis_texts = defaultdict(list)
        for agent_id, agent_obs in sim_obs.items():
            if agent_obs is None:
                continue

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

    vis = visdom.Visdom()

    while True:
        obs = obs_queue.get()

        for key, images in _vis_sim_obs(obs).items():
            title = json.dumps({"Type:": key})
            images = np.stack(images, axis=0)
            vis.images(images, win=key, opts={"title": title})
            vis.images(images[0], win=key, opts={"title": title})

        for key, readings in _vis_sim_text(obs).items():
            title = json.dumps({"Type:": key})
            vis.text(readings, win=key, opts={"title": title})
