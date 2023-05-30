import pathlib

import matplotlib
import numpy as np


def plotter3d(
    obs,
    rgb_gray: int = 3,
    channel_order: str = "first",
    name: str = "Plotter3D",
    pause: int = -1,
    save=False,
):
    """Plot images

    Args:
        obs (np.ndarray): Image in 3d, 4d, or 5d format.
            3d image format =(rgb*n_stack, height, width)
            4d image format =(batch, rgb*n_stack, height, width)
            5d image format =(batch, rgb, n_stack, height, width)
        rgb_gray (int, optional): 3 for rgb and 1 for grayscale. Defaults to 3.
        pause (int): Defaults to -1.
            negative int -> ignore plotting.
            zero         -> display plot and wait for user to close the image.
            positive int -> display plot and wait for `pause` seconds, then automatically close image.
        save (bool): If true, save image. Defaults to False.
    """

    import matplotlib.pyplot as plt

    dim = len(obs.shape)
    assert dim in [3, 4, 5]

    # print("PLOTTER3D: TYPE OF OBSERVATION OBJECT=",type(observation))
    # print("PLOTTER3D: OBSERVATION SHAPE INSIDE PLOTTER=",observation.shape)

    if isinstance(obs, np.ndarray):
        # print("PLOTTER3D: OBJECT IS ALREADY IN NUMPY FORMAT")
        obs = obs.copy()
    else:
        raise Exception("PLOTTER3D: INCORRECT FORMAT")

    # WARNING:matplotlib.image:Clipping input data to the valid range for
    # imshow with RGB data ([0..1] for floats or [0..255] for integers).
    if obs.dtype == np.float32 and np.amax(obs) > 1.0:
        # print("PLOTTER3D: CONVERTED IMAGE [0,255] float to [0,255] int")
        obs = (obs).astype(int)
    elif obs.dtype == np.float32 and np.amax(obs) <= 1.0:
        # print("PLOTTER3D: CONVERTED IMAGE [0,1] float to [0,255] int")
        obs = (obs * 255).astype(int)
    elif obs.dtype == int and np.amax(obs) > 1.0:
        # print("PLOTTER3D: OKAY IMAGE [0,255] int")
        pass
    elif obs.dtype == np.uint8 and np.amax(obs) <= 255:
        # print("PLOTTER3D: OKAY IMAGE [0,255] int")
        pass
    elif obs.dtype == np.int and np.amax(obs) <= 1.0:
        raise Exception("PLOTTER3D: ERROR IMAGE [0,1] int")
    else:
        # print(f"PLOTTER3D: ObsType: {obs.dtype}, max value {np.amax(obs)}")
        raise Exception("PLOTTER3D: UNKNOWN TYPE")

    if channel_order != "last" and channel_order != "first":
        raise Exception("PLOTTER3D: Unknown channel order")

    if dim == 3:
        if channel_order == "last":
            obs = np.moveaxis(obs, -1, 0)
        rows = 1
        columns = obs.shape[0] // rgb_gray
    elif dim == 4:
        if channel_order == "last":
            obs = np.moveaxis(obs, -1, 1)
        rows = obs.shape[0]
        columns = obs.shape[1] // rgb_gray
    else:  # dim==5
        if channel_order == "last":
            obs = np.moveaxis(obs, -1, 1)
        rows = obs.shape[0]
        columns = obs.shape[2]

    fig, axs = plt.subplots(nrows=rows, ncols=columns, squeeze=False, figsize=(20, 20))
    fig.set_size_inches(10, 4, forward=True)
    move_figure(fig, 0, 0)

    for row in range(0, rows):
        for col in range(0, columns):
            if dim == 3:
                img = obs[col * rgb_gray : col * rgb_gray + rgb_gray, :, :]
            elif dim == 4:
                img = obs[row, col * rgb_gray : col * rgb_gray + rgb_gray, :, :]
            else:  # dim == 5:
                img = obs[row, :, col, :, :]

            img = img.transpose(1, 2, 0)
            axs[row, col].imshow(img, cmap="viridis")
            axs[row, col].set_title(f"{name}")

    if save:
        save_path = pathlib.Path(__file__).resolve().parents[1] / "logs"
        plt.savefig(fname=save_path / "rgb.png", bbox_inches="tight")

    if pause >= 0:
        # plt.show()
        plt.pause(interval=pause)

    plt.close()


def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == "TkAgg":
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == "WXAgg":
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)


def load_pkl(file):
    import pickle

    from util import plotter3d

    file = open(
        "/home/adai/workspace/SMARTS/examples/output/Agent-history-vehicle-2884.pkl",
        "rb",
    )
    data = pickle.load(file)
    rgb = data[19.7].top_down_rgb.data
    plotter3d(rgb, pause=0, channel_order="last")
    rgb = data[19.7].occupancy_grid_map.data
    plotter3d(rgb, rgb_gray=1, pause=0, channel_order="last")
    rgb = data[19.7].drivable_area_grid_map.data
    plotter3d(rgb, rgb_gray=1, pause=0, channel_order="last")
