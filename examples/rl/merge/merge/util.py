import numpy as np


# def plotter3d(observation: Tensor, rgb_gray=3, name: str = "Plotter3D", block=True):
#     """Plot images

#     Args:
#         obs (np.ndarray): Image in 3d, 4d, or 5d format.
#             3d image format =(rgb*n_stack, height, width)
#             4d image format =(batch, rgb*n_stack, height, width)
#             5d image format =(batch, rgb, n_stack, height, width)
#         rgb_gray (int, optional): 3 for rgb and 1 for grayscale. Defaults to 1.
#     """

#     import matplotlib.pyplot as plt

#     dim = len(observation.shape)
#     assert dim in [3, 4, 5]

#     # print("PLOTTER3D: TYPE OF OBSERVATION OBJECT=",type(observation))
#     # print("PLOTTER3D: OBSERVATION SHAPE INSIDE PLOTTER=",observation.shape)

#     if th.is_tensor(observation):
#         # print("PLOTTER3D: CONVERTED TENSOR TO NUMPY")
#         obs = observation.detach().cpu().numpy()
#     elif isinstance(observation, np.ndarray):
#         # print("PLOTTER3D: OBJECT IS ALREADY IN NUMPY FORMAT")
#         obs = observation.copy()
#     else:
#         raise Exception("PLOTTER3D: INCORRECT FORMAT")

#     # WARNING:matplotlib.image:Clipping input data to the valid range for
#     # imshow with RGB data ([0..1] for floats or [0..255] for integers).
#     if obs.dtype == np.float32 and np.amax(obs) > 1.0:
#         # print("PLOTTER3D: CONVERTED IMAGE [0,255] float tO [0,255] int")
#         obs = (obs).astype(int)
#     elif obs.dtype == np.float32 and np.amax(obs) <= 1.0:
#         # print("PLOTTER3D: CONVERTED IMAGE [0,1] float to [0,255] int")
#         obs = (obs * 255).astype(int)
#     elif obs.dtype == np.int and np.amax(obs) > 1.0:
#         # print("PLOTTER3D: OKAY IMAGE [0,255] int")
#         pass
#     elif obs.dtype == np.int and np.amax(obs) <= 1.0:
#         raise Exception("PLOTTER3D: ERROR IMAGE [0,1] int")
#     else:
#         print(f"PLOTTER3D: ObsType: {obs.dtype}, max value {np.amax(obs)}")
#         raise Exception("PLOTTER3D: UNKNOWN TYPE")

#     if dim == 3:
#         rows = 1
#         columns = obs.shape[0] // rgb_gray
#     elif dim == 4:
#         rows = obs.shape[0]
#         columns = obs.shape[1] // rgb_gray
#     else:  # dim==5
#         rows = obs.shape[0]
#         columns = obs.shape[2]

#     fig, axs = plt.subplots(nrows=rows, ncols=columns, squeeze=False, figsize=(10, 10))
#     fig.suptitle("PLOTTER3D")

#     for row in range(0, rows):
#         for col in range(0, columns):
#             if dim == 3:
#                 img = obs[col * rgb_gray : col * rgb_gray + rgb_gray, :, :]
#             elif dim == 4:
#                 img = obs[row, col * rgb_gray : col * rgb_gray + rgb_gray, :, :]
#             elif dim == 5:
#                 img = obs[row, :, col, :, :]
#             else:
#                 raise Exception("NO SUCH DIMENSION")

#             img = img.transpose(1, 2, 0)
#             axs[row, col].imshow(img, cmap="viridis")
#             axs[row, col].set_title(f"{name}")

#     plt.show(block=block)
#     plt.pause(2)
#     plt.close()
