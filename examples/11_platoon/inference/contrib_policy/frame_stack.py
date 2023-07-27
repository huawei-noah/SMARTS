import copy
from collections import deque

import gymnasium as gym
import numpy as np


class FrameStack:
    """Stacks num_stack (default=3) consecutive frames, in a moving-window
    fashion, and returns the stacked_frames.

    Note:
        Wrapper returns a deep-copy of the stacked frames, which may be expensive
        for large frames and large num_stack values.
    """

    def __init__(self, input_space: gym.Space, num_stack: int = 3, stack_axis: int = 0):
        """
        Args:
            num_stack (int, optional): Number of frames to be stacked. Defaults to 3.
            stack_axis (int, optional): An int specifying the dimension over
                which to stack each observation.
        """
        assert num_stack > 1, f"Expected num_stack > 1, but got {num_stack}."
        self._num_stack = num_stack
        self._frames = deque(maxlen=self._num_stack)
        assert stack_axis >= 0 and stack_axis < len(input_space.shape)
        self._stack_axis = stack_axis

        dim_multiplier = np.ones_like(input_space.shape)
        dim_multiplier[stack_axis] = num_stack
        shape = dim_multiplier * input_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=shape,
            dtype=np.uint8,
        )

    def _stack(self, obs: np.ndarray) -> np.ndarray:
        """Update and return frames stack with given latest single frame."""

        self._frames.appendleft(obs)
        while len(self._frames) < self._num_stack:
            self._frames.appendleft(obs)
        frames_seq = tuple(self._frames)
        new_frames = copy.deepcopy(frames_seq)
        return np.concatenate(new_frames, axis=self._stack_axis)

    def stack(self, obs: np.ndarray) -> np.ndarray:
        """Stacks the latest obs with num_stack-1 past obs.

        Args:
            obs (np.ndarray): Numpy array input.

        Returns:
            np.ndarray: Stacked observation.
        """
        return self._stack(obs)

    def reset(self):
        """Resets the stacked obs."""
        self._frames.clear()
