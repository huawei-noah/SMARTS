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

import gym
import multiprocessing as mp
import numpy as np
import sys

from gym.vector.async_vector_env import AsyncState, AsyncVectorEnv
from gym.error import (
    AlreadyPendingCallError,
    NoAsyncCallError,
)
from smarts.env.wrappers.cloud_pickle import CloudpickleWrapper
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

__all__ = ["ParallelEnv"]


EnvConstructor = Callable[[str], gym.Env]


class ParallelEnv(AsyncVectorEnv):
    """Batch together multiple environments and step them in parallel. Each
    environment is simulated in an external process for lock-free parallelism
    using `multiprocessing` processes, and pipes for communication.

    Note:
        Number of environments should not exceed the number of available CPU
        logical cores.
    """

    def __init__(
        self,
        env_constructors: Sequence[EnvConstructor],
        auto_reset: bool,
        sim_name: Optional[str] = None,
        seed=42,
    ):
        """The environments can be different but must use the same action and
        observation specs.

        Args:
            env_constructors (Sequence[EnvConstructor]): List of callables that create environments.
            auto_reset (bool): Automatically resets an environment when episode ends. Defaults to True.
            sim_name (Optional[str], optional): Simulation name prefix. Defaults to None.
            seed (int, optional): Seed for the first environment. Defaults to 42.

        Raises:
            TypeError: If any environment constructor is not callable.
        """

        if any([not callable(ctor) for ctor in env_constructors]):
            raise TypeError(
                "Found non-callable `env_constructors`. Expected `env_constructors` of type `Sequence[Callable[[], gym.Env]]`,"
                "but got {}".format(env_constructors)
            )

        # Worker polling period in seconds.
        self._polling_period = 0.1

        mp_ctx = mp.get_context()
        self.env_constructors = env_constructors

        self.error_queue = mp_ctx.Queue()
        self.parent_pipes = []
        self.processes = []
        for idx, env_constructor in enumerate(self.env_constructors):
            parent_pipe, child_pipe = mp_ctx.Pipe()
            process = mp_ctx.Process(
                target=_worker,
                name=f"Worker<{type(self).__name__}>-<{idx}>",
                args=(
                    idx,
                    sim_name,
                    CloudpickleWrapper(env_constructor),
                    auto_reset,
                    child_pipe,
                    self.error_queue,
                    self._polling_period,
                ),
            )
            self.parent_pipes.append(parent_pipe)
            self.processes.append(process)

            # Daemonic subprocesses quit when parent process quits. However, daemonic
            # processes cannot spawn children. Hence, `process.daemon` is set to False.
            process.daemon = False
            process.start()
            child_pipe.close()

        self._state = AsyncState.DEFAULT

        # Wait for all environments to successfully startup
        _, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)

        # Get and check observation and action spaces
        observation_space, action_space = self._get_spaces()

        # TODO: This dummy `observation_space` and `action_space` should be removed after they
        # are properly specified in SMARTS/smarts/env/hiway_env.py:__init__() function.
        action_space = gym.spaces.Dict(
            {
                agent_id: gym.spaces.Box(
                    np.array([0, 0, -1]), np.array([+1, +1, +1]), dtype=np.float32
                )  # throttle, break, steering
                for agent_id in ["Agent1", "Agent2"]
            }
        )
        observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(256, 256, 3), dtype=np.float32
        )

        super(AsyncVectorEnv, self).__init__(
            num_envs=len(env_constructors),
            observation_space=observation_space,
            action_space=action_space,
        )

        # Seed all the environment
        self.seed(seed)

    def seed(self, seed: int):
        """Sets unique seed for each environment.

        Args:
            seed (int): Seed number.

        Raises:
            AlreadyPendingCallError: If `seed` is called while another function call is pending.
        """
        self._assert_is_running()
        seeds = [seed + i for i in range(self.num_envs)]
        assert len(seeds) == self.num_envs

        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `seed` while waiting "
                "for a pending call to `{0}` to complete.".format(self._state.value),
                self._state.value,
            )

        for pipe, seed in zip(self.parent_pipes, seeds):
            pipe.send(("seed", seed))
        _, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)

    def reset_wait(
        self, timeout: Union[int, float, None] = None
    ) -> Sequence[Dict[str, Any]]:
        """Waits for all environments to reset.

        Args:
            timeout (Union[int, float, None], optional): Seconds to wait before timing out.
                Defaults to None, and never times out.

        Raises:
            NoAsyncCallError: If `reset_wait` is called without calling `reset_async`.
            mp.TimeoutError: If response is not received from pipe within `timeout` seconds.

        Returns:
            Sequence[Dict[str, Any]]: A batch of observations from the vetorized environment.
        """

        self._assert_is_running()
        if self._state != AsyncState.WAITING_RESET:
            raise NoAsyncCallError(
                "Calling `reset_wait` without any prior call to `reset_async`.",
                AsyncState.WAITING_RESET.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                "The call to `reset_wait` has timed out after "
                "{0} second{1}.".format(timeout, "s" if timeout > 1 else "")
            )

        observations, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        return observations

    def step_wait(
        self, timeout: Union[int, float, None] = None
    ) -> Tuple[
        Sequence[Dict[str, Any]],
        Sequence[Dict[str, float]],
        Sequence[Dict[str, bool]],
        Sequence[Dict[str, Any]],
    ]:
        """Waits and returns batched (observations, rewards, dones, infos) from all environments after a single step.

        Args:
            timeout (Union[int, float, None], optional): Seconds to wait before timing out.
                Defaults to None, and never times out.

        Raises:
            NoAsyncCallError: If `step_wait` is called without calling `step_async`.
            mp.TimeoutError: If data is not received from pipe within `timeout` seconds.

        Returns:
            Tuple[ Sequence[Dict[str, Any]], Sequence[Dict[str, float]], Sequence[Dict[str, bool]], Sequence[Dict[str, Any]] ]:
                Returns (observations, rewards, dones, infos). Each tuple element is a batch from the vectorized environment.
        """

        self._assert_is_running()
        if self._state != AsyncState.WAITING_STEP:
            raise NoAsyncCallError(
                "Calling `step_wait` without any prior call to `step_async`.",
                AsyncState.WAITING_STEP.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                "The call to `step_wait` has timed out after "
                "{0} second{1}.".format(timeout, "s" if timeout > 1 else "")
            )

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT
        observations_list, rewards, dones, infos = zip(*results)

        return (
            observations_list,
            rewards,
            dones,
            infos,
        )

    def _get_spaces(self) -> Tuple[gym.Space, gym.Space]:
        for pipe in self.parent_pipes:
            pipe.send(("_get_spaces", None))
        spaces, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)

        observation_space = spaces[0][0]
        action_space = spaces[0][1]

        if not all([space[0] == observation_space for space in spaces]) or not all(
            [space[1] == action_space for space in spaces]
        ):
            raise RuntimeError(
                f"Expected all environments to have the same observation and action"
                f"spaces but got {spaces}."
            )

        return observation_space, action_space


def _worker(
    index: int,
    sim_name: str,
    env_constructor: CloudpickleWrapper,
    auto_reset: bool,
    pipe: mp.connection.Connection,
    error_queue: mp.Queue,
    polling_period: float = 0.1,
):
    """Process to build and run an environment. Using a pipe to
    communicate with parent, the process receives action, steps
    the environment, and returns the observations.

    Args:
        index (int): Environment index number.
        env_constructor (CloudpickleWrapper): Callable which constructs the environment.
        auto_reset (bool): If True, auto resets environment when episode ends.
        pipe (mp.connection.Connection): Child's end of the pipe.
        error_queue (mp.Queue): Queue to communicate error messages.
        polling_period (float): Time to wait for keyboard interrupts.
    """

    # Name and construct the environment
    name = f"env_{index}"
    if sim_name:
        name = sim_name + "_" + name
    env = env_constructor(sim_name=name)

    # Environment setup complete
    pipe.send((None, True))

    try:
        while True:
            # Short block for keyboard interrupts
            if not pipe.poll(polling_period):
                continue
            command, data = pipe.recv()
            if command == "reset":
                observation = env.reset()
                pipe.send((observation, True))
            elif command == "step":
                observation, reward, done, info = env.step(data)
                if done["__all__"] and auto_reset:
                    # Actual final observations can be obtained from `info`:
                    # ```
                    # final_obs = info[agent_id]["env_obs"]
                    # ```
                    observation = env.reset()
                pipe.send(((observation, reward, done, info), True))
            elif command == "seed":
                env_seed = env.seed(data)
                pipe.send((env_seed, True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_get_spaces":
                pipe.send(((env.observation_space, env.action_space), True))
            else:
                raise KeyError(f"Received unknown command `{command}`.")
    except Exception:
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()
        pipe.close()
