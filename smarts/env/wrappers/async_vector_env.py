import multiprocessing as mp
import numpy as np
import time
import sys
import traceback

from enum import Enum
from gym import logger
import gym
from gym.error import (
    AlreadyPendingCallError,
    NoAsyncCallError,
    ClosedEnvironmentError,
)
from gym.vector.utils import concatenate
from smarts.env.wrappers.cloud_pickle import CloudpickleWrapper
from typing import Any, Callable, Dict, Sequence, Tuple

__all__ = ["AsyncVectorEnv"]


EnvConstructor = Callable[[], gym.Env]


class AsyncState(Enum):
    DEFAULT = "default"
    WAITING_RESET = "reset"
    WAITING_STEP = "step"


class AsyncComm(Enum):
    READY = 1
    RESET = 2
    STEP = 3
    CLOSE = 4
    EXCEPTION = 5


class AsyncVectorEnv(gym.vector.VectorEnv):
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
        seed=42,
    ):
        """The environments can be different but must use the same action and
        observation specs.

        Args:
            env_constructors (Sequence[EnvConstructor]): List of callables that create environments.
            auto_reset (bool, optional): Automatically resets an environment when episode ends. Defaults to True.
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

        # Observation space and action space
        dummy_env = env_constructors[0]()
        self.metadata = dummy_env.metadata
        observation_space = dummy_env.observation_space
        action_space = dummy_env.action_space
        dummy_env.close()
        del dummy_env

        super(AsyncVectorEnv, self).__init__(
            num_envs=len(env_constructors),
            observation_space=observation_space,
            action_space=action_space,
        )

        # Wait for all environments to successfully startup
        _, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        print("ENVIRONMENTS STARTUP FINISH ------------------------------------")

    # def seed(self, seeds=None):
    #     self._assert_is_running()
    #     if seeds is None:
    #         seeds = [None for _ in range(self.num_envs)]
    #     if isinstance(seeds, int):
    #         seeds = [seeds + i for i in range(self.num_envs)]
    #     assert len(seeds) == self.num_envs

    #     if self._state != AsyncState.DEFAULT:
    #         raise AlreadyPendingCallError(
    #             "Calling `seed` while waiting "
    #             "for a pending call to `{0}` to complete.".format(self._state.value),
    #             self._state.value,
    #         )

    #     for pipe, seed in zip(self.parent_pipes, seeds):
    #         pipe.send(("seed", seed))
    #     _, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
    #     self._raise_if_errors(successes)

    def reset_async(self):
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `reset_async` while waiting "
                "for a pending call to `{0}` to complete".format(self._state.value),
                self._state.value,
            )

        for pipe in self.parent_pipes:
            pipe.send(("reset", None))
        self._state = AsyncState.WAITING_RESET

    def reset_wait(self, timeout=None):
        """
        Parameters
        ----------
        timeout : int or float, optional
            Number of seconds before the call to `reset_wait` times out. If
            `None`, the call to `reset_wait` never times out.
        Returns
        -------
        observations : sample from `observation_space`
            A batch of observations from the vectorized environment.
        """
        self._assert_is_running()
        if self._state != AsyncState.WAITING_RESET:
            raise NoAsyncCallError(
                "Calling `reset_wait` without any prior " "call to `reset_async`.",
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

    def step_async(self, actions):
        """
        Parameters
        ----------
        actions : iterable of samples from `action_space`
            List of actions.
        """
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `step_async` while waiting "
                "for a pending call to `{0}` to complete.".format(self._state.value),
                self._state.value,
            )

        for pipe, action in zip(self.parent_pipes, actions):
            pipe.send(("step", action))
        self._state = AsyncState.WAITING_STEP

    def step_wait(
        self, timeout=None
    ) -> Tuple[
        Sequence[Dict[str, Any]],
        Sequence[Dict[str, float]],
        Sequence[Dict[str, bool]],
        Sequence[Dict[str, Any]],
    ]:
        """
        Parameters
        ----------
        timeout : int or float, optional
            Number of seconds before the call to `step_wait` times out. If
            `None`, the call to `step_wait` never times out.
        Returns
        -------
        observations : sample from `observation_space`
            A batch of observations from the vectorized environment.
        rewards : `np.ndarray` instance (dtype `np.float_`)
            A vector of rewards from the vectorized environment.
        dones : `np.ndarray` instance (dtype `np.bool_`)
            A vector whose entries indicate whether the episode has ended.
        infos : list of dict
            A list of auxiliary diagnostic information.
        """
        self._assert_is_running()
        if self._state != AsyncState.WAITING_STEP:
            raise NoAsyncCallError(
                "Calling `step_wait` without any prior call " "to `step_async`.",
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

    def close_extras(self, timeout=None, terminate=False):
        """
        Parameters
        ----------
        timeout : int or float, optional
            Number of seconds before the call to `close` times out. If `None`,
            the call to `close` never times out. If the call to `close` times
            out, then all processes are terminated.
        terminate : bool (default: `False`)
            If `True`, then the `close` operation is forced and all processes
            are terminated.
        """
        timeout = 0 if terminate else timeout
        try:
            if self._state != AsyncState.DEFAULT:
                logger.warn(
                    "Calling `close` while waiting for a pending "
                    "call to `{0}` to complete.".format(self._state.value)
                )
                function = getattr(self, "{0}_wait".format(self._state.value))
                function(timeout)
        except mp.TimeoutError:
            terminate = True

        if terminate:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
        else:
            for pipe in self.parent_pipes:
                if (pipe is not None) and (not pipe.closed):
                    pipe.send(("close", None))
            for pipe in self.parent_pipes:
                if (pipe is not None) and (not pipe.closed):
                    pipe.recv()

        for pipe in self.parent_pipes:
            if pipe is not None:
                pipe.close()
        for process in self.processes:
            process.join()

    def _poll(self, timeout=None):
        self._assert_is_running()
        if timeout is None:
            return True
        end_time = time.time() + timeout
        delta = None
        for pipe in self.parent_pipes:
            delta = max(end_time - time.time(), 0)
            if pipe is None:
                return False
            if pipe.closed or (not pipe.poll(delta)):
                return False
        return True

    # def _get_spaces(self):
    #     self._assert_is_running()
    #     for pipe in self.parent_pipes:
    #         pipe.send(("_get_spaces", self.single_observation_space))
    #     same_spaces, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
    #     self._raise_if_errors(successes)
    #     if not all(same_spaces):
    #         raise RuntimeError(
    #             "Some environments have an observation space "
    #             "different from `{0}`. In order to batch observations, the "
    #             "observation spaces from all environments must be "
    #             "equal.".format(self.single_observation_space)
    #         )

    def _assert_is_running(self):
        if self.closed:
            raise ClosedEnvironmentError(
                "Trying to operate on `{0}`, after a "
                "call to `close()`.".format(type(self).__name__)
            )

    def _raise_if_errors(self, successes):
        if all(successes):
            return

        num_errors = self.num_envs - sum(successes)
        assert num_errors > 0
        for _ in range(num_errors):
            index, exctype, value = self.error_queue.get()
            logger.error(
                "Received the following error from Worker-{0}: "
                "{1}: {2}".format(index, exctype.__name__, value)
            )
            logger.error("Shutting down Worker-{0}.".format(index))
            self.parent_pipes[index].close()
            self.parent_pipes[index] = None

        logger.error("Raising the last exception back to the main process.")
        raise exctype(value)


def _worker(
    index: int,
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

    try:
        # Construct the environment
        env = env_constructor()

        # Environment setup complete
        pipe.send((None, True))

        print(f"ENVIRONMENT >> {index} STARTUP FINISH ------------------------------------")

        while True:
            try:
                # Short block for keyboard interrupts
                if not pipe.poll(polling_period):
                    continue
                command, data = pipe.recv()
            except (EOFError, KeyboardInterrupt):
                break

            if command == "reset":
                observation = env.reset()
                pipe.send((observation, True))
            elif command == "step":
                observation, reward, done, info = env.step(data)
                if done["__all__"] and auto_reset:
                    print("AUTO RESETTING NOW")
                    observation = env.reset()
                pipe.send(((observation, reward, done, info), True))
            # elif command == "seed":
            #     env.seed(data)
            #     pipe.send((None, True))
            elif command == "close":
                env.close()
                pipe.send((None, True))
                break
            # elif command == "_get_spaces":
            #     pipe.send((data == env.observation_space, True))
            else:
                raise KeyError(f"Received unknown command `{command}`.")
    except Exception:
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()
        pipe.close()
