import gym
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import sys

from enum import Enum
from gym import logger
from gym.error import (
    AlreadyPendingCallError,
    NoAsyncCallError,
    ClosedEnvironmentError,
)
from smarts.env.wrappers.cloud_pickle import CloudpickleWrapper
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

__all__ = ["AsyncVectorEnv"]


EnvConstructor = Callable[[], gym.Env]


class AsyncState(Enum):
    DEFAULT = "default"
    WAITING_RESET = "reset"
    WAITING_STEP = "step"


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
        sim_name: Optional[str] = None,
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
        self, timeout: Union[int, float, None] = None
    ) -> Tuple[
        Sequence[Dict[str, Any]],
        Sequence[Dict[str, float]],
        Sequence[Dict[str, bool]],
        Sequence[Dict[str, Any]],
    ]:
        """[summary]

        Args:
            timeout (Union[int, float, None], optional): Seconds to wait before timing out. 
                Defaults to None, and never times out.

        Raises:
            NoAsyncCallError: If `step_wait` is called without calling `step_async`.
            mp.TimeoutError: When 
            RuntimeError: [description]
            ClosedEnvironmentError: [description]
            exctype: [description]
            KeyError: [description]

        Returns:
            Tuple[ Sequence[Dict[str, Any]], Sequence[Dict[str, float]], Sequence[Dict[str, bool]], Sequence[Dict[str, Any]] ]: 
                Returns (observations, rewards, dones, infos). Each tuple element is a batch from the vectorized environment. 
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
