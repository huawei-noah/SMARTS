import numpy as np
from gym import Env
from gym.spaces import Box

from rlkit.core.serializable import Serializable

EPS = np.finfo(np.float32).eps.item()


class ProxyEnv(Serializable, Env):
    def __init__(self, wrapped_env):
        self._wrapped_env = wrapped_env
        Serializable.quick_init(self, locals())
        super().__init__()
        self.action_space_n = self._wrapped_env.action_space_n
        self.observation_space_n = self._wrapped_env.observation_space_n

    def __getattr__(self, attrname):
        return getattr(self._wrapped_env, attrname)

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        # Add these explicitly in case they were modified
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)

    def reset(self, **kwargs):
        return self._wrapped_env.reset(**kwargs)

    def step(self, action_n):
        return self._wrapped_env.step(action_n)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    def log_diagnostics(self, paths, *args, **kwargs):
        if hasattr(self._wrapped_env, "log_diagnostics"):
            self._wrapped_env.log_diagnostics(paths, *args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        if hasattr(self._wrapped_env, "terminate"):
            self._wrapped_env.terminate()

    def seed(self, seed):
        return self._wrapped_env.seed(seed)


class NormalizedBoxActEnv(ProxyEnv, Serializable):
    """
    Normalize action to in [-1, 1].
    """

    def __init__(
        self,
        env,
    ):
        # self._wrapped_env needs to be called first because
        # Serializable.quick_init calls getattr, on this class. And the
        # implementation of getattr (see below) calls self._wrapped_env.
        # Without setting this first, the call to self._wrapped_env would call
        # getattr again (since it's not set yet) and therefore loop forever.
        self._wrapped_env = env
        # Or else serialization gets delegated to the wrapped_env. Serialize
        # this env separately from the wrapped_env.
        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)
        self.action_space_n = dict()
        for agent_id in self._wrapped_env.agent_ids:
            if isinstance(self._wrapped_env.action_space_n[agent_id], Box):
                ub = np.ones(self._wrapped_env.action_space_n[agent_id].shape)
                self.action_space_n[agent_id] = Box(-1 * ub, ub, dtype=np.float64)
            else:
                self.action_space_n[agent_id] = self._wrapped_env.action_space_n[
                    agent_id
                ]

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        # Add these explicitly in case they were modified
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)

    def step(self, action_n):
        scaled_action_n = {}
        for agent_id, action in action_n.items():
            if isinstance(self.action_space_n[agent_id], Box):
                action = np.clip(action, -1.0, 1.0)
                lb = self._wrapped_env.action_space_n[agent_id].low
                ub = self._wrapped_env.action_space_n[agent_id].high
                scaled_action = lb + (action + 1.0) * 0.5 * (ub - lb)
                # scaled_action = np.clip(scaled_action, lb, ub)
                scaled_action_n[agent_id] = scaled_action
            else:
                scaled_action_n[agent_id] = action

        return self._wrapped_env.step(scaled_action_n)

    def __str__(self):
        return "Normalized: %s" % self._wrapped_env

    def log_diagnostics(self, paths, **kwargs):
        if hasattr(self._wrapped_env, "log_diagnostics"):
            return self._wrapped_env.log_diagnostics(paths, **kwargs)
        else:
            return None

    def __getattr__(self, attrname):
        return getattr(self._wrapped_env, attrname)


class ObsScaledEnv(ProxyEnv, Serializable):
    """
    Scale the obs if desired
    Unscale the acts if desired
    """

    def __init__(
        self,
        env,
        obs_mean,
        obs_std,
    ):
        self._wrapped_env = env
        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)

        self.obs_mean = obs_mean
        self.obs_std = obs_std

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        # Add these explicitly in case they were modified
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)

    def get_unscaled_obs(self, obs):
        return obs * (self.obs_std + EPS) + self.obs_mean

    def get_scaled_obs(self, obs):
        return (obs - self.obs_mean) / (self.obs_std + EPS)

    def step(self, action_n):
        observation_n, reward_n, done_n, info_n = self._wrapped_env.step(action_n)
        for agent_id in observation_n.keys():
            observation_n[agent_id] = (observation_n[agent_id] - self.obs_mean) / (
                self.obs_std + EPS
            )
        return observation_n, reward_n, done_n, info_n

    def reset(self, **kwargs):
        observation_n = self._wrapped_env.reset(**kwargs)
        for agent_id in observation_n.keys():
            observation_n[agent_id] = (observation_n[agent_id] - self.obs_mean) / (
                self.obs_std + EPS
            )
        return observation_n
