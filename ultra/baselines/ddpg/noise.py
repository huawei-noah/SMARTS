"""
source code: https://github.com/ShangtongZhang/DeepRL/deep_rl/component/random_process.py
"""
import numpy as np


class LinearSchedule:
    def __init__(self, start, end=None, steps=None):
        if end is None:
            end = start
            steps = 1
        self.inc = (end - start) / float(steps)
        self.current = start
        self.end = end
        if end > start:
            self.bound = min
        else:
            self.bound = max

    def __call__(self, steps=1):
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)
        return val


class OrnsteinUhlenbeckProcess:
    def __init__(self, size, std, mu, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.std = std
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            + self.std() * np.sqrt(self.dt) * np.random.randn(*self.size)
        )
        self.x_prev = x
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)
