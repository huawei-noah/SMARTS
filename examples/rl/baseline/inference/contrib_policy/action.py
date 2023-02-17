# import gym
# import numpy as np


# class Action(gym.ActionWrapper):
#     """Modifies the action space."""

#     def __init__(self, env: gym.Env):
#         """Sets identical action space, denoted by `space`, for all agents.

#         Args:
#             env (gym.Env): Gym env to be wrapped.
#         """
#         super().__init__(env)
#         self._wrapper, action_space = _discrete()

#         self.action_space = gym.spaces.Dict(
#             {agent_id: action_space for agent_id in env.action_space.spaces.keys()}
#         )

#     def action(self, action):
#         """Adapts the action input to the wrapped environment.

#         `self.saved_obs` is retrieved from SaveObs wrapper. It contains previously
#         saved observation parameters.

#         Note: Users should not directly call this method.
#         """
#         wrapped_act = self._wrapper(action, self.saved_obs)

#         return wrapped_act

