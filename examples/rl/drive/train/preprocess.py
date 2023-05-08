import gym
from contrib_policy.filter_obs import FilterObs
from contrib_policy.format_action import FormatAction
from contrib_policy.frame_stack import FrameStack
from contrib_policy.make_dict import MakeDict

from smarts.core.agent_interface import AgentInterface


class Preprocess(gym.Wrapper):
    def __init__(self, env: gym.Env, agent_interface: AgentInterface):
        super().__init__(env)

        self._filter_obs = FilterObs(top_down_rgb=agent_interface.top_down_rgb)
        self._frame_stack = FrameStack(
            input_space=self._filter_obs.observation_space,
            num_stack=3,
            stack_axis=0,
        )
        self._frame_stack.reset()
        self._make_dict = MakeDict(input_space=self._frame_stack.observation_space)

        self.observation_space = self._make_dict.observation_space

        self._prev_heading: float
        self._format_action = FormatAction(agent_interface.action)
        self.action_space = self._format_action.action_space

    def _process(self, obs):
        # if obs["ego_vehicle_state"]["position"][0] > 190:
        #     plot = True
        # else:
        #     plot = False

        obs = self._filter_obs.filter(obs)
        obs = self._frame_stack.stack(obs)
        obs = self._make_dict.make(obs)

        # if plot:
        #     from contrib_policy.helper import plotter3d
        #     print("-----------------------------")
        #     plotter3d(obs=obs["rgb"],rgb_gray=3,channel_order="first",pause=0)
        #     print("-----------------------------")

        return obs

    def step(self, action):
        """Uses the :meth:`step` of the :attr:`env` that can be overwritten to change the returned data."""
        formatted_action = self._format_action.format(
            action=action, prev_heading=self._prev_heading
        )
        obs, reward, done, info = self.env.step(formatted_action)
        self._prev_heading = obs["ego_vehicle_state"]["heading"]
        obs = self._process(obs)
        return obs, reward, done, info

    def reset(self):
        """Uses the :meth:`reset` of the :attr:`env` that can be overwritten to change the returned data."""

        self._frame_stack.reset()
        obs = self.env.reset()
        self._prev_heading = obs["ego_vehicle_state"]["heading"]
        obs = self._process(obs)
        return obs
