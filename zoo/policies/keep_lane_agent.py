from smarts.core.agent import Agent


class KeepLaneAgent(Agent):
    def act(self, obs):
        return "change_lane_left"
