from smarts.core.agent import Agent


class KeepLaneAgent(Agent):
    def act(self, obs):
        return "keep_lane"


class KeepLaneAgentFormatted(Agent):
    def act(self, obs):
        return 0
