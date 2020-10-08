from smarts.core.agent import AgentPolicy


class KeepLanePolicy(AgentPolicy):
    def act(self, obs):
        return "keep_lane"
