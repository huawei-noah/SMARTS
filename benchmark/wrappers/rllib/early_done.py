from benchmark.wrappers.rllib.frame_stack import FrameStack


class EarlyDone(FrameStack):
    def step(self, agent_actions):
        observations, rewards, dones, infos = super(EarlyDone, self).step(agent_actions)
        dones["__all__"] = any(list(dones.values()))
        return observations, rewards, dones, infos
