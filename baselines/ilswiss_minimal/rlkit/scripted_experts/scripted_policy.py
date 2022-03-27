import abc


class ScriptedPolicy:
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def reset(self, env, *args, **kwargs):
        """
        !! The environments must be reset before resetting a scripted expert !!
        """
        pass

    @abc.abstractmethod
    def get_action(self, obs, env, timestep):
        pass
