import abc


class Trainer(object, metaclass=abc.ABCMeta):
    """
    Inspired by the recent version of rlkit
    Trainers are the RL optimization algorithms that
    can be plugged into other algorithms
    E.g. SAC, TD3, etc.
    """

    @abc.abstractmethod
    def train_step(self, batch):
        pass

    def get_eval_statistics(self):
        return {}

    def get_snapshot(self):
        return {}

    def end_epoch(self):
        pass

    @property
    @abc.abstractmethod
    def networks(self):
        pass
