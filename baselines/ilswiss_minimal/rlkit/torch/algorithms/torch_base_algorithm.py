import abc

from rlkit.core.base_algorithm import BaseAlgorithm


class TorchBaseAlgorithm(BaseAlgorithm, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def networks_n(self):
        """
        Used in many settings such as moving to devices
        """
        pass

    def training_mode(self, mode):
        for networks in self.networks_n.values():
            for net in networks:
                net.train(mode)

    def to(self, device):
        for networks in self.networks_n.values():
            for net in networks:
                net.to(device)
