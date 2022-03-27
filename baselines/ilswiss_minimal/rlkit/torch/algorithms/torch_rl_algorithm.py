from collections import OrderedDict

from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.algorithms.torch_base_algorithm import TorchBaseAlgorithm
from rlkit.torch.algorithms.ppo.ppo import PPO


class TorchRLAlgorithm(TorchBaseAlgorithm):
    def __init__(
        self, trainer_n, batch_size, num_train_steps_per_train_call, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.trainer_n = trainer_n
        self.batch_size = batch_size
        self.num_train_steps_per_train_call = num_train_steps_per_train_call

    def get_batch(self, agent_id: str):
        batch = self.replay_buffer.random_batch(self.batch_size, agent_id)
        return np_to_pytorch_batch(batch)

    def get_all_trajs(self, agent_id: str):
        batch = self.replay_buffer.sample_all_trajs(agent_id)
        batch = [np_to_pytorch_batch(b) for b in batch]
        return batch

    def clear_buffer(self, agent_id: str):
        self.replay_buffer.clear(agent_id)

    @property
    def networks_n(self):
        return {p_id: self.trainer_n[p_id].networks for p_id in self.policy_ids}

    def training_mode(self, mode):
        for nets in self.networks_n.values():
            for net in nets:
                net.train(mode)

    def _do_training(self, epoch):
        for _ in range(self.num_train_steps_per_train_call):
            for a_id in self.agent_ids:
                p_id = self.policy_mapping_dict[a_id]
                if isinstance(self.trainer_n[p_id], PPO):
                    self.trainer_n[p_id].train_step(self.get_all_trajs(a_id))
                    self.clear_buffer(a_id)
                else:
                    self.trainer_n[p_id].train_step(self.get_batch(a_id))

    def get_epoch_snapshot(self, epoch):
        data_to_save = dict(epoch=epoch)
        data_to_save.update(
            {p_id: self.trainer_n[p_id].get_snapshot() for p_id in self.policy_ids}
        )
        return data_to_save

    def evaluate(self, epoch):
        self.eval_statistics = OrderedDict()
        for p_id in self.policy_ids:
            _statistics = self.trainer_n[p_id].get_eval_statistics()
            for name, data in _statistics.items():
                self.eval_statistics.update(OrderedDict({f"{p_id} {name}": data}))
        super().evaluate(epoch)

    def _end_epoch(self):
        for p_id in self.policy_ids:
            self.trainer_n[p_id].end_epoch()
        super()._end_epoch()
