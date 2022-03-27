from collections import OrderedDict
import gtimer as gt
from tqdm import tqdm

import torch
import torch.optim as optim

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.algorithms.torch_base_algorithm import TorchBaseAlgorithm


class BC(TorchBaseAlgorithm):
    def __init__(
        self,
        mode,  # 'MLE' or 'MSE'
        expert_replay_buffer,
        num_updates_per_train_call=1,
        batch_size=1024,
        lr=1e-3,
        momentum=0.0,
        optimizer_class=optim.Adam,
        **kwargs,
    ):
        assert mode in ["MLE", "MSE"], "Invalid mode!"
        super().__init__(**kwargs)

        self.mode = mode
        self.expert_replay_buffer = expert_replay_buffer

        self.batch_size = batch_size

        self.optimizer_n = {
            p_id: optimizer_class(
                self.exploration_policy_n[p_id].parameters(),
                lr=lr,
                betas=(momentum, 0.999),
            )
            for p_id in self.policy_ids
        }

        self.num_updates_per_train_call = num_updates_per_train_call

    def _can_evaluate(self, *args, **kwargs):
        return True

    def get_batch(self, batch_size, agent_id, keys=None, use_expert_buffer=True):
        if use_expert_buffer:
            rb = self.expert_replay_buffer
        else:
            rb = self.replay_buffer
        batch = rb.random_batch(batch_size, agent_id, keys=keys)
        batch = np_to_pytorch_batch(batch)
        return batch

    def start_training(self, start_epoch=0):
        for epoch in gt.timed_for(
            range(start_epoch, self.num_epochs),
            save_itrs=True,
        ):
            self._start_epoch(epoch)
            for steps_this_epoch in tqdm(range(self.num_env_steps_per_epoch)):
                self._n_env_steps_total += 1
                if self._n_env_steps_total % self.num_steps_between_train_calls == 0:
                    gt.stamp("sample")
                    self._try_to_train(epoch)
                    gt.stamp("train")

            gt.stamp("sample")
            self._try_to_eval(epoch)
            gt.stamp("eval")
            self._end_epoch()

    def _do_training(self, epoch):
        for t in range(self.num_updates_per_train_call):
            for a_id in self.agent_ids:
                self._do_update_step(epoch, a_id, use_expert_buffer=True)

    def _do_update_step(self, epoch, agent_id, use_expert_buffer=True):
        policy_id = self.policy_mapping_dict[agent_id]

        batch = self.get_batch(
            self.batch_size,
            agent_id,
            keys=["observations", "actions"],
            use_expert_buffer=use_expert_buffer,
        )

        obs = batch["observations"]
        acts = batch["actions"]

        self.optimizer_n[policy_id].zero_grad()
        if self.mode == "MLE":
            log_prob = self.exploration_policy_n[policy_id].get_log_prob(obs, acts)
            loss = -1.0 * log_prob.mean()
            if self.eval_statistics is None:
                self.eval_statistics = OrderedDict()
                self.eval_statistics[f"{policy_id} Log-Likelihood"] = ptu.get_numpy(
                    -1.0 * loss
                )
        else:
            pred_acts = self.exploration_policy_n[policy_id](obs)[0]
            squared_diff = (pred_acts - acts) ** 2
            loss = torch.sum(squared_diff, dim=-1).mean()
            if self.eval_statistics is None:
                self.eval_statistics = OrderedDict()
                self.eval_statistics[f"{policy_id} MSE"] = ptu.get_numpy(loss)
        loss.backward()
        self.optimizer_n[policy_id].step()

    @property
    def networks_n(self):
        return {p_id: [self.exploration_policy_n[p_id]] for p_id in self.policy_ids}

    def get_epoch_snapshot(self, epoch):
        """
        Probably will be overridden by each algorithm
        """
        data_to_save = dict(epoch=epoch)
        data_to_save.update(
            {
                p_id: dict(policy=self.exploration_policy_n[p_id])
                for p_id in self.policy_ids
            }
        )
        return data_to_save
