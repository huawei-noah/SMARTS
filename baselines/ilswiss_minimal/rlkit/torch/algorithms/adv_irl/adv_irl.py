import numpy as np
from collections import OrderedDict

import torch
import torch.optim as optim
from torch import nn
from torch import autograd
import torch.nn.functional as F

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.algorithms.torch_base_algorithm import TorchBaseAlgorithm


class AdvIRL(TorchBaseAlgorithm):
    """
    Depending on choice of reward function and size of replay
    buffer this will be:
        - AIRL
        - GAIL (without extra entropy term)
        - FAIRL
        - Discriminator Actor Critic

    I did not implement the reward-wrapping mentioned in
    https://arxiv.org/pdf/1809.02925.pdf though

    Features removed from v1.0:
        - gradient clipping
        - target disc (exponential moving average disc)
        - target policy (exponential moving average policy)
        - disc input noise
    """

    def __init__(
        self,
        mode,  # airl, gail, or fairl
        discriminator_n,
        policy_trainer_n,
        expert_replay_buffer,
        state_only=False,
        disc_optim_batch_size=1024,
        policy_optim_batch_size=1024,
        policy_optim_batch_size_from_expert=0,
        num_update_loops_per_train_call=1,
        num_disc_updates_per_loop_iter=100,
        num_policy_updates_per_loop_iter=100,
        disc_lr=1e-3,
        disc_momentum=0.0,
        disc_optimizer_class=optim.Adam,
        use_grad_pen=True,
        grad_pen_weight=10,
        rew_clip_min=None,
        rew_clip_max=None,
        **kwargs,
    ):
        assert mode in [
            "airl",
            "gail",
            "fairl",
            "gail2",
        ], "Invalid adversarial irl algorithm!"
        super().__init__(**kwargs)

        self.mode = mode
        self.state_only = state_only

        self.expert_replay_buffer = expert_replay_buffer

        self.policy_trainer_n = policy_trainer_n
        self.policy_optim_batch_size = policy_optim_batch_size
        self.policy_optim_batch_size_from_expert = policy_optim_batch_size_from_expert

        self.discriminator_n = discriminator_n
        self.disc_optimizer_n = {
            policy_id: disc_optimizer_class(
                self.discriminator_n[policy_id].parameters(),
                lr=disc_lr,
                betas=(disc_momentum, 0.999),
            )
            for policy_id in self.policy_ids
        }

        self.disc_optim_batch_size = disc_optim_batch_size
        print("\n\nDISC MOMENTUM: %f\n\n" % disc_momentum)

        self.bce = nn.BCEWithLogitsLoss()
        self.bce_targets = torch.cat(
            [
                torch.ones(disc_optim_batch_size, 1),
                torch.zeros(disc_optim_batch_size, 1),
            ],
            dim=0,
        )
        self.bce.to(ptu.device)
        self.bce_targets.to(ptu.device)

        self.use_grad_pen = use_grad_pen
        self.grad_pen_weight = grad_pen_weight

        self.num_update_loops_per_train_call = num_update_loops_per_train_call
        self.num_disc_updates_per_loop_iter = num_disc_updates_per_loop_iter
        self.num_policy_updates_per_loop_iter = num_policy_updates_per_loop_iter

        self.rew_clip_min = rew_clip_min
        self.rew_clip_max = rew_clip_max
        self.clip_min_rews = rew_clip_min is not None
        self.clip_max_rews = rew_clip_max is not None

        self.disc_eval_statistics = None

    def get_batch(self, batch_size, agent_id, from_expert, keys=None):
        if from_expert:
            buffer = self.expert_replay_buffer
        else:
            buffer = self.replay_buffer
        batch = buffer.random_batch(batch_size, agent_id, keys=keys)
        batch = np_to_pytorch_batch(batch)
        return batch

    def _end_epoch(self):
        for p_id in self.policy_ids:
            self.policy_trainer_n[p_id].end_epoch()
        self.disc_eval_statistics = None
        super()._end_epoch()

    def evaluate(self, epoch):
        self.eval_statistics = OrderedDict()
        self.eval_statistics.update(self.disc_eval_statistics)
        for p_id in self.policy_ids:
            _statistics = self.policy_trainer_n[p_id].get_eval_statistics()
            for name, data in _statistics.items():
                self.eval_statistics.update({f"{p_id} {name}": data})
        super().evaluate(epoch)

    def _do_training(self, epoch):
        for t in range(self.num_update_loops_per_train_call):
            for a_id in self.agent_ids:
                for _ in range(self.num_disc_updates_per_loop_iter):
                    self._do_reward_training(epoch, a_id)
                for _ in range(self.num_policy_updates_per_loop_iter):
                    self._do_policy_training(epoch, a_id)

    def _do_reward_training(self, epoch, agent_id):
        """
        Train the discriminator
        """

        policy_id = self.policy_mapping_dict[agent_id]

        self.disc_optimizer_n[policy_id].zero_grad()

        keys = ["observations"]
        if self.state_only:
            keys.append("next_observations")
        else:
            keys.append("actions")

        expert_batch = self.get_batch(self.disc_optim_batch_size, agent_id, True, keys)
        policy_batch = self.get_batch(self.disc_optim_batch_size, agent_id, False, keys)

        expert_obs = expert_batch["observations"]
        policy_obs = policy_batch["observations"]

        if self.state_only:
            expert_next_obs = expert_batch["next_observations"]
            policy_next_obs = policy_batch["next_observations"]

            expert_disc_input = torch.cat([expert_obs, expert_next_obs], dim=1)
            policy_disc_input = torch.cat([policy_obs, policy_next_obs], dim=1)
        else:
            expert_acts = expert_batch["actions"]
            policy_acts = policy_batch["actions"]
            expert_disc_input = torch.cat([expert_obs, expert_acts], dim=1)
            policy_disc_input = torch.cat([policy_obs, policy_acts], dim=1)
        disc_input = torch.cat([expert_disc_input, policy_disc_input], dim=0)

        disc_logits = self.discriminator_n[policy_id](disc_input)
        disc_preds = (disc_logits > 0).type(disc_logits.data.type())
        disc_ce_loss = self.bce(disc_logits, self.bce_targets)
        accuracy = (disc_preds == self.bce_targets).type(torch.FloatTensor).mean()

        if self.use_grad_pen:
            eps = ptu.rand(expert_obs.size(0), 1)
            eps.to(ptu.device)

            interp_obs = eps * expert_disc_input + (1 - eps) * policy_disc_input
            interp_obs = interp_obs.detach()
            interp_obs.requires_grad_(True)

            gradients = autograd.grad(
                outputs=self.discriminator_n[policy_id](interp_obs).sum(),
                inputs=[interp_obs],
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )
            total_grad = gradients[0]

            # GP from Gulrajani et al.
            gradient_penalty = ((total_grad.norm(2, dim=1) - 1) ** 2).mean()
            disc_grad_pen_loss = gradient_penalty * self.grad_pen_weight

            # # GP from Mescheder et al.
            # gradient_penalty = (total_grad.norm(2, dim=1) ** 2).mean()
            # disc_grad_pen_loss = gradient_penalty * 0.5 * self.grad_pen_weight
        else:
            disc_grad_pen_loss = 0.0

        disc_total_loss = disc_ce_loss + disc_grad_pen_loss
        disc_total_loss.backward()
        self.disc_optimizer_n[policy_id].step()

        """
        Save some statistics for eval
        """
        if self.disc_eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.disc_eval_statistics = OrderedDict()

            self.disc_eval_statistics[f"{policy_id} Disc CE Loss"] = np.mean(
                ptu.get_numpy(disc_ce_loss)
            )
            self.disc_eval_statistics[f"{policy_id} Disc Acc"] = np.mean(
                ptu.get_numpy(accuracy)
            )
            if self.use_grad_pen:
                self.disc_eval_statistics[f"{policy_id} Grad Pen"] = np.mean(
                    ptu.get_numpy(gradient_penalty)
                )
                self.disc_eval_statistics[f"{policy_id} Grad Pen W"] = np.mean(
                    self.grad_pen_weight
                )

    def _do_policy_training(self, epoch, agent_id):

        policy_id = self.policy_mapping_dict[agent_id]

        if self.policy_optim_batch_size_from_expert > 0:
            policy_batch_from_policy_buffer = self.get_batch(
                self.policy_optim_batch_size - self.policy_optim_batch_size_from_expert,
                agent_id,
                False,
            )
            policy_batch_from_expert_buffer = self.get_batch(
                self.policy_optim_batch_size_from_expert,
                agent_id,
                True,
            )
            policy_batch = {}
            for k in policy_batch_from_policy_buffer:
                policy_batch[k] = torch.cat(
                    [
                        policy_batch_from_policy_buffer[k],
                        policy_batch_from_expert_buffer[k],
                    ],
                    dim=0,
                )
        else:
            policy_batch = self.get_batch(self.policy_optim_batch_size, agent_id, False)

        obs = policy_batch["observations"]
        acts = policy_batch["actions"]
        next_obs = policy_batch["next_observations"]

        self.discriminator_n[policy_id].eval()
        if self.state_only:
            disc_input = torch.cat([obs, next_obs], dim=1)
        else:
            disc_input = torch.cat([obs, acts], dim=1)
        disc_logits = self.discriminator_n[policy_id](disc_input).detach()
        self.discriminator_n[policy_id].train()

        # compute the reward using the algorithm
        if self.mode == "airl":
            # If you compute log(D) - log(1-D) then you just get the logits
            policy_batch["rewards"] = disc_logits
        elif self.mode == "gail":  # -log (1-D) > 0
            policy_batch["rewards"] = F.softplus(
                disc_logits, beta=1
            )  # F.softplus(disc_logits, beta=-1)
        elif self.mode == "gail2":  # log D < 0
            policy_batch["rewards"] = F.softplus(
                disc_logits, beta=-1
            )  # F.softplus(disc_logits, beta=-1)
        else:  # fairl
            policy_batch["rewards"] = torch.exp(disc_logits) * (-1.0 * disc_logits)

        if self.clip_max_rews:
            policy_batch["rewards"] = torch.clamp(
                policy_batch["rewards"], max=self.rew_clip_max
            )
        if self.clip_min_rews:
            policy_batch["rewards"] = torch.clamp(
                policy_batch["rewards"], min=self.rew_clip_min
            )

        # policy optimization step
        self.policy_trainer_n[policy_id].train_step(policy_batch)

        self.disc_eval_statistics[f"{agent_id} Disc Rew Mean"] = np.mean(
            ptu.get_numpy(policy_batch["rewards"])
        )
        self.disc_eval_statistics[f"{agent_id} Disc Rew Std"] = np.std(
            ptu.get_numpy(policy_batch["rewards"])
        )
        self.disc_eval_statistics[f"{agent_id} Disc Rew Max"] = np.max(
            ptu.get_numpy(policy_batch["rewards"])
        )
        self.disc_eval_statistics[f"{agent_id} Disc Rew Min"] = np.min(
            ptu.get_numpy(policy_batch["rewards"])
        )

    @property
    def networks_n(self):
        return {
            p_id: [self.discriminator_n[p_id]] + self.policy_trainer_n[p_id].networks
            for p_id in self.policy_ids
        }

    def get_epoch_snapshot(self, epoch):
        # snapshot = super().get_epoch_snapshot(epoch)
        snapshot = dict(epoch=epoch)
        for p_id in self.policy_ids:
            snapshot[p_id] = self.policy_trainer_n[p_id].get_snapshot()
            # snapshot.update(
            #     p_id=self.policy_trainer_n[p_id].get_snapshot()
            # )
            snapshot[p_id].update(disc=self.discriminator_n[p_id])
        return snapshot

    def to(self, device):
        self.bce.to(ptu.device)
        self.bce_targets = self.bce_targets.to(ptu.device)
        super().to(device)
