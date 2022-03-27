from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.core.trainer import Trainer
from rlkit.core.eval_util import create_stats_ordered_dict


class SoftActorCritic(Trainer):
    """
    version that:
        - uses reparameterization trick
        - has two Q functions and a V function
    TODO: Recently in rlkit there is a version which only uses two Q functions
    as well as an implementation of entropy tuning but I have not implemented
    those
    """

    def __init__(
        self,
        policy,
        qf1,
        qf2,
        vf,
        reward_scale=1.0,
        discount=0.99,
        alpha=1.0,
        policy_lr=1e-3,
        qf_lr=1e-3,
        vf_lr=1e-3,
        soft_target_tau=1e-2,
        policy_mean_reg_weight=1e-3,
        policy_std_reg_weight=1e-3,
        optimizer_class=optim.Adam,
        beta_1=0.9,
        **kwargs,
    ):
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.vf = vf
        self.reward_scale = reward_scale
        self.discount = discount
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight

        self.target_vf = vf.copy()
        self.eval_statistics = None

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(), lr=policy_lr, betas=(beta_1, 0.999)
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(), lr=qf_lr, betas=(beta_1, 0.999)
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(), lr=qf_lr, betas=(beta_1, 0.999)
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(), lr=vf_lr, betas=(beta_1, 0.999)
        )

        self.alpha = alpha

    def train_step(self, batch):
        # q_params = itertools.chain(self.qf1.parameters(), self.qf2.parameters())
        # v_params = itertools.chain(self.vf.parameters())
        # policy_params = itertools.chain(self.policy.parameters())

        rewards = self.reward_scale * batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]

        """
        QF Loss
        """
        # Only unfreeze parameter of Q
        # for p in itertools.chain(self.qf1.parameters(), self.qf2.parameters()):
        #     p.requires_grad = True
        # for p in self.vf.parameters():
        #     p.requires_grad = False
        # for p in self.policy.parameters():
        #     p.requires_grad = False
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        target_v_values = self.target_vf(
            next_obs
        )  # do not need grad || it's the shared part of two calculation
        q_target = (
            rewards + (1.0 - terminals) * self.discount * target_v_values
        )  # original implementation has detach
        q_target = q_target.detach()
        qf1_loss = 0.5 * torch.mean((q1_pred - q_target) ** 2)
        qf2_loss = 0.5 * torch.mean((q2_pred - q_target) ** 2)

        # freeze parameter of Q
        # for p in itertools.chain(self.qf1.parameters(), self.qf2.parameters()):
        #     p.requires_grad = False

        """
        VF Loss
        """
        # Only unfreeze parameter of V
        # for p in itertools.chain(self.qf1.parameters(), self.qf2.parameters()):
        #     p.requires_grad = False
        # for p in self.vf.parameters():
        #     p.requires_grad = True
        # for p in self.policy.parameters():
        #     p.requires_grad = True  ##
        self.vf_optimizer.zero_grad()
        v_pred = self.vf(obs)
        # Make sure policy accounts for squashing functions like tanh correctly!
        policy_outputs = self.policy(obs, return_log_prob=True)
        # in this part, we only need new_actions and log_pi with no grad
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        q1_new_acts = self.qf1(obs, new_actions)
        q2_new_acts = self.qf2(obs, new_actions)  # error
        q_new_actions = torch.min(q1_new_acts, q2_new_acts)
        v_target = q_new_actions - self.alpha * log_pi
        v_target = v_target.detach()
        vf_loss = 0.5 * torch.mean((v_pred - v_target) ** 2)

        qf1_loss.backward()
        qf2_loss.backward()
        vf_loss.backward()

        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        self.vf_optimizer.step()

        """
        Policy Loss
        """
        # for p in itertools.chain(self.qf1.parameters(), self.qf2.parameters()):
        #     p.requires_grad = False
        # for p in self.vf.parameters():
        #     p.requires_grad = False
        # for p in self.policy.parameters():
        #     p.requires_grad = True
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        q1_new_acts = self.qf1(obs, new_actions)
        q2_new_acts = self.qf2(obs, new_actions)  # error
        q_new_actions = torch.min(q1_new_acts, q2_new_acts)

        self.policy_optimizer.zero_grad()
        policy_loss = torch.mean(self.alpha * log_pi - q_new_actions)
        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        policy_reg_loss = mean_reg_loss + std_reg_loss
        policy_loss = policy_loss + policy_reg_loss
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Update networks
        """
        # unfreeze all -> initial states
        # for p in itertools.chain(self.qf1.parameters(), self.qf2.parameters()):
        #     p.requires_grad = True
        # for p in self.vf.parameters():
        #     p.requires_grad = True
        # for p in self.policy.parameters():
        #     p.requires_grad = True

        # unfreeze parameter of Q
        # for p in itertools.chain(self.qf1.parameters(), self.qf2.parameters()):
        #     p.requires_grad = True

        self._update_target_network()

        """
        Save some statistics for eval
        """
        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()
            self.eval_statistics["Reward Scale"] = self.reward_scale
            self.eval_statistics["QF1 Loss"] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics["QF2 Loss"] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics["VF Loss"] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics["Policy Loss"] = np.mean(ptu.get_numpy(policy_loss))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q1 Predictions",
                    ptu.get_numpy(q1_pred),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q2 Predictions",
                    ptu.get_numpy(q2_pred),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "V Predictions",
                    ptu.get_numpy(v_pred),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Log Pis",
                    ptu.get_numpy(log_pi),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Policy mu",
                    ptu.get_numpy(policy_mean),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Policy log std",
                    ptu.get_numpy(policy_log_std),
                )
            )

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.vf,
            self.target_vf,
        ]

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def get_snapshot(self):
        return dict(
            qf1=self.qf1,
            qf2=self.qf2,
            policy=self.policy,
            vf=self.vf,
            target_vf=self.target_vf,
        )

    def get_eval_statistics(self):
        return self.eval_statistics

    def end_epoch(self):
        self.eval_statistics = None
