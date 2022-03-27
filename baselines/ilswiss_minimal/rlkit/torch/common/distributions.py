import torch
from numbers import Number
from numpy import log as np_log
from numpy import pi

from rlkit.torch.utils import pytorch_util as ptu


log_2pi = np_log(2 * pi)


class ReparamMultivariateNormalDiag:
    """
    My reparameterized normal implementation
    """

    def __init__(self, mean, log_sig_diag):
        self.mean = mean
        self.log_sig_diag = log_sig_diag
        self.log_cov = 2.0 * log_sig_diag
        self.cov = torch.exp(self.log_cov)
        self.sig = torch.exp(self.log_sig_diag)

    def sample(self):
        eps = torch.randn(self.mean.size(), requires_grad=False)
        if ptu.gpu_enabled():
            eps = eps.cuda(ptu.device)
        samples = eps * self.sig + self.mean
        return samples

    def sample_n(self, n):
        # cleanly expand float or Tensor or Variable parameters
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v], requires_grad=False).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        expanded_mean = expand(self.mean)
        expanded_sig = expand(self.sig)
        eps = torch.randn(expanded_mean.size(), requires_grad=False)
        return eps * expanded_sig + expanded_mean

    def log_prob(self, value):
        assert value.dim() >= 2, "Where is the batch dimension?"

        log_prob = -0.5 * torch.sum(
            (self.mean - value) ** 2 / self.cov, -1, keepdim=True
        )
        rest = torch.sum(self.log_sig_diag, -1, keepdim=True) + 0.5 * log_2pi
        log_prob -= rest
        return log_prob


class ReparamTanhMultivariateNormal:
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ MultivariateNormal(mean, log_sig_diag)
    """

    def __init__(self, normal_mean, normal_log_sig_diag, epsilon=1e-6):
        """
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal = ReparamMultivariateNormalDiag(normal_mean, normal_log_sig_diag)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            # assert False, 'Not handling this'
            # pre_tanh_value = torch.log(
            #     (1 + value) / (1 - value)
            # ) / 2
            pre_tanh_value = 0.5 * (
                torch.log(1.0 + value + self.epsilon)
                - torch.log(1.0 - value + self.epsilon)
            )
        normal_log_prob = self.normal.log_prob(pre_tanh_value)
        # print(torch.max(normal_log_prob))
        jacobi_term = torch.sum(
            torch.log(1 - value**2 + self.epsilon), -1, keepdim=True
        )
        # print(torch.min(jacobi_term))
        log_prob = normal_log_prob - jacobi_term
        # print(torch.max(log_prob))
        return log_prob

    def sample(self, return_pretanh_value=False):
        z = self.normal.sample()
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)
