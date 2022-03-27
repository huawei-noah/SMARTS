import torch
import torch.nn as nn
import torch.nn.functional as F

from rlkit.torch.utils import pytorch_util as ptu


class RNNDisc(nn.Module):
    def __init__(
        self,
        input_dim,
        hid_dim=100,
        hid_act="relu",
        rnn_act="gru",  # gru, lstm
        num_layers=2,
        use_bn=True,
        drop_out=-1,
        bidirectional=True,
        clamp_magnitude=10.0,
    ):
        super().__init__()

        self.hid_dim = hid_dim
        self.drop_out = 0
        if drop_out > 0:
            self.drop_out = drop_out
        self.rnn_act = rnn_act

        if hid_act == "relu":
            hid_act_class = nn.ReLU
        elif hid_act == "tanh":
            hid_act_class = nn.Tanh
        else:
            raise NotImplementedError()

        self.clamp_magnitude = clamp_magnitude

        self.mod_list = nn.ModuleList([nn.Linear(input_dim, hid_dim)])
        if use_bn:
            self.mod_list.append(nn.BatchNorm1d(hid_dim))
        self.mod_list.append(hid_act_class())

        self.before_linear_model = nn.Sequential(*self.mod_list)

        self.rnn = None
        if self.rnn_act == "gru":
            self.rnn = nn.GRU(
                input_size=hid_dim,
                hidden_size=hid_dim,
                num_layers=num_layers,
                batch_first=True,
                bias=True,
                dropout=self.drop_out,
                bidirectional=bidirectional,
            )
        elif self.rnn_act == "lstm":
            self.rnn = nn.LSTM(
                input_size=hid_dim,
                hidden_size=hid_dim,
                num_layers=num_layers,
                batch_first=True,
                bias=True,
                dropout=self.drop_out,
                bidirectional=bidirectional,
            )
        else:
            raise NotImplementedError

        if bidirectional:
            hid_dim *= 2

        self.after_linear_model = nn.Sequential(nn.Linear(hid_dim, 1))

    def forward(self, batch):  # shape (batch_size, length, dim)
        output = self.before_linear_model(batch)
        output = output.permute(1, 0, 2)  # （length, batch_size, dim）

        # output, (final_hidden_state, final_cell_state) = self.rnn(output)
        output, final_state = self.rnn(output)

        output = self.after_linear_model(output)

        output = torch.clamp(
            output, min=-1.0 * self.clamp_magnitude, max=self.clamp_magnitude
        )  # （length, batch_si, dim）

        output = output.permute(1, 0, 2)

        return output  # (batch_size, length, 1)
