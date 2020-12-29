import torch
from torch import nn


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class DQNCNN(nn.Module):
    def __init__(
        self,
        n_in_channels,
        image_dim,
        state_size,
        num_actions,
        hidden_dim=128,
        activation=nn.ReLU,
    ):
        super(DQNCNN, self).__init__()
        self.im_feature = nn.Sequential(
            nn.Conv2d(n_in_channels, 32, 8, 4),
            activation(),
            nn.Conv2d(32, 64, 4, 2),
            activation(),
            nn.Conv2d(64, 64, 3, 1),
            activation(),
            Flatten(),
        )

        dummy = torch.zeros((1, n_in_channels, *image_dim))
        im_feature_size = self.im_feature(dummy).data.cpu().numpy().size

        self.q_outs = nn.ModuleList()
        for num_action in num_actions:
            q_out = nn.Sequential(
                nn.Linear(im_feature_size + state_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_action),
            )
            self.q_outs.append(q_out)

        self.init()

    def init(self):
        for q_out in self.q_outs:
            # nn.init.normal_(q_out[-1].weight.data, 0.0, 1e-2)
            nn.init.constant_(q_out[-1].bias.data, 0.0)

    def forward(self, image, state_size):
        im_feature = self.im_feature(image)
        x = torch.cat([im_feature, state_size], dim=-1)
        x = [e(x) for e in self.q_outs]
        return x


class DQNFC(nn.Module):
    def __init__(
        self,
        num_actions,
        state_size,
        hidden_dim=256,
        activation=nn.ReLU,
    ):
        super(DQNFC, self).__init__()

        self.q_outs = nn.ModuleList()
        for num_action in num_actions:
            q_out = nn.Sequential(
                nn.Linear(state_size, hidden_dim),
                activation(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                activation(),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                activation(),
                nn.Linear(hidden_dim // 4, num_action),
            )
            self.q_outs.append(q_out)

        self.init()

    def init(self):
        for q_out in self.q_outs:
            nn.init.constant_(q_out[-1].bias.data, 0.0)
            # nn.init.normal_(q_out[-1].weight.data, 0.0, 1e-3)

    def forward(self, state, training=False):
        low_dim_state = state["low_dim_states"]
        if len(low_dim_state.shape) == 1:
            low_dim_state = torch.unsqueeze(low_dim_state, 0)
            unsqueezed = True
        else:
            unsqueezed = False
        x = low_dim_state
        # x = self.feature(low_dim_state)
        x = [e(x) for e in self.q_outs]
        if unsqueezed:
            x = [torch.squeeze(e, 0) for e in x]

        if training:
            aux_losses = {}
            return x, aux_losses
        else:
            return x


class DQNWithSocialEncoder(nn.Module):
    def __init__(
        self,
        num_actions,
        state_size,
        hidden_dim=256,
        activation=nn.ReLU,
        social_feature_encoder_class=None,
        social_feature_encoder_params=None,
    ):
        super(DQNWithSocialEncoder, self).__init__()

        self.social_feature_encoder = (
            social_feature_encoder_class(**social_feature_encoder_params)
            if social_feature_encoder_class
            else None
        )

        self.state_size = state_size
        self.q_outs = nn.ModuleList()
        for num_action in num_actions:
            q_out = nn.Sequential(
                nn.Linear(state_size, hidden_dim),
                activation(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                activation(),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                activation(),
                nn.Linear(hidden_dim // 4, num_action),
            )
            self.q_outs.append(q_out)

        self.init()

    def init(self):
        for q_out in self.q_outs:
            nn.init.constant_(q_out[-1].bias.data, 0.0)

    def forward(self, state, training=False):
        low_dim_state = state["low_dim_states"]
        social_vehicles_state = state["social_vehicles"]

        aux_losses = {}
        social_feature = []
        if self.social_feature_encoder is not None:
            social_feature, social_encoder_aux_losses = self.social_feature_encoder(
                social_vehicles_state, training
            )
            aux_losses.update(social_encoder_aux_losses)
        else:
            social_feature = [e.reshape(1, -1) for e in social_vehicles_state]

        social_feature = torch.cat(social_feature, 0) if len(social_feature) > 0 else []
        x = (
            torch.cat([low_dim_state, social_feature], -1)
            if len(social_feature) > 0
            else low_dim_state
        )
        x = [e(x) for e in self.q_outs]

        if training:
            return x, aux_losses
        else:
            return x
