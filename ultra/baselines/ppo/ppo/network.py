import torch
from torch import nn
from torch.distributions.normal import Normal


class ActorNetwork(nn.Module):
    def __init__(
        self, state_size, action_size, hidden_units, social_feature_encoder=None
    ):
        super(ActorNetwork, self).__init__()
        self.social_feature_encoder = social_feature_encoder

        self.model = nn.Sequential(
            nn.Linear(state_size, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, action_size),
            nn.Tanh(),
        )

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
        state = (
            torch.cat([low_dim_state, social_feature], -1)
            if len(social_feature) > 0
            else low_dim_state
        )
        a = self.model(state)

        if training:
            return a, aux_losses
        else:
            return a, {}


class CriticNetwork(nn.Module):
    def __init__(self, state_size, hidden_units, social_feature_encoder=None):
        super(CriticNetwork, self).__init__()
        self.social_feature_encoder = social_feature_encoder

        self.model = nn.Sequential(
            nn.Linear(state_size, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
        )

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

        state = (
            torch.cat([low_dim_state, social_feature], -1)
            if len(social_feature) > 0
            else low_dim_state
        )
        q = self.model(state)
        if training:
            return q, aux_losses
        else:
            return q, {}


class PPONetwork(nn.Module):
    def __init__(
        self,
        action_size,
        state_size,
        seed=None,
        hidden_units=64,
        init_std=0.5,
        social_feature_encoder_class=None,
        social_feature_encoder_params=None,
    ):
        super(PPONetwork, self).__init__()

        if seed is not None:
            torch.manual_seed(seed)
        self.social_feature_encoder_class = social_feature_encoder_class
        self.social_feature_encoder_params = social_feature_encoder_params

        self.critic = CriticNetwork(
            state_size,
            hidden_units,
            social_feature_encoder=self.social_feature_encoder_class(
                **self.social_feature_encoder_params
            )
            if self.social_feature_encoder_class
            else None,
        )

        self.actor = ActorNetwork(
            state_size,
            action_size,
            hidden_units,
            social_feature_encoder=self.social_feature_encoder_class(
                **self.social_feature_encoder_params
            )
            if self.social_feature_encoder_class
            else None,
        )

        # self.init_last_layer(self.actor)
        self.log_std = nn.Parameter(torch.log(init_std * torch.ones(1, action_size)))

    def forward(self, x, training=False):
        value, critic_aux_loss = self.critic(x, training=training)
        mu, actor_aux_loss = self.actor(x, training=training)
        std = torch.ones_like(mu) * 0.5
        dist = Normal(mu, std)

        if training:
            aux_losses = {}
            for k, v in actor_aux_loss.items():
                aux_losses.update({"actor/{}".format(k): v})
            for k, v in critic_aux_loss.items():
                aux_losses.update({"critic/{}".format(k): v})
            return (dist, value), aux_losses
        else:
            return dist, value

    def init_last_layer(self, actor):
        """Initialize steering to zero and throttle to maximum"""
        # nn.init.constant_(actor.model[-2].weight.data[0], 1.0)
        # nn.init.constant_(actor.model[-2].weight.data[1], 0.0)
        nn.init.constant_(actor.model[-2].bias.data[0], 2.0)
