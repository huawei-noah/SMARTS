import torch
import torch.nn as nn
import numpy as np


class PrecogFeatureExtractor(nn.Module):
    def __init__(
        self,
        hidden_units=64,
        n_social_features=None,
        social_capacity=None,
        embed_dim=None,
        seed=None,
    ):
        super(PrecogFeatureExtractor, self).__init__()
        if seed is not None:
            self.seed = torch.manual_seed(seed)

        self.social_capacity = social_capacity
        self.embed_dim = embed_dim
        self.n_social_features = n_social_features
        self.social_net = nn.Sequential(
            nn.Linear(n_social_features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, embed_dim),
        )

        self.output_dim = embed_dim * social_capacity

    def forward(self, social_states, training=False):
        if self.social_capacity == 0:
            social_states = [torch.empty(0, self.n_social_features)]
        else:
            social_states = [e[: self.social_capacity] for e in social_states]

        social_lens = [len(e) for e in social_states]
        slice_indices = list(np.cumsum(social_lens))
        slice_indices.insert(0, 0)
        social_features = torch.cat(social_states, 0)
        social_embeddings = self.social_net(social_features)
        masked_embeddings = [
            self._mask_embedding(
                social_embeddings[slice_indices[j] : slice_indices[j + 1]]
            )
            for j in range(len(social_states))
        ]

        state = [e.unsqueeze(0) for e in masked_embeddings]

        return state, {}

    def _mask_embedding(self, embedding):
        device = self.social_net[0].weight.device
        output = torch.zeros(self.social_capacity, self.embed_dim, device=device)
        limit = min(embedding.shape[0], self.social_capacity)
        output[:limit] = embedding[:limit]
        return output.flatten()
