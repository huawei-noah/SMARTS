import copy
import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.pe

        return self.dropout(x)


class SelfTransformer(nn.Module):
    def __init__(self, heads=8, dim=128, dropout=0.1):
        super(SelfTransformer, self).__init__()
        self.self_attention = nn.MultiheadAttention(
            dim, heads, dropout, batch_first=True
        )
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs, mask=None):
        if mask.shape[1] == 11:
            mask[:, -1] = False
        else:
            mask[:, 0] = False

        attention_output, _ = self.self_attention(
            inputs, inputs, inputs, key_padding_mask=mask
        )
        attention_output = self.norm_1(attention_output + inputs)
        output = self.norm_2(self.ffn(attention_output) + attention_output)

        return output


class AgentEncoder(nn.Module):
    def __init__(self):
        super(AgentEncoder, self).__init__()
        self.position = nn.Sequential(nn.Linear(5, 128), nn.ReLU(), nn.Linear(128, 256))
        self.encode = PositionalEncoding(d_model=256, max_len=11)

    def forward(self, inputs):
        output = self.encode(self.position(inputs))

        return output


class InteractionEncoder(nn.Module):
    def __init__(self):
        super(InteractionEncoder, self).__init__()
        self.time_1 = SelfTransformer(dim=256)
        self.agent_1 = SelfTransformer(dim=256)
        self.time_2 = SelfTransformer(dim=256)
        self.agent_2 = SelfTransformer(dim=256)
        self.time_3 = SelfTransformer(dim=256)
        self.agent_3 = SelfTransformer(dim=256)

    def forward(self, actors, mask):
        N = actors.shape[1]
        T = actors.shape[2]

        actors = torch.stack(
            [self.time_1(actors[:, i], mask[:, i]) for i in range(N)], dim=1
        )
        actors = torch.stack(
            [self.agent_1(actors[:, :, t], mask[:, :, t]) for t in range(T)], dim=2
        )
        actors = torch.stack(
            [self.time_2(actors[:, i], mask[:, i]) for i in range(N)], dim=1
        )
        actors = torch.stack(
            [self.agent_2(actors[:, :, t], mask[:, :, t]) for t in range(T)], dim=2
        )
        actors = torch.stack(
            [self.time_3(actors[:, i], mask[:, i]) for i in range(N)], dim=1
        )
        actors = torch.stack(
            [self.agent_3(actors[:, :, t], mask[:, :, t]) for t in range(T)], dim=2
        )
        actors = actors[:, :, -1]

        return actors


class Decoder(nn.Module):
    def __init__(self, modalities=3):
        super(Decoder, self).__init__()
        self.modalities = modalities
        decoder = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 30 * 3)
        )
        self.decoders = nn.ModuleList(
            [copy.deepcopy(decoder) for _ in range(modalities)]
        )
        scorer = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(384, 128), nn.ELU(), nn.Linear(128, 1)
        )
        self.scorers = nn.ModuleList([copy.deepcopy(scorer) for _ in range(modalities)])
        self.encode = nn.Linear(90, 128)

    def forward(self, hidden, state):
        N = state.shape[1]
        trajs = []
        scores = []

        for i in range(self.modalities):
            pred = self.decoders[i](hidden).view(-1, N, 30, 3)
            trajs.append(pred + state[:, :, None, :3])
            score = self.scorers[i](
                torch.cat([hidden, self.encode(pred.view(-1, N, 90).detach())], dim=-1)
            )
            scores.append(score.squeeze(-1))

        trajs = torch.stack(trajs, dim=2)
        scores = torch.stack(scores, dim=2)

        return trajs, scores


class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        # Observation space
        # Ego: (B, T_h, 4)
        # Neighbors: (B, N_n, T_h, 4)

        self.agent_encoder = AgentEncoder()
        self.interaction_encoder = InteractionEncoder()
        self.decoder = Decoder()

    def forward(self, observations):
        # get inputs and encode them
        for key, sub_space in observations.items():
            if key == "ego_state":
                ego = sub_space
                encoded_ego = [self.agent_encoder(ego)]
            elif key == "neighbors_state":
                neighbors = sub_space
                encoded_neighbors = [
                    self.agent_encoder(neighbors[:, i])
                    for i in range(neighbors.shape[1])
                ]
            else:
                raise KeyError

        # interaction Transformer
        actors = torch.cat([ego.unsqueeze(1), neighbors], dim=1)
        actors_mask = torch.eq(actors.sum(-1), 0)
        encoded_actors = torch.stack(encoded_ego + encoded_neighbors, dim=1)
        encoded_actors = self.interaction_encoder(encoded_actors, actors_mask)
        prediction = self.decoder(encoded_actors[:, 1:], neighbors[:, :, -1])

        return prediction
