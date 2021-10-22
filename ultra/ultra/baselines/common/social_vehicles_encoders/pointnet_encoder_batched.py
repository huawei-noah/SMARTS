# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# Inspired by https://github.com/fxia22/pointnet.pytorch/blob/master/pointnet/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class STNkd(nn.Module):
    def __init__(self, k=64, nc=16):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, nc, 1)
        self.conv2 = torch.nn.Conv1d(nc, nc * 4, 1)
        self.conv3 = torch.nn.Conv1d(nc * 4, nc * 16, 1)
        self.fc1 = nn.Linear(nc * 16, nc * 4)
        self.fc2 = nn.Linear(nc * 4, nc)
        self.fc3 = nn.Linear(nc, k * k)
        self.relu = nn.ReLU()

        # self.bn1 = nn.BatchNorm1d(nc)
        # self.bn2 = nn.BatchNorm1d(nc * 4)
        # self.bn3 = nn.BatchNorm1d(nc * 16)
        # self.bn4 = nn.BatchNorm1d(nc * 4)
        # self.bn5 = nn.BatchNorm1d(nc)
        identity = lambda x: x
        self.bn1, self.bn2, self.bn3, self.bn4, self.bn5 = [identity] * 5

        self.k = k
        self.nc = nc

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.nc * 16)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = (
            Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)))
            .view(1, self.k * self.k)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PNEncoderBatched(nn.Module):
    def __init__(
        self,
        input_dim=3,
        global_features=True,
        feature_transform=True,
        nc=16,
        transform_loss_weight=0.1,
    ):
        assert global_features
        super(PNEncoderBatched, self).__init__()
        self.input_dim = input_dim
        self.nc = nc
        self.global_features = global_features
        self.feature_transform = feature_transform
        self.transform_loss_weight = transform_loss_weight

        self.transformD = STNkd(k=input_dim, nc=nc)
        self.conv1 = nn.Conv1d(self.input_dim, nc, 1)
        # self.bn1 = nn.BatchNorm1d(nc)
        self.conv2 = nn.Conv1d(nc, nc * 4, 1)
        # self.bn2 = nn.BatchNorm1d(nc * 4)
        self.conv3 = nn.Conv1d(nc * 4, nc * 16, 1)
        # self.bn3 = nn.BatchNorm1d(nc * 16)
        identity = lambda x: x
        self.bn1, self.bn2, self.bn3 = [identity] * 3
        # self.bn1, self.bn2, self.bn3 = nn.LayerNorm(nc), nn.LayerNorm(nc * 4), nn.LayerNorm(nc * 16)

        self.output_dim = nc * 16

        if self.feature_transform:
            self.transformF = STNkd(k=self.nc, nc=nc)

        self.empty_fill = torch.from_numpy(
            np.asarray(
                [
                    -1,
                    -1,  # bottom left very far from ego
                    -1,
                    0,  # move backward from ego
                ]
            )
        ).unsqueeze(0)

    def transform_loss(self, transD, transF):
        transform_loss_raw_number = [
            [
                feature_transform_regularizer(e.cpu()).to(e.device)
                if e is not None
                else 0.0
                for e in transD
            ],
            [
                feature_transform_regularizer(e.cpu()).to(e.device)
                if e is not None
                else 0.0
                for e in transF
            ],
        ]
        transform_loss = [
            sum(transform_loss_raw_number[0]) / len(transform_loss_raw_number[0]),
            sum(transform_loss_raw_number[1]) / len(transform_loss_raw_number[1]),
        ]
        mean_transform_loss = sum(transform_loss) / len(transform_loss)
        aux_losses = {
            "transform": {
                "value": mean_transform_loss,
                "weight": self.transform_loss_weight,
            }
        }
        return aux_losses

    def forward(self, social_vehicles_state, training=False):
        max_num_social_in_batch = max([len(e) for e in social_vehicles_state])
        max_num_social_in_batch = max(max_num_social_in_batch, 1)
        # because PointNet is taking the trajectory-wise max
        # padding with a bunch of zero here wouldn't change the final training result
        # if training is proper (need to test)
        # also, it is possible to pad with any amount of zeros
        self.empty_fill = self.empty_fill.to(social_vehicles_state[0].device)
        input_tensor = torch.zeros(
            len(social_vehicles_state),
            max_num_social_in_batch,
            social_vehicles_state[0].shape[-1],
            device=social_vehicles_state[0].device,
        )
        for j in range(len(social_vehicles_state)):
            input_tensor[j, : len(social_vehicles_state[j])] = social_vehicles_state[j]
            # if want to fill with zero, comment line below or set self.empty_fil to 0
            input_tensor[j, len(social_vehicles_state[j]) :] = self.empty_fill
        social_features, transD, transF = self._forward(input_tensor)
        social_features = [e.unsqueeze(0) for e in social_features]
        transD = [e.unsqueeze(0) for e in transD]
        transF = [e.unsqueeze(0) for e in transF]
        if training:
            aux_losses = self.transform_loss(transD, transF)
            return social_features, aux_losses
        else:
            return social_features, {}

    def _forward(self, points):
        points = points.transpose(-1, -2)
        transD = self.transformD(points)
        x = points.transpose(2, 1)
        x = torch.bmm(x, transD)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            transF = self.transformF(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, transF)
            x = x.transpose(2, 1)
        else:
            transF = None

        point_features = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.output_dim)

        if self.global_features:
            return x, transD, transF
        else:
            N = points.size()[2]
            x = x.view(-1, self.nc, 1).repeat(1, 1, N)
            return torch.cat([x, point_features], 1), transD, transF


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(
        torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2))
    )
    if torch.isnan(loss):
        loss = 0.0
    return loss
