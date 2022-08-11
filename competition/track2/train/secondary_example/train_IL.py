import os
import pickle
import random
import argparse
import math
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import Dataset

from utility_IL import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ]
)


def create_datasets(dataset_path, save_path=None, cache=False, batch_size=32):
    class TensorDatasetTransforms(torch.utils.data.TensorDataset):
        def __init__(self, x, y):
            super().__init__(x, y)

        def __getitem__(self, index):
            tensor = data_transform(self.tensors[0][index])

            return (tensor,) + tuple(t[index] for t in self.tensors[1:])

    class CustomTensorDataset(Dataset):
        def __init__(self, x, y):
            tensors = (x, y)
            self.tensors = tensors

        def __getitem__(self, index):
            img = data_transform(self.tensors[0][0][index])
            goal = self.tensors[0][1][index]
            y = self.tensors[1][index]

            return [img, goal], y

        def __len__(self):

            return self.tensors[0][1].size(0)

    x, y = load_data(dataset_path, save_path, cache=cache)

    x_train = x[:]
    y_train = y[:]
    img = np.array(x_train[:, 0].tolist())
    goal = np.array(x_train[:, 1].tolist())
    x_train = [torch.tensor(img), torch.tensor(goal)]

    train_set = CustomTensorDataset(x_train, torch.tensor(y_train))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )

    return train_loader


"""
Model
"""


class Flatten(nn.Module):
    def forward(self, x):

        return x.view(x.size()[0], -1)


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 32, 4, 4)
        self.bachnorm1 = torch.nn.BatchNorm2d(32)
        self.elu = torch.nn.ELU()
        self.dropout2d = torch.nn.Dropout2d(0.5)
        self.conv2 = torch.nn.Conv2d(32, 64, 2, 2)
        self.batchnorm2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 64, 4, 4)
        self.flatten = Flatten()
        self.batchnorm3 = torch.nn.BatchNorm1d(64 * 8 * 8)
        self.dropout = torch.nn.Dropout()
        self.linear1 = torch.nn.Linear(64 * 8 * 8, 128)
        self.batchnorm4 = torch.nn.BatchNorm1d(128)
        self.linear2 = torch.nn.Linear(128, 32)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bachnorm1(x)
        x = self.elu(x)
        x = self.dropout2d(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.elu(x)
        x = self.dropout2d(x)

        x = self.conv3(x)
        x = self.elu(x)
        x = self.flatten(x)
        x = self.batchnorm3(x)
        x = self.dropout(x)

        x = self.linear1(x)
        x = self.elu(x)
        x = self.batchnorm4(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = self.elu(x)

        return x


class EmbBlock(nn.Module):
    def __init__(self):
        super(EmbBlock, self).__init__()

        self.linear1 = torch.nn.Linear(2, 32)
        self.elu = torch.nn.ELU()

    def forward(self, x):
        x = self.linear1(x.float())
        x = self.elu(x)

        return x


class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        self.convblock = ConvBlock()
        self.embblock = EmbBlock()

        self.linear1 = torch.nn.Linear(64, 32)
        self.linear4 = torch.nn.Linear(32, 1)
        self.linear2 = torch.nn.Linear(64, 32)
        self.linear5 = torch.nn.Linear(32, 1)
        self.linear3 = torch.nn.Linear(64, 32)
        self.linear6 = torch.nn.Linear(32, 1)

        self.elu = torch.nn.ELU()

        self.criterion = nn.MSELoss()

    def forward(self, x):
        img_emb = self.convblock(x[0])
        goal_emb = self.embblock(x[1])

        x = torch.cat((img_emb, goal_emb), dim=1)

        dx = self.linear1(x)
        dy = self.linear2(x)
        dheading = self.linear3(x)

        dx = self.elu(dx)
        dy = self.elu(dy)
        dheading = self.elu(dheading)

        nn_outputs = {}
        nn_outputs["dx"] = self.linear4(dx)
        nn_outputs["dy"] = self.linear5(dy)
        nn_outputs["d_heading"] = self.linear6(dheading)

        return nn_outputs

    def compute_loss(self, nn_outputs, labels):
        dx_loss = self.criterion(
            nn_outputs["dx"], labels[:, 0].view(labels.shape[0], 1)
        )
        dy_loss = self.criterion(
            nn_outputs["dy"], labels[:, 1].view(labels.shape[0], 1)
        )
        dh_loss = self.criterion(
            nn_outputs["d_heading"], labels[:, 2].view(labels.shape[0], 1)
        )
        total_loss = dx_loss + dy_loss + dh_loss

        return total_loss, dx_loss, dy_loss, dh_loss


def save_losses(
    train_total_losses,
    train_total_loss,
    train_dx_losses,
    train_dx_loss,
    train_dy_losses,
    train_dy_loss,
    test_total_losses,
    test_total_loss,
    test_dx_losses,
    test_dx_loss,
    test_dy_losses,
    test_dy_loss,
):

    train_total_losses.append(train_total_loss)
    train_dx_losses.append(train_dx_loss)
    train_dy_losses.append(train_dy_loss)
    test_total_losses.append(test_total_loss)
    test_dx_losses.append(test_dx_loss)
    test_dy_losses.append(test_dy_loss)

    return (
        train_total_losses,
        train_dx_losses,
        train_dy_losses,
        test_total_losses,
        test_dx_losses,
        test_dy_losses,
    )


def train(
    model,
    dataset_path,
    checkpoint_path,
    cache=False,
    lr=0.001,
    num_epochs=100,
    batch_size=32,
    save_steps=30,
):
    """
    Training main method
    :param model: the network
    """

    model = model.to(device)

    train_total_losses, train_dx_losses, train_dy_losses = ([], [], [])
    test_total_losses, test_dx_losses, test_dy_losses = ([], [], [])
    epochs = []

    loss_function = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = create_datasets(
        dataset_path, save_path=checkpoint_path, cache=cache, batch_size=batch_size
    )  # read datasets

    # train
    save_step = 0

    for epoch in range(num_epochs):
        save_step += 1
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        epochs.append(epoch)
        train_total_loss, train_dx_loss, train_dy_loss = train_epoch(
            model, loss_function, optimizer, train_loader
        )
        test_total_loss = 0
        test_dx_loss = 0
        test_dy_loss = 0

        (
            train_total_losses,
            train_dx_losses,
            train_dy_losses,
            test_total_losses,
            test_dx_losses,
            test_dy_losses,
        ) = save_losses(
            train_total_losses,
            train_total_loss,
            train_dx_losses,
            train_dx_loss,
            train_dy_losses,
            train_dy_loss,
            test_total_losses,
            test_total_loss,
            test_dx_losses,
            test_dx_loss,
            test_dy_losses,
            test_dy_loss,
        )
        plt.ylim(0, 1)
        plt.plot(epochs, train_total_losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train Loss")
        plt.savefig("train_losses")
        plt.clf()

        # save model
        if save_step == save_steps:
            save_step = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss_history": train_total_loss,
                },
                os.path.join(checkpoint_path, "ckpt_{}".format(epoch)),
            )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss_history": train_total_loss,
        },
        os.path.join(checkpoint_path, "ckpt_{}".format(num_epochs)),
    )


def train_epoch(model, loss_function, optimizer, data_loader):
    """Train for a single epoch"""
    dx_losses = 0.0
    dy_losses = 0.0
    dh_losses = 0.0
    current_loss = 0.0
    current_acc = 0

    model.train()

    for i, (inputs, labels) in enumerate(data_loader):
        inputs[0] = inputs[0].to(device)
        inputs[1] = inputs[1].to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            # forward
            outputs = model(inputs)
            labels = labels.float()
            loss, dx_l, dy_l, dh_l = model.compute_loss(outputs, labels)

            # backward
            loss.backward()
            optimizer.step()

        current_loss += loss.item() * inputs[0].size(0)
        dx_losses += dx_l.item() * inputs[0].size(0)
        dy_losses += dy_l.item() * inputs[0].size(0)
        dh_losses += dh_l.item() * inputs[0].size(0)

    total_loss = current_loss / len(data_loader.dataset)
    total_dx_loss = dx_losses / len(data_loader.dataset)
    total_dy_loss = dy_losses / len(data_loader.dataset)
    total_dh_loss = dh_losses / len(data_loader.dataset)

    print("Train Loss: {:.4f}".format(total_loss))
    print("Train dx ;oss: {:.4f}".format(total_dx_loss))
    print("Train dy loss: {:.4f}".format(total_dy_loss))
    print("Train dh loss: {:.4f}".format(total_dh_loss))

    return total_loss, total_dx_loss, total_dy_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        default='/offline_dataset',
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_path",
        default="/output",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--cache",
        default=False,
        type=bool,
        required=False,
    )
    parser.add_argument(
        "--save_steps",
        default=30,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--learning_rate",
        default=0.001,
        type=float,
        required=False,
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--num_epochs",
        default=100,
        type=int,
        required=False,
    )
    args = parser.parse_args()

    print("Training...")
    m = MainNet()
    train(
        m,
        args.dataset_path,
        args.output_path,
        cache=args.cache,
        lr=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        save_steps=args.save_steps,
    )
    print("Training Done.")


if __name__ == "__main__":
    main()

    # python train.py --dataset-path /data/training
