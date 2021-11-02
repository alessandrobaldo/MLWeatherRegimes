import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from PIL import Image

class Sampler(nn.Module):
    def __init__(self):
        super(Sampler, self).__init__()

    def forward(self, x):
        mu, cov = x
        return mu + torch.exp(0.5 * cov) * torch.randn_like(cov)


class Block(nn.Module):
    def __init__(self, in_channels=1, filters=(32, 64, 128, 128)):
        super(Block, self).__init__()

        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels, filters[-1], kernel_size=9),
            nn.ReLU()
        )

        seq = []
        for i, f in enumerate(filters):
            seq.append(nn.Conv2d(in_channels, f, kernel_size=3))
            seq.append(nn.ReLU())
            in_channels = f
        self.seq2 = nn.Sequential(*seq)

        self.seq3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3),
            nn.BatchNorm2d(filters[-1]),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.seq1(x)
        x2 = self.seq2(x)
        x3 = self.seq3(torch.add(x1, x2))

        return x3


class VariationalEncoder(nn.Module):
    def __init__(self):
        super(VariationalEncoder, self).__init__()

        self.block1 = Block(in_channels=1, filters=(16, 32, 32, 64))
        self.block2 = Block(in_channels=64, filters=(64, 128, 128, 256))
        self.relu = nn.ReLU()
        self.linear = nn.Linear(256 * 23 * 49, 16)
        self.linear_mu = nn.Linear(16, 10)
        self.linear_cov = nn.Linear(16, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        x = self.relu(x)
        mu = self.linear_mu(x)
        cov = self.linear_cov(x)

        return mu, cov


class VariationalDecoder(nn.Module):
    def __init__(self):
        super(VariationalDecoder, self).__init__()

        self.linear = nn.Linear(10, 7200)
        self.relu = nn.ReLU()
        self.conv2dtranspose = nn.ConvTranspose2d(1, 16, kernel_size=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        x = self.linear(z)
        x = self.relu(x)
        x = x.view((x.shape[0], 1, 60, 120))

        x = self.conv2dtranspose(x)
        x = x.view((x.shape[0], 1, 248, 488))
        x = F.interpolate(x, (241, 481))

        x = self.sigmoid(x)

        return x


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = VariationalEncoder()
        self.sampler = Sampler()
        self.decoder = VariationalDecoder()

    def forward(self, x):
        mu, cov = self.encoder(x)
        z = self.sampler((mu, cov))
        x_hat = self.decoder(z)

        return (mu, cov), x_hat


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Interpolate(nn.Module):
    def __init__(self, *args):
        super(Interpolate, self).__init__()
        self.shape = args

    def forward(self, x):
        return F.interpolate(x, self.shape)


class BaseVAE(nn.Module):
    def __init__(self):
        super(BaseVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(64 * 61 * 121, 16),
            nn.ReLU()
        )

        self.linear_mu = nn.Linear(16, 10)
        self.linear_cov = nn.Linear(16, 10)

        self.sampler = Sampler()

        self.decoder = nn.Sequential(
            nn.Linear(10, 64 * 61 * 121),
            nn.ReLU(),
            Reshape(-1, 64, 61, 121),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1),
            # Interpolate((241,481)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        mu, cov = self.linear_mu(x), self.linear_cov(x)
        z = self.sampler((mu, cov))
        x_hat = self.decoder(z)

        return (mu, cov), x_hat