import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MINE(nn.Module):
    def __init__(self, x_dim, z_dim):
        super(MINE, self).__init__()
        self.expand = nn.Linear(z_dim, 400)
        self.compress = nn.Linear(x_dim, 400)
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(800, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 1))

    def forward(self, x, z):
        x, z = self.compress(x.flatten(start_dim=1)), self.expand(z)
        return self.layers(torch.cat([x, z], dim=1))


def mine_loss(mi, mi_m):
    '''
    Args:
        mi: the output of MINE considering data drawn from the joint distribution
        mi_m: the output of MINE considering data drawn from the marginal distributions

    Returns:
    - the variational loss converging to true mutual information
    '''
    return - (mi.mean() - mi_m.exp().mean().log())
