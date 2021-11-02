import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

class GeoDataset(torch.utils.data.Dataset):
    '''
    Class to create a torch.utils.data.Dataset to be fed inside a DataLoader
    Args:
        - data: data to be stored
        - transforms: the set of torchvision.transforms to be applied on data when __getitem__ is called
    '''
    def __init__(self, data, transforms=None):
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = Image.fromarray((self.data[idx] * 255).astype(np.uint8))

        if self.transforms:
            sample = self.transforms(sample)

        return sample.to(device)


class ToTensor(object):
    """
    Method to convert ndarrays in sample to Tensors
    """

    def __call__(self, x):
        return torch.from_numpy(np.expand_dims(np.asarray(x).astype('float32') / 255., axis=0)).type(torch.FloatTensor)


def to_loaders(train_X, val_X, batch_train=32, batch_val=32, transforms_fn=True):
    '''
    Method to convert data structures to DataLoaders in PyTorch
    Args:
    - train_X, val_X: training and validation data structures

    Returns:
    - training and validation data loaders
    '''
    if hasattr(train_X, 'values'):
        train_X = train_X.values

    if hasattr(val_X, 'values'):
        val_X = val_X.values

    if transforms_fn:
        tfs = torchvision.transforms.Compose([
            torchvision.transforms.Resize((240, 480)),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomRotation(10),
            ToTensor()])
    else:
        tfs = torchvision.transforms.Compose([
            torchvision.transforms.Resize((240, 480)),
            ToTensor()])

    train_loader = torch.utils.data.DataLoader(
        GeoDataset(train_X, transforms=tfs), batch_size=batch_train, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        GeoDataset(val_X, transforms=tfs),
        batch_size=batch_val, shuffle=False)

    return train_loader, val_loader


def train(model, optimizer, train_loader, val_loader, epoch, est = None, args = None):
    '''
    Function to train a generic VAE
    Args:
        model: the instance of the torch model
        optimizer: the instance of the torch.optim
        train_loader: torch.utils.DataLoader containing training data
        val_loader: torch.utils.DataLoader containing validation data
        epoch: current epoch in the training procedure
        est: an external estimator to assess VAE performances (typically sklearn.BaseEstimator)
        args: a set of args to pass
    '''
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        # Run VAE
        recon_batch, mu, logvar = model(data)

        # Compute loss
        rec, kl = model.loss_function(recon_batch, data, mu, logvar)

        if est is not None:
            est = est.partial_fit(mu.detach().cpu())
            assignments = est.predict(mu.detach().cpu())
            centroids = torch.from_numpy(est.cluster_centers_).type(torch.FloatTensor).to(device)
            est_loss = torch.nn.MSELoss()(mu, centroids[assignments])

            total_loss = rec + kl + est_loss
        else:
            est_loss = torch.Tensor([0])
            total_loss = rec + kl
        total_loss.backward()
        train_loss += total_loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('\r', end='')
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tMSE: {:.6f}\tKL: {:.6f}\tlog_sigma: {:f}\tClustering: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                           rec.item() / len(data),
                           kl.item() / len(data),
                    model.log_sigma,
                           est_loss.item() / len(data)), end='')

    # Plot reconstructions
    n = min(data.size(0), 8)
    data = next(iter(val_loader))
    data = data.to(device)
    recon_batch, mu, logvar = model(data)
    comparison = torch.cat([data[:n], recon_batch.view(64, -1, 240, 480)[:n]])
    comparison = torchvision.utils.make_grid(comparison)
    print("Reconstructions: ")
    plt.imshow(comparison.detach().cpu().numpy().transpose(1, 2, 0))
    plt.show()

    train_loss /= len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))

def train_mine(mine, model, mine_loss, optimizer_mine, train_loader, val_loader, epoch):
    '''
    Function to train a Mutual Information Neural Estimator (MINE) model
    Args:
        mine: the instance of the MINE model
        model: the instance of the VAE to recreate the latent space
        mine_loss: the loss function evaluating the objective of MINE
        optimizer_mine: the instance of the torch.optim
        train_loader: tprch.utils.DataLoader containing training data
        val_loader: torch.utils.DataLoader containing validation data
        epoch: the epoch number of the training procedure
    '''
    mine.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer_mine.zero_grad()

        with torch.no_grad():
            recon_batch, mu, logvar = model(data)
            idx = torch.randperm(mu.shape[0])
            mu_shuffle = mu[idx].view(mu.size())

        mi = mine_loss(mine(data, mu), mine(data, mu_shuffle))
        mi.backward()
        train_loss += mi.item()
        optimizer_mine.step()

        print('\r', end='')
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tMutual Info Training: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader),
                   train_loss / (batch_idx + 1)),
            end='')
    return train_loss / len(train_loader)