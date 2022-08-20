import logging
import sys
sys.path.append('../')
from matplotlib import pyplot as plt

# start-include-and-data
from sklearn import manifold
import torch
import torch.nn.functional as F
import torchvision

from paradime import dr as pddr
from paradime import relations as pdrel
from paradime import transforms as pdtf
from paradime import loss as pdloss
from paradime import utils as pdutils

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(
    root='../data',
    train=True,
    download=True,
    transform=transform
)

testset = torchvision.datasets.CIFAR10(
    root='../data',
    train=False,
    download=True,
    transform=transform
)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


num_items = 5000
# end-include-and-data

class FullyConnectedEmbeddingModel(torch.nn.Module):
    def __init__(self,
        in_dim: int,
        out_dim: int,
        hidden_dims: list[int]
    ):
        super().__init__()
        
        self.layers = torch.nn.ModuleList()
        
        cur_dim = in_dim
        for hdim in hidden_dims:
            self.layers.append(torch.nn.Linear(cur_dim, hdim))
            cur_dim = hdim
        self.layers.append(torch.nn.Linear(cur_dim, out_dim))

        self.alpha = torch.nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        for layer in self.layers[:-1]:
            # x = torch.sigmoid(layer(x))
            x = torch.nn.Softplus()(layer(x))
        out = self.layers[-1](x)
        return out

    def embed(self, x):
        return self.forward(x)