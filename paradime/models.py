"""Default models for ParaDime.

The :mod:`paradime.models` module implements default models for several
applications, such as a simple
:class:`paradime.models.FullyConnectedEmbeddingModel`.
"""

import torch


class Model(torch.nn.Module):
    """A placeholder model that lists all methods used by the losses defined
    in :mod:`paradime.loss`.
    """

    def embed(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def decode(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def classify(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class FullyConnectedEmbeddingModel(Model):
    """A fully connected network for embeddings.

    Args:
        in_dim: Input dimension.
        out_dim: Output dimensions.
        hidden_dims: List of hidden layer dimensions.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dims: list[int]):
        super().__init__()

        self.layers = torch.nn.ModuleList()

        cur_dim = in_dim
        for hdim in hidden_dims:
            self.layers.append(torch.nn.Linear(cur_dim, hdim))
            cur_dim = hdim
        self.layers.append(torch.nn.Linear(cur_dim, out_dim))

        self.alpha = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        for layer in self.layers[:-1]:
            # x = torch.sigmoid(layer(x))
            x = torch.nn.Softplus()(layer(x))
        out = self.layers[-1](x)
        return out

    def embed(self, x):
        return self.forward(x)
