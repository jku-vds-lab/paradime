import torch

class Model(torch.nn.Module):
    """A placeholder model that has lists all methods used by the losses
    defined in :mod:`paradime.loss`.
    """

    def embed(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def decode(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def classify(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
