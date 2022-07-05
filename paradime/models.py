import torch

class Model(torch.nn.Module):

    def embed(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def decode(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def classify(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
