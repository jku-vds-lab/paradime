import torch

class Encoder(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def encode(self,
        X: torch.Tensor
        ) -> torch.Tensor:

        return self.forward()

class Decoder(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def decode(self,
        X: torch.Tensor
        ) -> torch.Tensor:

        return self.forward()
