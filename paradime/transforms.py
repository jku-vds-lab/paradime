from .types import Tensor

class DissimilarityTransform():

    def __init__(self):
        pass

    def __call__(self, X: Tensor) -> Tensor:
        pass

class Identity(DissimilarityTransform):
    def __call__(self, X: Tensor) -> Tensor:
        return X

class PerplexityBased(DissimilarityTransform):
    
    def __init__(self,
        perp: float = 30
        ) -> None:

        self.perplexity = perp