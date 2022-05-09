class Dissimilarity():
    pass
class Gaussian():
    pass
class PerplexityRescaler():
    pass
class ParametricDR():
    pass
class TSNESymmetrizer():
    pass
class StudentT():
    pass
class EmbeddingLoss():
    pass
class CompoundLoss():
    pass
class MSELoss():
    pass
class ClassificationLoss():
    pass
class Encoder():
    pass
class ReconstructionLoss():
    pass
class ShiftedGaussian():
    pass
class ConnectivityRescaler():
    pass
class EuclideanDistance():
    pass
class UMAPSymmetrizer():
    pass
class RescaledStudentT():
    pass
class PrecomputedDistances():
    pass

perp = 200
hdim = 20
ldim = 2
k = 50
a = 1.
b = 0.1
distmat = [[0.1, 0.4], [0.2, 0.3]]


import torch.nn as nn

tsne_diss_hd = Dissimilarity(
    metric = Gaussian(),
    method = 'exact',
    rescale = PerplexityRescaler(perp),
    symmetrize = TSNESymmetrizer(
        normalize = False)
  )

tsne_diss_ld = Dissimilarity(
      metric = StudentT()
  )

tsne_loss = EmbeddingLoss(
      metric = nn.KLDivLoss()
)

tsne = ParametricDR(
    hd_dissimilarity = tsne_diss_hd,
    ld_dissimilarity = tsne_diss_ld,
    loss = tsne_loss
)

tsne_loss_super = CompoundLoss([
    EmbeddingLoss(
        metric = nn.KLDivLoss()
    ),
    ClassificationLoss(
        metric = nn.CrossEntropyLoss()
    ),
    ReconstructionLoss(
        metric = nn.MSELoss()
    )
],
    weights = [0.5, 0.3, 0.2]
)

tsne_super = ParametricDR(
    hd_dissimilarity = tsne_diss_hd,
    ld_dissimilarity = tsne_diss_ld,
    loss = tsne_loss_super
)

linearized_tsne = ParametricDR(
    hd_dissimilarity = tsne_diss_hd,
    ld_dissimilarity = tsne_diss_ld,
    loss = tsne_loss,
    encoder = Encoder(
        embedder = nn.Linear(hdim, ldim)
    )
)


umap_diss_hd = Dissimilarity(
    metric = ShiftedGaussian(),
    method = 'approximate',
    rescale = ConnectivityRescaler(k),
    symmetrize = UMAPSymmetrizer()
  )

umap_diss_ld = Dissimilarity(
    metric = RescaledStudentT(a, b)
  )

umap_loss = EmbeddingLoss(
    metric = nn.CrossEntropyLoss()
)

umap = ParametricDR(
    hd_dissimilarity = umap_diss_hd,
    ld_dissimilarity = umap_diss_ld,
    loss = umap_loss
)


mds = ParametricDR(
    hd_dissimilarity = Dissimilarity(
        metric = PrecomputedDistances(distmat)
    ),
    ld_dissimilarity = Dissimilarity(
        metric = EuclideanDistance()
    ),
    loss = EmbeddingLoss(
        metric = MSELoss()
    )
)