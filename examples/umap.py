import logging
import sys
sys.path.append('../')
from matplotlib import pyplot as plt

# start-include-and-data
from sklearn import manifold
import torch
import torchvision

from paradime import dr as pddr
from paradime import relations as pdrel
from paradime import transforms as pdtf
from paradime import loss as pdloss
from paradime import utils as pdutils

mnist = torchvision.datasets.MNIST(
    '../data',
    train=True,
    download=True,
)
mnist_data = mnist.data.reshape(-1, 28*28) / 255.
num_items = 5000
# end-include-and-data

pdutils.logging.set_logfile('logs/umap.log', 'w')

# start-global-rel
global_rel = pdrel.NeighborBasedPDist(
    transform=[
        pdtf.ConnectivityBasedRescale(n_neighbors=30),
        pdtf.Symmetrize(subtract_product=True),
    ]
)
# end-global-rel

# start-batch-rel
batch_rel = pdrel.DistsFromTo(
    transform=[
        pdtf.ModifiedCauchyTransform(
            min_dist=0.1,
            spread=1,
        )
    ]
)
# end-batch-rel

# start-setup-dr
pumap = pddr.ParametricDR(
    dataset=mnist_data[:num_items],
    global_relations=global_rel,
    batch_relations=batch_rel,
    use_cuda=True,
    verbose=True,
)
# end-setup-dr

# start-compute-rel
pumap.compute_global_relations()
# end-compute-rel

# start-add-spectral
affinities = pumap.global_relation_data['rel'].to_square_array().data
spectral = manifold.SpectralEmbedding(
    affinity='precomputed'
).fit_transform(affinities)
spectral = (spectral - spectral.mean(axis=0)) / spectral.std(axis=0)

pumap.add_to_dataset({
    'spectral': spectral
})
# end-add-spectral

# start-setup-init
init_phase = pddr.TrainingPhase(
    name='spectral_init',
    epochs=10,
    batch_size=500,
    loss=pdloss.PositionLoss(position_key='spectral'),
    learning_rate=0.01,
    report_interval=2,
)
# end-setup-init

# start-setup-main
main_phase = pddr.TrainingPhase(
    name='main_embedding',
    epochs=60,
    batches_per_epoch=50,
    batch_size=100,
    sampling='negative_edge',
    neg_sampling_rate=5,
    loss=pdloss.RelationLoss(loss_function=pdloss.cross_entropy_loss),
    learning_rate=0.001,
    report_interval=5,
)
# end-setup-main

# start-training
pumap.add_training_phase(init_phase)
pumap.add_training_phase(main_phase)
pumap.train()
# end-training

# start-plot-train
reduced = pumap.apply(mnist_data[:num_items])
pdutils.plotting.scatterplot(reduced, mnist.targets[:num_items])
# end-plot-train

plt.savefig('images/umap-1.png',
    facecolor='#fcfcfc',
    edgecolor='None',
    bbox_inches='tight',
)

# start-plot-rest
pdutils.plotting.scatterplot(pumap.apply(mnist_data), mnist.targets)
# end-plot-rest

plt.savefig('images/umap-2.png',
    facecolor='#fcfcfc',
    edgecolor='None',
    bbox_inches='tight',
)
