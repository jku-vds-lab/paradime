import logging
import sys
sys.path.append('../')
from matplotlib import pyplot as plt

# start-include-and-data
import paradime.routines
import paradime.utils
import torchvision

mnist = torchvision.datasets.MNIST(
    '../data',
    train=True,
    download=True,
)
mnist_data = mnist.data.reshape(-1, 28*28) / 255.
num_items = 5000
# end-include-and-data

paradime.utils.logging.set_logfile('logs/predefined.log', 'w')

# start-define
dr = paradime.routines.ParametricTSNE(
    perplexity=100,
    dataset=mnist_data[:num_items],
    epochs=40,
    use_cuda=True,
    verbose=True,
)
# end-define

# start-train
dr.train()
# end-train

# start-apply-to-train-set
reduced = dr.apply(mnist_data[:num_items])
# end-apply-to-train-set

# start-plot-train
paradime.utils.plotting.scatterplot(reduced, mnist.targets[:num_items])
# end-plot-train

plt.savefig('images/predefined-1.png',
    facecolor='#fcfcfc',
    edgecolor='None',
    bbox_inches='tight',
)

# start-apply-and-plot-rest
paradime.utils.plotting.scatterplot(dr.apply(mnist_data), mnist.targets)
# end-apply-and-plot-rest

plt.savefig('images/predefined-2.png',
    facecolor='#fcfcfc',
    edgecolor='None',
    bbox_inches='tight',
)