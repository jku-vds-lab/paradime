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
# end-include-and-data

logger = logging.getLogger('paradime')
fh = logging.FileHandler('logs/predefined.log')
logger.__format__

# start-define-and-train
dr = paradime.routines.ParametricTSNE(
    perplexity=50,
    dataset=mnist_data[:5000],
    batch_size=500,
    epochs=50,
    use_cuda=True,
    verbose=True,
)

dr.train()
# end-define-and-train

# start-apply-to-train-set
reduced = dr.apply(mnist_data[:5000])
# end-apply-to-train-set

# start-plot-train
paradime.utils.scatterplot(reduced, mnist.targets[:5000])
# end-plot-train

plt.savefig('images/predefined-1.png',
    facecolor='#fcfcfc',
    edgecolor='None',    
)

# start-apply-and-plot-rest
paradime.utils.scatterplot(dr.apply(mnist_data), mnist.targets)
# end-apply-and-plot-rest

plt.savefig('images/predefined-2.png',
    facecolor='#fcfcfc',
    edgecolor='None',    
)