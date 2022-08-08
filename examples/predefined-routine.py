import sys
sys.path.append('../')

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

# start-define-dr
dr = paradime.routines.ParametricTSNE(
    perplexity=100,
    dataset=mnist_data[:5000],
    batch_size=500,
    use_cuda=True,
    verbose=True,
)
# end-define-dr

#start-train
dr.train()
#end-train

# start-apply-to-train-set
reduced = dr.apply(mnist_data[:5000])
# end-apply-to-train-set

# start-plot-train
paradime.utils.scatterplot(reduced, mnist.targets[:5000])
# end-plot-train

# start-apply-and-plot-rest
paradime.utils.scatterplot(dr.apply(mnist_data), mnist.targets)
# end-apply-and-plot-rest