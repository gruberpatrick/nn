import numpy as np
import matplotlib.pyplot as plt

from data.test_datasets import load_radial_easy
from nn.layer import Linear
from nn.activation import Sigmoid
from nn.model import Sequential


if __name__ == '__main__':

    X, Y = load_radial_easy()

    plt.scatter(X[0, :], X[1, :], c=Y.reshape(400,), s=40, cmap=plt.cm.viridis)
    plt.show()

    model_definition = [
        Linear(2, 4),
        Sigmoid(),
        Linear(4, 1),
        Sigmoid(),
    ]

    model = Sequential(model_definition)
    try:
        model.train(X, Y, n_h=4, epochs=100000, print_loss=True)
    except KeyboardInterrupt:
        pass

    heatmap = np.mgrid[-1:1:200j,-1:1:200j]
    heatmap_x = heatmap[0].flatten()
    heatmap_y = heatmap[1].flatten()
    heatmap = np.array([heatmap_x, heatmap_y])
    heatmap_class = model.forward(heatmap)

    print(heatmap_class[0][0])
    print(heatmap_class[-1][-1])

    plt.scatter(heatmap[0, :], heatmap[1, :], c=heatmap_class.reshape(40000,), s=40, cmap=plt.cm.Spectral, label="Heatmap")
    plt.scatter(X[0, :], X[1, :], c=Y.reshape(400,), s=40, cmap=plt.cm.viridis)
    plt.show()
