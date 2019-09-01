import numpy as np
import matplotlib.pyplot as plt
from activation import Sigmoid, Tanh, ReLU, Activation
from loss import CrossEntropy, Loss
from layer import Linear, Layer
from data.test_datasets import load_planar_dataset, load_easy_dataset


sigmoid = Sigmoid()
cross_entropy = CrossEntropy()
tanh = Tanh()
relu = ReLU()


class Sequential:

    _cache = {}
    _explain = False

    def __init__(self, layers):

        self._layers = layers

    def forward(self, X):

        out = X
        self._cache["l0"] = out

        for it in range(len(self._layers)):
            out = self._layers[it].forward(out)
            self._cache["l"+str(it+1)] = out
            if self._explain:
                print("  l"+str(it + 1))

        return out

    def backward(self, X, Y):

        layer_count = len(self._layers)

        if self._explain:
            print("==========================")
            print("  l"+str(layer_count))

        out = self._loss.backward(self._cache["l"+str(layer_count)], Y, self._explain)

        for it in range(len(self._layers) - 1, -1, -1):
            if self._explain:
                print("  l"+str(it))
            if isinstance(self._layers[it], Activation):
                out = self._layers[it].backward(out, self._cache["l"+str(it)], self._explain)
            elif isinstance(self._layers[it], Layer):
                self._layers[it].backward(out, self._cache["l"+str(it)], self._explain)
                if it > 0:
                    if self._explain:
                        print("dL/dA = dL/dZ * W")
                    out = np.dot(self._layers[it]._W.T, out)

    def optim_step(self, learning_rate=1.2):

        for layer in self._layers:
            if not isinstance(layer, Layer):
                continue
            W, b = layer._W, layer._b
            dW, db = layer._dw, layer._db
            layer._W = W - (learning_rate * dW)
            layer._b = b - (learning_rate * db)

    def zero_grad(self):

        for layer in self._layers:
            if not isinstance(layer, Layer):
                continue
            layer.reset_gradients()

    def train(self, X, Y, n_h, loss=CrossEntropy(), num_iterations=10000, print_cost=False):

        self._loss = loss

        for i in range(0, num_iterations):

            self.zero_grad()

            y_hat = self.forward(X)
            cost = self._loss.forward(y_hat, Y)

            self.backward(X, Y)
            self.optim_step(learning_rate=1.2)

            if self._explain:
                print("Aboring training to cut output.")
                exit()

            if print_cost and i % 1000 == 0:
                print("Cost after iteration %i: %f" % (i, cost))


if __name__ == '__main__':

    X, Y = load_easy_dataset()

    # plt.scatter(X[0, :], X[1, :], c=Y.reshape(400,), s=40, cmap=plt.cm.Spectral)
    # plt.show()

    model_definition = [
        Linear(2, 3),
        Sigmoid(),
        Linear(3, 1),
        Sigmoid(),
    ]

    model = Sequential(model_definition)
    model.train(X, Y, n_h=4, num_iterations=100000, print_cost=True)
