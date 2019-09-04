import numpy as np
from nn.activation import Activation
from nn.loss import BinaryCrossEntropy
from nn.layer import Layer
from nn.initializer import XavierNormal


class Sequential:

    _cache = {}
    _explain = False

    def __init__(self, layers, initializer=XavierNormal()):

        self._layers = layers

        if initializer:
            for layer in self._layers:
                if isinstance(layer, Layer):
                    initializer.init_layer(layer)

    def forward(self, X):

        out = X
        self._cache["l0"] = out
        if self._explain:
            print("==========================")
            print("  l0")

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

    def train(self, X, Y, n_h, loss=BinaryCrossEntropy(), epochs=10000, print_loss=False):

        self._loss = loss

        for i in range(0, epochs):

            self.zero_grad()

            y_hat = self.forward(X)
            loss = self._loss.forward(y_hat, Y)

            self.backward(X, Y)
            self.optim_step(learning_rate=1.2)

            if self._explain:
                print("X.shape", X.shape)
                print("Y.shape", Y.shape)
                print("y_hat.shape", y_hat.shape)
                print("Aboring training to cut output.")
                return

            if print_loss and i % 1000 == 0:
                print("Loss after {:.2f}%: {}".format(i * 100 / epochs, loss))
