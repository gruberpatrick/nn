import numpy as np


class Layer:

    def forward(self, z):
        raise Exception("Function `forward` not implemented.")

    def backward(self, da, z):
        raise Exception("Function `backward` not implemented.")


class Linear(Layer):

    def __init__(self, in_size, out_size):
        self._in_size = in_size
        self._out_size = out_size
        self._W = np.random.randn(out_size, in_size) * 0.01
        self._b = np.zeros((out_size, 1))
        self.reset_gradients()

    def reset_gradients(self):
        self._dw = None
        self._db = None

    def forward(self, X):
        return self._W.dot(X) + self._b

    def backward(self, dz, a, explain):
        if explain:
            print("dL/dW = dL/dZ * a")
            print("dL/db = dL/dZ * 1")
        m = a.shape[1]
        dw = np.dot(dz, a.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        self._dw = dw
        self._db = db
        return dw, db
