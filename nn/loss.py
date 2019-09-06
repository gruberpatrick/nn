import numpy as np


class Loss:

    def forward(self, z):
        raise Exception("Function `forward` not implemented.")

    def backward(self, da, z):
        raise Exception("Function `backward` not implemented.")


class BinaryCrossEntropy(Loss):

    def forward(self, a, Y):
        m = Y.shape[1]
        res = -(1 / m) * np.sum(np.dot(Y, np.log(a).T) + np.dot((1 - Y), np.log(1 - a).T))
        return np.squeeze(res)

    def backward(self, a, Y, explain):
        if explain:
            print("dL/dA = -(y/a) + ((1-y)/(1-a))")
        return -(Y / a) + ((1 - Y) / (1 - a))


class MeanSquaredError(Loss):

    def forward(self, a, Y):
        m = Y.shape[0]
        res = 1/(2*m) * np.sum(np.power(a - Y, 2))
        return np.squeeze(res)

    def backward(self, a, Y, explain):
        if explain:
            print("dL/dA = a - Y")
        return a - Y
