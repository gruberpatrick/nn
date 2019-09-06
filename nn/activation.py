import numpy as np


class Activation:

    def forward(self, z):
        raise Exception("Function `forward` not implemented.")

    def backward(self, da, z, explain):
        raise Exception("Function `backward` not implemented.")


class Sigmoid(Activation):

    def forward(self, z):
        res = 1 / (1 + np.exp(-z))
        res[res == 1.0] = 0.99999
        res[res == 0.0] = 0.00001
        return res

    def backward(self, da, z, explain):
        if explain:
            print("dL/dZ = dL/dA * sig * (1 - sig)")
        sig = self.forward(z)
        return da * sig * (1 - sig)


class Tanh(Activation):

    def forward(self, z):
        return np.tanh(z)

    def backward(self, da, z, explain):
        return da * (1 - np.power(z, 2))


class ReLU(Activation):

    def forward(self, z):
        return np.maximum(0, z)

    def backward(self, da, z, explain):
        if explain:
            print("dL/dZ = dL/dA * `ReLU")
        dz = np.array(da, copy=True)
        dz[z <= 0] = 0
        return dz


class Linear(Activation):

    def forward(self, z):
        return z

    def backward(self, da, z, explain):
        return da
