import numpy as np
from nn.layer import Layer


class Initializer:

    def init_layer(self, layer):

        raise Exception("Function `init_layer` not implemented.")


class XavierNormal(Initializer):

    def init_layer(self, layer, fan_type="fan_avg"):
        """
        Source: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        """

        if not isinstance(layer, Layer):
            raise Exception("Provided parameter is not a layer.")

        fan_in = layer._in_size
        fan_out = layer._out_size

        if fan_type == "fan_in":
            fan_divisor = fan_in
        elif fan_type == "fan_out":
            fan_divisor = fan_out
        else:
            fan_divisor = (fan_in + fan_out) / 2

        stdev = np.sqrt(1 / fan_divisor)

        layer._W = np.random.normal(loc=0, scale=stdev, size=(fan_out, fan_in))


class HeNormal(Initializer):

    def init_layer(self, layer):
        """
        Source: https://arxiv.org/pdf/1502.01852.pdf
        """

        if not isinstance(layer, Layer):
            raise Exception("Provided parameter is not a layer.")

        fan_in = layer._in_size
        fan_out = layer._out_size

        stdev = np.sqrt(1 / fan_in)
        layer._W = np.random.normal(loc=0, scale=stdev, size=(fan_out, fan_in))
