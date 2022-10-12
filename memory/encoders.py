from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Dense


class Encoder(Layer):
    def __init__(self, sizes: int | list, activation: str):
        super(Encoder, self).__init__()
        if isinstance(sizes, int):
            sizes = [sizes]
        self._layers = Sequential([Dense(sz, activation=activation) for sz in sizes])

    def __call__(self, inputs, **kwargs):
        y = self._layers(inputs, **kwargs)
        return y


class Decoder(Layer):
    def __init__(self, sizes: int | list, activation: str):
        super(Decoder, self).__init__()
        if isinstance(sizes, int):
            sizes = [sizes]
        self._layers = Sequential([Dense(sz, activation=activation) for sz in sizes])

    def __call__(self, inputs, **kwargs):
        y = self._layers(inputs, **kwargs)
        return y


class AutoEncoder(Model):
    def __init__(
        self,
        encoder_sizes: int | list,
        decoder_sizes: int | list,
        activation: str = "relu",
    ):
        super(AutoEncoder, self).__init__()
        self._encoder = Encoder(encoder_sizes, activation)
        self._decoder = Decoder(decoder_sizes, activation)

    def __call__(self, inputs, **kwargs):
        encoded = self._encoder(inputs, **kwargs)
        return self._decoder(encoded, **kwargs)
