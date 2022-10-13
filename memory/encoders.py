from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Dense, Lambda
from tensorflow.keras.losses import Loss
from tensorflow import (
    random,
    GradientTape,
    concat,
    zeros,
    constant,
    expand_dims,
    TensorArray,
    function,
    Variable,
)
from tensorflow.math import argmax


class Encoder(Layer):
    def __init__(self, sizes: int | list, activation: str):
        super(Encoder, self).__init__()
        if isinstance(sizes, int):
            sizes = [sizes]
        self._layers = Sequential(
            [
                Dense(sz, activation=activation, name="Encoder" + str(i))
                for i, sz in enumerate(sizes)
            ]
        )

    def __call__(self, inputs, **kwargs):
        y = self._layers(inputs, **kwargs)
        return y


class Decoder(Layer):
    def __init__(
        self, sizes: int | list, activation: str, classification: bool = False
    ):
        super(Decoder, self).__init__()
        if isinstance(sizes, int):
            sizes = [sizes]
        if classification:
            self._layers = Sequential(
                [
                    Dense(sz, activation=activation, name="Decoder" + str(i))
                    for i, sz in enumerate(sizes[:-1])
                ]
                + [Dense(sizes[-1], activation="softmax")]
            )
        else:
            self._layers = Sequential(
                [
                    Dense(sz, activation=activation, name="Decoder" + str(i))
                    for i, sz in enumerate(sizes)
                ]
            )

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


class MemoryEncoder(Layer):
    """
    A single Encoder trained on a decoder that learns to interpret from the encoder whether or not
    a given observation X has been seen by the Encoder.

    The goal is to have the Encoder learn a latent form for the data that contains the information
    of whether or not X has occurred, thus being a form of memory.

    M_t = Memory at time t
    X_t = Observation at time t
    XQ = Query indicating the question "has X been seen?"

    M_t __
          \ __ Encoder __ M_t+1 __  Decoder __ Y/N
        __/                  XQ __/
    X_t

    The output of the Encoder on the previous step is intended to be passed as part of the input to the next step
    with the Encoder learning how to add X to the current Memory.

    This class holds the pieces of the encoder/decoder relationship but contains no training states as it is to
    be passed into a tf.function and must be compiled.
    """

    def __init__(
        self,
        observation_size: int,
        encoder_sizes: list = list([300, 300]),
        activation: str = "relu",
        memory_size: int = 200,
        memory_length: int = 10,
        memory_decay: float = 0.9,
    ):
        """
        args:
            encoder_sizes: a list of integers indicating the number of units in each hidden layer of the Encoder. Does not
                           include final layer.
            memory_size:   determines the length of the Memory size or last layer of the Encoder.
            activation:    activation fns on each layer in Encoder and Decoder.
            memory_length: how many inputs into the past the Encoder should try to remember.
            memory_decay:  this factor reduces the loss for each step into the past the given state occurred.

        """
        super(MemoryEncoder, self).__init__()
        self._encoder = Encoder(encoder_sizes + [memory_size], activation)
        self._decoder = Decoder(
            encoder_sizes + [2], activation=activation, classification=True
        )

        self._decays = constant(
            [memory_decay**n for n in range(memory_length)], dtype="float32"
        )
        self._mem_decay = memory_decay
        self._obs_size = observation_size
        self._mem_size = memory_size
        self._mem_length = memory_length

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    @property
    def observation_size(self):
        return self._obs_size

    @property
    def memory_size(self):
        return self._mem_size

    @property
    def memory_length(self):
        return self._mem_length

    @property
    def memory_loss_decay(self):
        return self._mem_decay


@function
def can_run(memory_encoder, M, X):
    encoded = memory_encoder.encoder(concat((M, X), axis=-1))
    return memory_encoder.decoder(concat((encoded, X), axis=-1))


@function
def train_episode(
    me, X, Xnoise, loss_fn, optimizer, memory, past_states, unseen_states
):
    q_idx = 0
    x_idx = 0
    cumulative_loss = 0.0
    # zip not allowed in compiled function
    for x in X:
        x = expand_dims(x, axis=0)
        past_states[q_idx % me.memory_length].assign(x)
        unseen_states[q_idx % me.memory_length].assign(
            expand_dims(Xnoise[x_idx], axis=0)
        )

        with GradientTape() as tape:
            # Update memory after generating new memory
            encoded = me.encoder(concat((memory, x), axis=-1))
            memory.assign(encoded)

            loss = 0.0
            # test the states loaded into memory within the goal memory_length
            for query in past_states[
                : q_idx + 1 if q_idx + 1 < me.memory_length else me.memory_length
            ]:
                query = expand_dims(query, axis=0)
                loss += (0.7) * loss_fn(
                    me.decoder(concat((encoded, query), axis=-1)),
                    constant([[1.0, 0.0]]),
                )
            # test states not seen
            for query in unseen_states[
                : q_idx + 1 if q_idx + 1 < me.memory_length else me.memory_length
            ]:
                query = expand_dims(query, axis=0)
                loss += loss_fn(
                    me.decoder(concat((encoded, query), axis=-1)),
                    constant([[0.0, 1.0]]),
                )
        gradient = tape.gradient(loss, me.trainable_variables)
        optimizer.apply_gradients(zip(gradient, me.trainable_variables))
        q_idx += 1
        x_idx += 1

        cumulative_loss += loss

    return cumulative_loss


def train_memory(
    me: MemoryEncoder, X, Xnoise, steps_in_episode: int, loss_fn, optimizer
):
    memory = Variable(zeros(shape=(1, me.memory_size)), trainable=False)
    past_states = Variable(zeros(shape=(me.memory_length, me.observation_size)))
    unseen_states = Variable(zeros(shape=(me.memory_length, me.observation_size)))

    losses = []
    sie = steps_in_episode
    for i in range(len(X) // steps_in_episode):
        loss = train_episode(
            me,
            X[i * sie : (i + 1) * sie],
            Xnoise[i * sie : (i + 1) * sie],
            loss_fn,
            optimizer,
            memory,
            past_states,
            unseen_states,
        )
        losses.append(loss)
        reset(me, memory, past_states, unseen_states)
    return losses


def reset(me, memory, past_states, unseen_states):
    memory.assign(zeros(shape=(1, me.memory_size)))
    past_states.assign(zeros(shape=(me.memory_length, me.observation_size)))
    unseen_states.assign(zeros(shape=(me.memory_length, me.observation_size)))
