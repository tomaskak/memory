from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Dense, Lambda
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Mean, CategoricalAccuracy
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
    broadcast_to,
    float32,
    transpose,
    argmax,
    abs,
    cast,
)
from tensorflow.math import argmax


class Encoder(Layer):
    def __init__(self, sizes, activation: str):
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
    def __init__(self, sizes, activation: str, classification: bool = False):
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
        encoder_sizes,
        decoder_sizes,
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
            [memory_decay ** n for n in range(memory_length)], dtype="float32"
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


def run_encoder(me, memory, x):
    return me.encoder(concat((memory, x), axis=-1))


def query_new_memory(
    me, correct_labels, encoded, mem_size, past_states, unseen_states, loss_fn
):
    correct_inputs = concat(
        (broadcast_to(encoded, (me.memory_length, mem_size)), past_states),
        axis=-1,
    )
    incorrect_inputs = concat(
        (broadcast_to(encoded, (me.memory_length, mem_size)), unseen_states),
        axis=-1,
    )
    correct_in_preds = me.decoder(correct_inputs)
    incorrect_in_preds = me.decoder(incorrect_inputs)
    # correct_labels = broadcast_to(constant([1.0, 0.0]), (me.memory_length, 2))
    incorrect_labels = broadcast_to(constant([0.0, 1.0]), (me.memory_length, 2))
    loss = loss_fn(correct_in_preds, correct_labels)
    loss += loss_fn(incorrect_in_preds, incorrect_labels)

    return (
        loss,
        (correct_in_preds, correct_labels),
        (incorrect_in_preds, incorrect_labels),
    )


@function(jit_compile=True)
def train_segment(
    me,
    X,
    Xnoise,
    iterations,
    past_states,
    unseen_states,
    memory,
    loss_fn,
    optimizer,
    loss_metric,
    accuracy_metric,
):
    cumulative_loss = 0.0
    x_idx = 0

    for i in range(iterations):
        x = X[x_idx]

        correct_labels = me.decoder(
            concat(
                (
                    broadcast_to(memory, (me.memory_length - 1, me.memory_size)),
                    past_states[i : i + me.memory_length - 1],
                ),
                axis=-1,
            )
        )
        correct_labels = transpose(
            concat(
                (
                    expand_dims(argmax(correct_labels, axis=1), axis=0),
                    expand_dims(abs(argmax(correct_labels, axis=1) - 1), axis=0),
                ),
                axis=0,
            )
        )
        correct_labels = concat(
            (correct_labels, constant([[1, 0]], dtype="int64")), axis=0
        )
        correct_labels = cast(correct_labels, dtype=float32)

        with GradientTape(persistent=True) as tape:
            # Update memory after generating new memory
            x = expand_dims(x, axis=0)
            encoded = run_encoder(me, memory, x)

            loss, correct_outputs, incorrect_outputs = query_new_memory(
                me,
                correct_labels,
                encoded,
                me.memory_size,
                past_states[i : i + me.memory_length],
                unseen_states[i : i + me.memory_length],
                loss_fn,
            )
            memory.assign(encoded)

        loss_metric(loss)
        accuracy_metric(*correct_outputs)
        accuracy_metric(*incorrect_outputs)

        gradient = tape.gradient(loss, me.trainable_variables)
        optimizer.apply_gradients(zip(gradient, me.trainable_variables))

        x_idx += 1

        cumulative_loss += loss
    return cumulative_loss


@function
def train_episode(
    me,
    X,
    Xnoise,
    loss_fn,
    optimizer,
    memory,
    x_len,
    loss_metric=lambda *x, **y: None,
    accuracy_metric=lambda *x, **y: None,
):
    cumulative_loss = 0.0

    # The number of iterations to run at a time is limited by GPU registers, XLA compilation
    # will flatten the loop causing intermediate values to be stored in registers generating
    # a warning.
    idx = 0
    iterations = 8
    for i in range(x_len // iterations):
        replay_idx = idx - me.memory_length if idx > me.memory_length else 0
        replay_end = replay_idx + me.memory_length + iterations
        current_loss = train_segment(
            me,
            X[idx : idx + iterations],
            Xnoise[idx : idx + iterations],
            iterations,
            X[replay_idx:replay_end],
            Xnoise[replay_idx:replay_end],
            memory,
            loss_fn,
            optimizer,
            loss_metric,
            accuracy_metric,
        )
        cumulative_loss += current_loss
        idx += iterations

    return cumulative_loss


def train_memory(
    me: MemoryEncoder,
    X,
    Xnoise,
    steps_in_episode: int,
    loss_fn,
    optimizer,
    log_metric_fn=lambda *x, **y: None,
):
    assert (
        steps_in_episode > me.memory_length
    ), "less steps in episode then memory length."

    memory = Variable(zeros(shape=(1, me.memory_size)), trainable=False)
    loss_metric = Mean("train_loss", dtype=float32)
    accuracy_metric = CategoricalAccuracy("train_accuracy")

    losses = []
    sie = steps_in_episode
    for i in range(len(X) // steps_in_episode):
        reset(me, memory)

        loss = train_episode(
            me,
            X[i * sie : (i + 1) * sie],
            Xnoise[i * sie : (i + 1) * sie],
            loss_fn,
            optimizer,
            memory,
            sie,
            loss_metric,
            accuracy_metric,
        )
        losses.append(loss / sie)

        log_metric_fn(loss_metric, accuracy_metric, episode=i)

        loss_metric.reset_states()
        accuracy_metric.reset_states()

    compiled_fns = train_segment.experimental_get_tracing_count()
    compiled_fns += train_segment.experimental_get_tracing_count()
    print("number of JIT retracings on training functions: {}".format(compiled_fns))

    return losses


def reset(me, memory):
    memory.assign(zeros(shape=(1, me.memory_size)))
