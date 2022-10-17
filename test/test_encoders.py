import pytest

import tensorflow as tf

from memory.encoders import AutoEncoder, MemoryEncoder, train_episode, train_memory


@pytest.mark.slow
@pytest.mark.train
def test_autoencoder():
    ae = AutoEncoder([4, 2], [4, 3], activation="linear")
    ae.compile(
        optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.1),
        loss=tf.keras.losses.MeanSquaredError(),
        jit_compile=True,
    )

    DATA = 100000
    X = tf.concat(
        (tf.random.normal((DATA, 2), 1.0, 2.0), tf.constant(6.0, shape=(DATA, 1))),
        axis=-1,
    )
    ae.fit(X, X, epochs=20, batch_size=32, validation_split=0.3)

    mse = tf.keras.losses.MeanSquaredError()
    x = tf.constant([[1.0, 2.0, 6.0]])
    assert mse(ae(x), x) < 1e-2


@pytest.mark.slow
@pytest.mark.train
def test_mem_encoder_trains_without_exception():
    MEM_SIZE = 100
    me = MemoryEncoder(3, [100, 100], memory_size=MEM_SIZE, memory_length=10)

    DATA = 50
    X = tf.random.normal((DATA, 3), 1.0, 2.0)
    Xnoise = tf.random.normal((DATA, 3), 0.0, 0.5)
    M = tf.zeros(shape=(1, MEM_SIZE))

    opt = tf.keras.optimizers.Adadelta(learning_rate=0.01)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    SPE = 4
    print(
        f"cumloss={[t.numpy() for t in train_memory(me, X, Xnoise, SPE, loss_fn, opt)]}"
    )
