import pytest

import tensorflow as tf

from memory.encoders import AutoEncoder


@pytest.mark.slow
@pytest.mark.train
def test_autoencoder():
    ae = AutoEncoder([4, 2], [4, 3], activation="linear")
    ae.compile(
        optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.1),
        loss=tf.keras.losses.MeanSquaredError(),
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
