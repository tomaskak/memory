import pytest

import tensorflow as tf

from memory.datagen import gen_distance_loss_factors


def test_generate_distance_loss_data():
    X = tf.constant([[0.0, 1.0], [0.0, 1.0], [3.0, 0.0], [3.0, 0.0]])
    Xnoise = tf.zeros(shape=(4, 2))

    actual = gen_distance_loss_factors(X, Xnoise, episode_length=4, update_length=2)
    expected = tf.constant([[1.0, 1.0], [1.0, 1.0], [3.0, 3.0]])
    print(actual)
    print(expected)
    assert tf.reduce_all(tf.equal(actual, expected))
