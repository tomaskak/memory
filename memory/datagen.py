from tensorflow import (
    broadcast_to,
    concat,
    stack,
    function,
    expand_dims,
    constant,
    zeros,
    float32,
)
from tensorflow.math import reduce_euclidean_norm, reduce_min, reduce_sum, sqrt
from tensorflow.linalg import set_diag

from tensorflow_probability import math


def euclidean_norm(two_d_tensor):
    return sqrt(reduce_sum(two_d_tensor ** 2, axis=(2)))


def gen_update_factors_for(X, Xnoise):
    replicated_targets = broadcast_to(expand_dims(X, axis=0), (len(Xnoise), *X.shape))

    replicated_Y = broadcast_to(expand_dims(Xnoise, axis=1), (len(Xnoise), *X.shape))

    distances = euclidean_norm(replicated_targets - replicated_Y)

    N = len(distances)
    sz = ((N) * (N + 1)) // 2
    triag_base = constant(5000.0, float32, shape=(sz,))
    mask = math.fill_triangular(triag_base, upper=True)
    mask = set_diag(mask, zeros(shape=(len(mask),)))
    mins = reduce_min(distances + mask, axis=1)

    return mins


@function(jit_compile=True)
def gen_episode_of_distance_loss_factors(X, Xnoise, update_length):
    factors = []
    for window_start in range(len(X) - update_length + 1):
        offset = window_start
        factors.append(
            gen_update_factors_for(
                X[offset : offset + update_length],
                Xnoise[offset : offset + update_length],
            )
        )
    # factors.append(zeros(shape=(update_length-1,X.shape[1])))

    # Pad with entries of zero for end values which should not be tested as there is not enough
    # data in the episode remaining to test the full memory length.
    return concat(
        (stack(factors, axis=0), zeros(shape=(update_length - 1, len(factors[0])))),
        axis=0,
    )


def gen_distance_loss_factors(X, Xnoise, episode_length, update_length):
    """
    Generates the factors for weighting loss of labelling an unseen value.
    Weights are derived from the euclidean distance of an unseen value to the nearest seen
    value in memory.

    episode_length is how many rows in X and Xnoise pertain to a single episode.
    update_length is the number of rows that are expected to be present in memory at a time.

    Outputs a matrix with an extra axis as compared to Xnoise where each element on Xnoise's final
    axis maps to a vector in the output.
    """
    factors = []
    for episode in range(len(X) // episode_length):

        if episode % 100 == 0:
            print(f"generated distance loss for episode {episode}.")

        episode_offset = episode * episode_length
        factors.append(
            gen_episode_of_distance_loss_factors(
                X[episode_offset : episode_offset + episode_length],
                Xnoise[episode_offset : episode_offset + episode_length],
                update_length,
            )
        )
    print(f"finished generated distance losses. number generated={len(factors)}")
    print(f"x_len={len(X)}, e_length={episode_length}, u_length={update_length}")

    return concat(factors, axis=0)
