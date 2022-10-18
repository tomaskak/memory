#!python
"""
Runs a training benchmark on the MemoryEncoder.
"""

from memory.encoders import MemoryEncoder, train_memory

from tensorflow import (
    tensor_scatter_nd_update,
    concat,
    constant,
    abs,
    zeros,
    expand_dims,
    gather,
    function,
)
from tensorflow.random import normal, uniform
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.summary import create_file_writer, scalar

from time import time
from datetime import datetime

from argparse import ArgumentParser


@function(jit_compile=True)
def gen_episode(e_length, obs_size, mu, sigma):
    base = normal((e_length, obs_size), mu, sigma)
    current = normal((1, obs_size), mu, sigma)
    rows = []
    for i in range(e_length):
        b = abs(gather(base, [i], axis=0))
        current = current + b
        rows.append(current)
    return concat(rows, axis=0)


def gen_increasing(shape, mu, sigma, episode_length):
    data_size, obs_size = shape

    base = normal(shape, mu, sigma)

    rows = []
    current = None
    for i in range(data_size // episode_length):
        rows.append(gen_episode(episode_length, obs_size, mu, sigma))
        if i % (10 * episode_length) == 0:
            print(f"{i} episodes generated.")
    print("episode generation completed.")

    return concat(rows, axis=0)


def main(data: str, noise: str):
    """
    Benchmark for performance of remembering a number of states produces from random sampling
    i.e. no correlation from which to enhance compression ability.
    """
    # starting with one case
    observation_size = 10
    encoder_sizes = [1200, 1200]
    memory_size = 50
    memory_length = 5
    memory_decay = 0.9

    me = MemoryEncoder(
        observation_size,
        encoder_sizes,
        memory_size=memory_size,
        memory_length=memory_length,
        memory_decay=memory_decay,
    )

    learning_rate = 0.001

    opt = Adadelta(learning_rate=learning_rate)
    loss_fn = CategoricalCrossentropy()

    data_size = 30 * 1000 * 1000
    steps_in_episode = memory_length * 50
    mu = 0.0
    sigma = 1.0
    noise_sigma = 0.5

    num_paths = 20

    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    def train(key, X, Xnoise):
        train_log_dir = "logs/" + now_str + "-" + key + "/train"
        train_summary_writer = create_file_writer(train_log_dir)

        log_template = "{}-{}: \tEpisode={}, Step={}, Loss={}, Accuracy={}"

        before_training_time = time()

        def log_training_result(loss_metric, accuracy_metric, episode):
            with train_summary_writer.as_default():
                step = (episode + 1) * steps_in_episode
                scalar("loss", loss_metric.result(), step=episode + 1)
                scalar("accuracy", accuracy_metric.result(), step=episode + 1)
                print(
                    log_template.format(
                        key,
                        f"{time() - before_training_time:.3f}",
                        episode,
                        step,
                        f"{loss_metric.result():.3f}",
                        f"{accuracy_metric.result():.3f}",
                    )
                )

        cumulative_losses = train_memory(
            me, X, Xnoise, steps_in_episode, loss_fn, opt, log_training_result
        )

    if data == "all":
        data = "random,increasing"
    noise_types = noise.split(",")
    data_types = data.split(",")

    for data_type in data_types:
        if data_type == "random":
            X = normal((data_size, me.observation_size), mu, sigma)
        elif data_type == "increasing":
            # Creates data from monotonically increasing paths
            X = gen_increasing(
                (data_size, me.observation_size), mu, sigma / 10.0, steps_in_episode
            )

        if noise == "all" or "same-dist" in noise_types:
            if data_type == "random":
                Xnoise = normal((data_size, me.observation_size), mu, sigma)
            elif data_type == "increasing":
                Xnoise = gen_increasing(
                    (data_size, me.observation_size), mu, sigma / 10.0, steps_in_episode
                )

            before = time()
            train(data_type + "-same-dist", X, Xnoise)
            print(f"same-dist finished training in {time() - before}s")

        if noise == "all" or "added-noise" in noise_types:
            # Random noise to all dimensions
            Xnoise = X + normal((data_size, me.observation_size), mu, noise_sigma)

            before = time()
            train(data_type + "-added-noise-sigma-" + str(noise_sigma), X, Xnoise)
            print(f"added-noise-sigma finished training in {time() - before}s")

        if noise == "all" or "1D-added-noise" in noise_types:
            # Random noise to one dimension at a time
            tensor = X
            indices = concat(
                (
                    constant([[i] for i in range(data_size)], dtype="int64"),
                    uniform(
                        shape=(data_size, 1),
                        minval=0,
                        maxval=me.observation_size,
                        dtype="int64",
                    ),
                ),
                axis=-1,
            )
            updates = normal((data_size,), mu, noise_sigma)
            Xnoise = tensor_scatter_nd_update(X, indices, updates)

            before = time()
            train(data_type + "-1D-added-noise-" + str(noise_sigma), X, Xnoise)
            print(f"1D-added-noise finished training in {time() - before}s")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="run training tests that benchmark the memory encoder performance under various conditions."
    )
    parser.add_argument(
        "--noise-src",
        "-n",
        dest="noise",
        help="key indicating which types of noise data should be run. mutiple values can be inputted as a comma-separated string.",
        default="all",
    )

    parser.add_argument(
        "--data-src",
        "-d",
        dest="data",
        help="key indicating which data source to run. mutiple values can be inputted as a comma-separated string.",
        default="all",
    )

    args = parser.parse_args()

    start = time()
    main(args.data, args.noise)
    fin = time()
    print(f"all benchmarks completed in {fin - start}s")
