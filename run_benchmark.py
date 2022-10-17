#!python
"""
Runs a training benchmark on the MemoryEncoder.
"""

from memory.encoders import MemoryEncoder, train_memory

from tensorflow import tensor_scatter_nd_update, concat, constant
from tensorflow.random import normal, uniform
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.summary import create_file_writer, scalar

from time import time
from datetime import datetime

from argparse import ArgumentParser


def main(noise: str):
    """
    Benchmark for performance of remembering a number of states produces from random sampling
    i.e. no correlation from which to enhance compression ability.
    """
    # starting with one case
    observation_size = 3
    encoder_sizes = [500, 500]
    memory_size = 30
    memory_length = 2
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

    def train(noise_key, X, Xnoise):
        train_log_dir = (
            "logs/"
            + datetime.now().strftime("%Y%m%d-%H%M%S")
            + "-"
            + noise_key
            + "/train"
        )
        train_summary_writer = create_file_writer(train_log_dir)

        log_template = "{}: \tEpisode={}, Step={}, Loss={}, Accuracy={}"

        before_training_time = time()

        def log_training_result(loss_metric, accuracy_metric, episode):
            with train_summary_writer.as_default():
                step = (episode + 1) * steps_in_episode
                scalar("loss", loss_metric.result(), step=episode + 1)
                scalar("accuracy", accuracy_metric.result(), step=episode + 1)
                print(
                    log_template.format(
                        time() - before_training_time,
                        episode,
                        step,
                        loss_metric.result(),
                        accuracy_metric.result(),
                    )
                )

        cumulative_losses = train_memory(
            me, X, Xnoise, steps_in_episode, loss_fn, opt, log_training_result
        )

    noise_types = noise.split(",")

    X = normal((data_size, me.observation_size), mu, sigma)

    if noise == "all" or "same-dist" in noise_types:
        Xnoise = normal((data_size, me.observation_size), mu, sigma)

        before = time()
        train("same-dist", X, Xnoise)
        print(f"same-dist finished training in {time() - before}s")

    if noise == "all" or "added-noise" in noise_types:
        # Random noise to all dimensions
        Xnoise = X + normal((data_size, me.observation_size), mu, noise_sigma)

        before = time()
        train("added-noise-sigma-" + str(noise_sigma), X, Xnoise)
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
        train("1D-added-noise-" + str(noise_sigma), X, Xnoise)
        print(f"1D-added-noise finished training in {time() - before}s")

    # print(f"training progression: {[t.numpy() for t in cumulative_losses]}")


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
    args = parser.parse_args()

    start = time()
    main(args.noise)
    fin = time()
    print(f"all benchmarks completed in {fin - start}s")