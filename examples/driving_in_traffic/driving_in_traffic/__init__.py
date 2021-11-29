import random

import numpy as np
import tensorflow as tf


def seed(a: int):
    np.random.seed(a)
    random.seed(a)
    # set_seed() makes random number generation in TensorFlow backend to
    # have a well-defined initial state.
    # https://www.tensorflow.org/api_docs/python/tf/random/set_seed
    tf.random.set_seed(a)
