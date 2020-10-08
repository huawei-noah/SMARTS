"""smarts.core
===========

Core functionality of the SMARTS simulator
"""

import random
import uuid

import numpy as np


def seed(a):
    random.seed(a)
    np.random.seed(a)


def gen_id():
    """Generates a unique but deterministic id if `smarts.core.seed` has set the core seed."""
    return uuid.UUID(int=random.getrandbits(128))
