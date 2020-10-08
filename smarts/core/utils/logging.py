import logging
from time import time
from contextlib import contextmanager


@contextmanager
def timeit(name: str):
    start = time()
    yield
    ellapsed_time = (time() - start) * 1000

    logging.info(f'"{name}" took: {ellapsed_time:4f}ms')
