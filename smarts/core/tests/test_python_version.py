import sys


def test_python_version():
    assert sys.version_info >= (3, 7, 0)
