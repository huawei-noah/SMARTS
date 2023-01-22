# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import dataclasses
import hashlib
import os
import pickle
import shutil
import struct
from contextlib import contextmanager
from ctypes import c_int64
from typing import Any, Generator

import smarts


def file_in_folder(filename: str, path: str) -> bool:
    """Checks to see if a file exists
    Args:
        filename: The name of the file.
        path: The path to the directory of the file.
    Returns:
        If the file exists.
    """
    return os.path.exists(os.path.join(path, filename))


# https://stackoverflow.com/a/2166841
def isnamedtupleinstance(x):
    """Check to see if an object is a named tuple."""
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(t, "_fields", None)
    if not isinstance(f, tuple):
        return False
    return all(type(n) == str for n in f)


def replace(obj: Any, **kwargs):
    """Replace dataclasses and named tuples with the same interface."""
    if isnamedtupleinstance(obj):
        return obj._replace(**kwargs)
    elif dataclasses.is_dataclass(obj):
        return dataclasses.replace(obj, **kwargs)

    raise ValueError("Must be a namedtuple or dataclass.")


def isdataclass(x):
    """Check if an object is a dataclass."""
    return dataclasses.is_dataclass(x)


def unpack(obj):
    """A helper that can be used to print `nestedtuples`. For example,
    ```python
    pprint(unpack(obs), indent=1, width=80, compact=True)
    ```
    """
    if isinstance(obj, dict):
        return {key: unpack(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [unpack(value) for value in obj]
    elif isnamedtupleinstance(obj):
        return unpack(obj._asdict())
    elif isdataclass(obj):
        return unpack(dataclasses.asdict(obj))
    elif isinstance(obj, tuple):
        return tuple(unpack(value) for value in obj)
    else:
        return obj


def copy_tree(from_path, to_path, overwrite=False):
    """Copy a directory tree (including files) to another location.
    Args:
        from_path: The directory to copy.
        to_path: The output directory.
        overwrite: If to overwrite the output directory.
    """
    if os.path.exists(to_path):
        if overwrite:
            shutil.rmtree(to_path)
        else:
            raise FileExistsError(
                "The destination path={} already exists.".format(to_path)
            )

    shutil.copytree(from_path, to_path)


def path2hash(file_path: str):
    """Converts a file path to a hash value."""
    m = hashlib.md5()
    m.update(bytes(file_path, "utf-8"))
    return m.hexdigest()


def file_md5_hash(file_path: str) -> str:
    """Converts file contents to a hash value. Useful for doing a file diff."""
    hasher = hashlib.md5()
    with open(file_path) as f:
        hasher.update(f.read().encode())

    return str(hasher.hexdigest())


def pickle_hash(obj, include_version=False) -> str:
    """Converts a Python object to a hash value. NOTE: NOT stable across different Python versions."""
    pickle_bytes = pickle.dumps(obj, protocol=4)
    hasher = hashlib.md5()
    hasher.update(pickle_bytes)

    if include_version:
        hasher.update(smarts.VERSION.encode())

    return hasher.hexdigest()


def pickle_hash_int(obj) -> int:
    """Converts a Python object to a hash value. NOTE: NOT stable across different Python versions."""
    hash_str = pickle_hash(obj)
    val = int(hash_str, 16)
    return c_int64(val).value


def smarts_log_dir() -> str:
    """Retrieves the smarts logging directory."""
    ## Following should work for linux and macos
    smarts_dir = os.path.join(os.path.expanduser("~"), ".smarts")
    os.makedirs(smarts_dir, exist_ok=True)
    return smarts_dir


def make_dir_in_smarts_log_dir(dir):
    """Return a new directory location in the smarts logging directory."""
    return os.path.join(smarts_log_dir(), dir)


@contextmanager
def suppress_pkg_resources():
    """A context manager that injects an `ImportError` into the `pkg_resources` module to force
    package fallbacks in imports that can use alternatives to `pkg_resources`.
    """
    import sys

    import pkg_resources

    from smarts.core.utils.invalid import raise_import_error

    pkg_res = sys.modules["pkg_resources"]
    sys.modules["pkg_resources"] = property(raise_import_error)
    yield
    sys.modules["pkg_resources"] = pkg_res


def read_tfrecord_file(path: str) -> Generator[bytes, None, None]:
    """Iterate over the records in a TFRecord file and return the bytes of each record.

    path: The path to the TFRecord file
    """
    with open(path, "rb") as f:
        while True:
            length_bytes = f.read(8)
            if len(length_bytes) != 8:
                return
            record_len = int(struct.unpack("Q", length_bytes)[0])
            _ = f.read(4)  # masked_crc32_of_length (ignore)
            record_data = f.read(record_len)
            _ = f.read(4)  # masked_crc32_of_data (ignore)
            yield record_data
