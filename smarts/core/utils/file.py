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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import dataclasses
import hashlib
import os
import shutil


# https://stackoverflow.com/a/2166841
def isnamedtupleinstance(x):
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(t, "_fields", None)
    if not isinstance(f, tuple):
        return False
    return all(type(n) == str for n in f)


def isdataclass(x):
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
        return {key: unpack(value) for key, value in obj._asdict().items()}
    elif isdataclass(obj):
        return dataclasses.asdict(obj)
    elif isinstance(obj, tuple):
        return tuple(unpack(value) for value in obj)
    else:
        return obj


def copy_tree(from_path, to_path, overwrite=False):
    if os.path.exists(to_path):
        if overwrite:
            shutil.rmtree(to_path)
        else:
            raise FileExistsError(
                "The destination path={} already exists.".format(to_path)
            )

    shutil.copytree(from_path, to_path)


def path2hash(file_path: str):
    m = hashlib.md5()
    m.update(bytes(file_path, "utf-8"))
    return m.hexdigest()


def file_md5_hash(file_path: str) -> str:
    hasher = hashlib.md5()
    with open(file_path) as f:
        hasher.update(f.read().encode())

    return str(hasher.hexdigest())


def smarts_log_dir() -> str:
    ## Following should work for linux and macos
    smarts_dir = os.path.join(os.path.expanduser("~"), ".smarts")
    os.makedirs(smarts_dir, exist_ok=True)
    return smarts_dir


def make_dir_in_smarts_log_dir(dir):
    return os.path.join(smarts_log_dir(), dir)
