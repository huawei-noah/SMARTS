# MIT License

# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Filename: profiler.py
import cProfile
import pstats
import line_profiler
import atexit

import pathlib

ROOT = pathlib.Path(__file__).parent.absolute()

profile_line = line_profiler.LineProfiler()
atexit.register(profile_line.dump_stats, ROOT / "results" / "profile_line.prof")
stream = open(ROOT / "results" / "profile_line.txt", "w")
atexit.register(profile_line.print_stats, stream)

# Function profiling decorator
def profile_function(func):
    def _profile_function(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()

        # save stats into file
        pr.dump_stats(ROOT / "results" / "profile_function.prof")
        stream = open(ROOT / "results" / "profile_function.txt", "w")
        ps = pstats.Stats(
            str((ROOT / "results" / "profile_function.prof").resolve()), stream=stream
        )
        ps.sort_stats("tottime")
        ps.print_stats()

        return result

    return _profile_function
