# Filename: profiler.py
import cProfile
import pstats
import line_profiler
import atexit

import pathlib
ROOT = pathlib.Path(__file__).parent.absolute()

profile_line = line_profiler.LineProfiler()
atexit.register(profile_line.dump_stats, ROOT/'results'/'profile_line.prof')
stream = open(ROOT/'results'/'profile_line.txt', 'w')
atexit.register(profile_line.print_stats, stream)

# Function profiling decorator
def profile_function(func):
    def _profile_function(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()

        # save stats into file
        pr.dump_stats(ROOT/'results'/'profile_function.prof')
        stream = open(ROOT/'results'/'profile_function.txt', 'w')
        ps = pstats.Stats(str((ROOT/'results'/'profile_function.prof').resolve()), stream=stream)
        ps.sort_stats('tottime')
        ps.print_stats()
        
        return result  
    return _profile_function