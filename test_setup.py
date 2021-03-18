#!/usr/bin/python3
import sys, os, subprocess
import multiprocessing

def exit_with_error(msg):
    print('ERROR:', msg)
    sys.exit(-1)

# Check SUMO_HOME is properly set
SUMO_HOME = 'SUMO_HOME'
sumo_path = os.getenv(SUMO_HOME)
if sumo_path is None:
    exit_with_error('{} environment variable is not set'.format(SUMO_HOME))
elif not os.path.exists(sumo_path):
    exit_with_error('{} path: {} is invalid'.format(SUMO_HOME, sumo_path))

# #PYTHONHASHSEED=42 
# pytest_command = [
#     'pytest', '-v',
#     '--cov=smarts',
#     '--doctest-modules',
#     '--forked',
#     '--dist=loadscope',
#     '-n {}'.format(multiprocessing.cpu_count()-2),
#     './smarts/contrib','./smarts/core', './smarts/env','./smarts/sstudio','./tests',
#     "-k 'not test_long_determinism'",
# ]

# ignored_tests = [
#     './smarts/core/tests/test_smarts_memory_growth.py',
#     './smarts/env/tests/test_benchmark.py'
#     './smarts/env/tests/test_learning.py'
# ]

# for test in ignored_tests:
#     pytest_command.append('--ignore={}'.format(test))

# subprocess.run(pytest_command)

# # pytest -x           # stop after first failure
# # pytest --maxfail=2  # stop after two failures