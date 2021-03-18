#!/usr/bin/python3
import sys, os, subprocess
import multiprocessing


def exit_with_error(msg):
    print("ERROR:", msg)
    sys.exit(-1)


# Check SUMO_HOME is properly set
SUMO_HOME = "SUMO_HOME"
sumo_path = os.getenv(SUMO_HOME)
if sumo_path is None:
    exit_with_error("{} environment variable is not set".format(SUMO_HOME))
elif not os.path.exists(sumo_path):
    exit_with_error("{} path: {} is invalid".format(SUMO_HOME, sumo_path))
