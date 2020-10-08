"""Importing this module "redirects" the import to the "real" sumolib. This is available
for convenience and to reduce code duplication as sumolib lives under SUMO_HOME.
"""

import os
import sys

# Check for sumo home
if "SUMO_HOME" not in os.environ:
    raise ImportError("SUMO_HOME not set, can't import sumolib")

sumo_path = os.environ["SUMO_HOME"]
tools_path = os.path.join(sumo_path, "tools")
if tools_path not in sys.path:
    sys.path.append(tools_path)


# Intentionally making this available
SUMO_PATH = sumo_path

import sumolib
import traci
