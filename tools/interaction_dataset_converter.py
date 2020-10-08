import argparse
import csv
import json
import math

from collections import defaultdict

import numpy as np

"""Meta information of the track files, please see:
https://interaction-dataset.com/details-and-format
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser("interaction-dataset-converter")
    parser.add_argument(
        "input", help="Tracks file in csv format", type=str,
    )

    parser.add_argument(
        "output", help="History file in JSON format", type=str,
    )

    args = parser.parse_args()
    traffic_history = defaultdict(dict)
    with open(args.input, newline="") as csvfile:
        spamreader = csv.DictReader(csvfile)
        for row in spamreader:
            state = {
                "vehicle_id": row["track_id"],
                "vehicle_type": row["agent_type"],
                "position": [float(row["x"]), float(row["y"]), 0],
                "speed": np.linalg.norm([float(row["vx"]), float(row["vy"])]),
                "heading": float(row["psi_rad"]) - math.pi / 2,
                "vehicle_length": float(row["length"]),
                "vehicle_width": float(row["width"]),
            }
            traffic_history[round(int(row["timestamp_ms"]) / 1000, 3)][
                state["vehicle_id"]
            ] = state

    with open(args.output, "w") as f:
        f.write(json.dumps(traffic_history))
