# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys
import time
import _pickle
import timeit
import argparse
import glob

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci
import traci.constants as tc


v_types = ["default", "aggressive", "slow", "blocker", "crusher"]


def get_vehicle_type(vehicle_name):
    type_id = traci.vehicle.getTypeID(vehicle_name)
    for t in v_types:
        if t in type_id:
            return t


def _base_sumo_load_params(net_file, route_file, delta_t):
    load_params = [
        "--lanechange.duration=3.0",
        "--step-length=%f" % delta_t,
        "--begin=0",
        "--end=31536000",
        #
        "--num-clients=1",
        "--net-file=%s" % net_file,
        "--quit-on-end",
        # "--no-step-log",
        # "--no-warnings=1",
        "--seed=%s" % 354354,
        "--time-to-teleport=%s" % -1,
        "--collision.check-junctions=true",
        "--collision.action=none",
        "--log=%s" % "sumo_log.log"
        #
    ]
    load_params.append("--start")
    load_params.append("--route-files={}".format(route_file))

    return load_params


def get_simulation_info(net_file, route_file, delta_t=1.0):
    params = _base_sumo_load_params(net_file, route_file, delta_t)
    traci.start(["sumo", *params])
    time.sleep(0.1)

    warmup = True
    t = 0.0

    vehicle_stats = {}
    # exited = {}

    while warmup or traci.vehicle.getIDCount() > 0:
        traci.simulationStep(t)
        for id in traci.vehicle.getIDList():
            # if id in exited.keys():
            #     print("Exited reappear")

            if id not in vehicle_stats:
                v_t = get_vehicle_type(id)
                v_r = traci.vehicle.getRoute(id)
                vehicle_stat = {
                    "id": id,
                    "start": t,
                    "end": t,
                    "stop_time": 0,
                    "type": v_t,
                    "route": "{}-{}".format(*v_r),
                }
                vehicle_stats[id] = vehicle_stat
            else:
                vehicle_stats[id]["end"] = t

            if traci.vehicle.getSpeed(id) < 0.01:
                vehicle_stats[id]["stop_time"] += delta_t

            # _exited_id = list(set(vehicle_stats.keys()) - set(traci.vehicle.getIDList()))
            # for _id in _exited_id:
            #     exited[_id] = 1
            # print(len(exited.keys()))

        t += delta_t
        if t > 20:
            warmup = False

    vehicle_types_time_taken = {t: [] for t in v_types}

    routes_time_taken = {
        r: [] for r in list(set([v["route"] for v in vehicle_stats.values()]))
    }
    stop_pcts = {r: [] for r in list(set([v["route"] for v in vehicle_stats.values()]))}

    max_t = 0
    for v in vehicle_stats.values():
        v_t = v["type"]
        v_r = v["route"]
        time_taken = v["end"] - v["start"]
        vehicle_types_time_taken[v_t].append(time_taken)
        routes_time_taken[v_r].append(time_taken)
        _stop_pct = v["stop_time"] / time_taken
        stop_pcts[v_r].append(_stop_pct)
        if v_r != "edge-south-SN-edge-west-EW":
            max_t = max(time_taken, max_t)

    traci.close()

    return {
        "time": t,
        "vehicle_types_time_taken": vehicle_types_time_taken,
        "routes_time_taken": routes_time_taken,
        "max_time": max_t,
        "stop_pcts": stop_pcts,
    }


def save_histogram(time_taken, fn, title):
    plt.figure()
    plt.title(title)
    plt.hist(time_taken, normed=False, bins=20)
    plt.savefig(fn)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="wildcard path to scenarios dirs")
    parser.add_argument("--plot_only", "-p", dest="plot_only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)

    args = parse_args()
    plot_only = args.plot_only
    dirs = args.path

    if not plot_only:
        delta_t = 0.1
        failed_scenario = 0

        dirs = sorted(glob.glob(dirs))
        episode_time_taken = []
        vehicle_types_time_taken = {t: [] for t in v_types}
        routes_time_taken = {}
        max_t = []
        stop_pcts = {}

        for j, d in enumerate(dirs):
            start = timeit.default_timer()

            info = get_simulation_info(
                os.path.join(d, "map.net.xml"),
                os.path.join(d, "traffic/all.rou.xml"),
                delta_t,
            )
            _episode_time_taken = info["time"]
            _vehicle_types_time_taken = info["vehicle_types_time_taken"]
            _routes_time_taken = info["routes_time_taken"]
            _max_t = info["max_time"]
            _stop_pcts = info["stop_pcts"]

            episode_time_taken.append(_episode_time_taken)
            for t in v_types:
                vehicle_types_time_taken[t].extend(_vehicle_types_time_taken[t])
            for rn, rt in _routes_time_taken.items():
                if rn not in routes_time_taken:
                    routes_time_taken[rn] = []
                routes_time_taken[rn].extend(rt)
            max_t.append(_max_t)
            for rn, rt in _stop_pcts.items():
                if rn not in stop_pcts:
                    stop_pcts[rn] = []
                stop_pcts[rn].extend(rt)

            print(
                "Scenario {} ({}) take {}s, {}s simulation".format(
                    j, d, timeit.default_timer() - start, _episode_time_taken
                )
            )
    else:
        info = _pickle.load(open(".info.pkl", "rb"))
        episode_time_taken = info["episode_time_taken"]
        vehicle_types_time_taken = info["vehicle_types_time_taken"]
        routes_time_taken = info["routes_time_taken"]
        max_t = info["max_time"]
        stop_pcts = info["stop_pcts"]

    save_histogram(episode_time_taken, "plots/time_episode.png", "time episode")
    for v_t in vehicle_types_time_taken.keys():
        save_histogram(
            vehicle_types_time_taken[v_t],
            "plots/time_type_{}.png".format(v_t),
            "time {}".format(v_t),
        )
    for v_r in routes_time_taken.keys():
        save_histogram(
            routes_time_taken[v_r],
            "plots/time_route_{}.png".format(v_r),
            "time {}".format(v_r),
        )
    save_histogram(
        sum(list(vehicle_types_time_taken.values()), []),
        "plots/time_all.png",
        "time all",
    )
    save_histogram(max_t, "plots/time_max_t.png", "max_t")
    for v_r in stop_pcts.keys():
        save_histogram(
            stop_pcts[v_r],
            "plots/stop_pct_route_{}.png".format(v_r),
            "stop % {}".format(v_r),
        )
    save_histogram(
        sum(list(stop_pcts.values()), []), "plots/stop_pct_all.png", "stop % all"
    )

    _pickle_info = {
        "episode_time_taken": episode_time_taken,
        "vehicle_types_time_taken": vehicle_types_time_taken,
        "routes_time_taken": routes_time_taken,
        "max_time": max_t,
        "stop_pcts": stop_pcts,
    }

    _pickle.dump(_pickle_info, open(".info.pkl", "wb"))
