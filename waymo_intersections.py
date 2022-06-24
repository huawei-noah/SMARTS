from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Dict, Set, Tuple
import numpy as np
import matplotlib.pyplot as plt
from smarts.core.utils.math import line_intersect

from smarts.core.waymo_map import WaymoMap
from smarts.sstudio.types import MapSpec

expected_intersection_dict = {
    "70": set(),
    "71": set(),
    "80": set(),
    "81": set(),
    "83": set(),
    "84": set(),
    "85": set(),
    "86": {"87"},
    "87": {"86"},
    "86_4": set(),
    "86_16": {"92_32"},
    "92_32": {"86_16"},
    "87_4": {"92_6", "90_6"},
    "89_17": {"87_26"},
    "87_26": {"89_17"},
    "88": set(),
    "89": {"90"},
    "90": {"89"},
    "89_6": set(),
    "90_6": {"92_6", "87_4"},
    "91_24": {"90_32"},
    "90_32": {"91_24"},
    "91": {"92"},
    "92": {"91"},
    "91_5": set(),
    "92_6": {"90_6", "87_4"},
    "93": set(),
    "95": set(),
    "110": {"109", "96"},
    "96": {"110", "109"},
    "109": {"110", "96"},
    "110_5": set(),
    "96_4": set(),
    "109_4": set(),
    "110_6": set(),
    "96_5": set(),
    "109_5": set(),
    "96_7": set(),
    "96_8": {"102_9", "101_8", "97_9", "105_10"},
    "96_36": set(),
    "108_35": set(),
    "98_21": set(),
    "96_38": set(),
    "108_37": set(),
    "98_22": set(),
    "96_39": set(),
    "108_38": set(),
    "98_23": set(),
    "96_40": set(),
    "108_39": set(),
    "98": {"102", "97"},
    "102": {"98", "97"},
    "97": {"102", "98"},
    "102_4": set(),
    "97_4": set(),
    "102_5": set(),
    "97_5": set(),
    "97_9": {"96_8", "101_8", "109_8", "105_10"},
    "103_37": set(),
    "97_32": set(),
    "107_19": set(),
    "103_40": set(),
    "97_35": set(),
    "107_20": {"97_36", "103_41"},
    "103_41": {"107_20"},
    "97_36": {"107_20"},
    "98_4": set(),
    "99": set(),
    "100": set(),
    "107": {"108", "101"},
    "101": {"108", "107"},
    "108": {"107", "101"},
    "107_3": set(),
    "101_2": set(),
    "108_2": set(),
    "101_4": set(),
    "108_5": set(),
    "101_8": {"96_8", "97_9", "109_8", "103_9"},
    "101_34": set(),
    "105_34": set(),
    "110_24": set(),
    "101_36": set(),
    "105_36": set(),
    "110_25": {"105_37"},
    "101_37": set(),
    "105_37": {"110_25"},
    "102_9": {"96_8", "105_10", "108_9", "103_9"},
    "102_33": set(),
    "109_34": set(),
    "106_26": set(),
    "102_35": set(),
    "109_36": set(),
    "106_27": set(),
    "102_36": set(),
    "109_37": set(),
    "106": {"103", "105"},
    "103": {"106", "105"},
    "105": {"106", "103"},
    "106_6": set(),
    "103_6": set(),
    "105_7": set(),
    "103_8": set(),
    "103_9": {"102_9", "101_8", "109_8", "108_9"},
    "104": set(),
    "105_10": {"102_9", "96_8", "97_9", "108_9"},
    "106_7": set(),
    "106_8": set(),
    "107_5": set(),
    "108_4": set(),
    "108_9": {"102_9", "105_10", "109_8", "103_9"},
    "109_8": {"101_8", "97_9", "108_9", "103_9"},
    "110_8": set(),
    "111": set(),
    "112": set(),
    "113": {"121"},
    "121": {"113"},
    "113_4": set(),
    "113_23": {"114_30"},
    "114_30": {"113_23"},
    "114": {"119_4", "121_5", "115"},
    "115": {"114"},
    "115_18": set(),
    "119_27": set(),
    "116": set(),
    "117": {"119"},
    "119": {"117"},
    "117_5": set(),
    "117_20": {"121_28"},
    "121_28": {"117_20"},
    "128": {"118"},
    "118": {"128"},
    "118_8": {"130", "123_5"},
    "131_22": set(),
    "118_33": set(),
    "119_4": {"121_5", "114"},
    "120": set(),
    "121_5": {"119_4", "114"},
    "122": set(),
    "126": {"123"},
    "123": {"126"},
    "123_5": {"130", "118_8"},
    "128_24": set(),
    "123_28": set(),
    "124": set(),
    "125": set(),
    "126_5": set(),
    "126_24": {"130_37"},
    "130_37": {"126_24"},
    "127": {"136_9", "138_12", "167_9"},
    "127_15": set(),
    "145_20": set(),
    "135_22": set(),
    "127_18": set(),
    "145_23": set(),
    "135_23": set(),
    "127_19": set(),
    "145_24": set(),
    "135_24": {"145_25", "127_20"},
    "127_20": {"135_24"},
    "145_25": {"135_24"},
    "128_7": set(),
    "129": set(),
    "130": {"118_8", "123_5", "131"},
    "131": {"130"},
    "132": set(),
    "133": set(),
    "135": {"167", "136"},
    "136": {"135", "167"},
    "167": {"135", "136"},
    "135_5": set(),
    "167_4": set(),
    "136_4": set(),
    "136_9": {"138_12", "127"},
    "137": {"141", "138"},
    "138": {"137", "141"},
    "141": {"137", "138"},
    "137_8": set(),
    "141_7": set(),
    "138_8": set(),
    "137_9": set(),
    "141_8": set(),
    "138_9": set(),
    "137_12": set(),
    "137_27": set(),
    "167_35": set(),
    "143_15": set(),
    "137_28": {"143_16"},
    "167_36": set(),
    "143_16": {"137_28"},
    "138_12": {"136_9", "167_9", "127", "145"},
    "139": set(),
    "141_11": {"143", "167_9", "145"},
    "143": {"141_11", "145"},
    "167_30": set(),
    "143_10": set(),
    "145": {"141_11", "143", "138_12", "167_9"},
    "152": {"151"},
    "151": {"152"},
    "151_5": {"153", "160_4"},
    "161_18": {"151_28"},
    "151_28": {"161_18"},
    "152_5": set(),
    "152_17": {"153_19"},
    "153_19": {"152_17"},
    "153": {"151_5", "160_4"},
    "156": set(),
    "157": set(),
    "158": set(),
    "159": set(),
    "161": {"160"},
    "160": {"161"},
    "160_4": {"153", "151_5"},
    "161_3": set(),
    "162": set(),
    "163": set(),
    "164": set(),
    "165": set(),
    "167_9": {"141_11", "138_12", "127", "145"},
}


def intersections_endpoints(map: WaymoMap) -> Tuple[Dict[str, Set[str]], float]:
    start = perf_counter()
    intersections = defaultdict(lambda: set())
    for lane_id, lane in map._lanes.items():
        intersections[lane_id] = set()
        for test_lane_id, test_lane in map._lanes.items():
            in_ids = [l.lane_id for l in lane.incoming_lanes]
            out_ids = [l.lane_id for l in lane.outgoing_lanes]
            if test_lane_id in in_ids + out_ids + [lane_id]:
                continue
            a = lane._lane_pts[0]
            b = lane._lane_pts[-1]
            c = test_lane._lane_pts[0]
            d = test_lane._lane_pts[-1]
            if line_intersect(a, b, c, d) is not None:
                intersections[lane_id].add(test_lane_id)
    end = perf_counter()
    elapsed = round((end - start), 3)
    return intersections, elapsed


def intersections_bruteforce(map: WaymoMap) -> Tuple[Dict[str, Set[str]], float]:
    start = perf_counter()
    intersections = defaultdict(lambda: set())
    for lane_id, lane in map._lanes.items():
        intersections[lane_id] = set()
        for test_lane_id, test_lane in map._lanes.items():
            in_ids = [l.lane_id for l in lane.incoming_lanes]
            out_ids = [l.lane_id for l in lane.outgoing_lanes]
            if test_lane_id in in_ids + out_ids + [lane_id]:
                continue
            done = False
            for i in range(lane._n_pts - 1):
                if done:
                    break
                a = lane._lane_pts[i]
                b = lane._lane_pts[i + 1]
                for j in range(test_lane._n_pts - 1):
                    c = test_lane._lane_pts[j]
                    d = test_lane._lane_pts[j + 1]
                    if line_intersect(a, b, c, d) is not None:
                        intersections[lane_id].add(test_lane_id)
                        done = True
    end = perf_counter()
    elapsed = round((end - start), 3)
    return intersections, elapsed


def main():
    # Load map
    dataset_path = "/home/saul/Downloads/waymo/1.1/uncompressed_scenario_training_20s_training_20s.tfrecord-00000-of-01000"
    scenario_id = "4f30f060069bbeb9"
    spec = MapSpec(f"{dataset_path}#{scenario_id}")
    map = WaymoMap.from_spec(spec)

    # Map info
    dataset_file = Path(dataset_path)
    print(f"Dataset File: {dataset_file.stem + dataset_file.suffix}")
    print(f"Scenario: {scenario_id}")
    print(f"Lanes: {len(map._lanes)}")

    # Test intersection algorithms
    intersections, elapsed_time = intersections_endpoints(map)
    # intersections, elapsed_time = intersections_bruteforce(map)

    print(f"Time: {elapsed_time} s")

    # for k, v in intersections.items():
    #     print(f"'{k}': {v},")

    for lane_id, lane in map._lanes.items():
        assert lane_id in expected_intersection_dict
        assert lane_id in intersections
        # assert (
        #     expected_intersection_dict[lane_id] == intersections[lane_id]
        # ), f"expected: {expected_intersection_dict[lane_id]}, got: {intersections[lane_id]}"
        # if expected_intersection_dict[lane_id] != intersections[lane_id]:
        #     print(lane_id)

    # Plot map

    plt.figure()
    mng = plt.get_current_fig_manager()
    mng.resize(1000, 1000)
    plt.title(f"Scenario {scenario_id}")

    lane_ids = {
        # "81",
        # "86",
        # "86_4",
        # "86_16",
        # "88",
        # "87",
        "87_4",
        # "87_26",
        # "93",
        # "92",
        # "92_6",
        # "92_32",
    }

    intersected_ids = set()
    for lane_id in lane_ids:
        intersected_ids |= intersections[lane_id]

    for lane_id, lane in map._lanes.items():
        # if lane_id not in lane_ids:
        #     continue
        pts = np.array(lane._lane_pts)
        # if lane_id in lane_ids:
        #     plt.plot(pts[:, 0], pts[:, 1], color="blue")
        # elif lane_id in intersected_ids:
        #     plt.plot(pts[:, 0], pts[:, 1], color="red")
        if len(expected_intersection_dict[lane_id]) > 0:
            plt.plot(pts[:, 0], pts[:, 1], color="red")
        else:
            plt.plot(pts[:, 0], pts[:, 1], linestyle=":", color="gray")
            # plt.scatter(pts[:, 0], pts[:, 1], color="gray", s=2)

    plt.show()


if __name__ == "__main__":
    main()
