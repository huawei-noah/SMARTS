"""
General classes, functions, utilities that are used throughout rlkit.
"""

from collections import defaultdict


def list_dict_to_dict_list(list_dict):
    dict_list = defaultdict(list)
    for _dict in list_dict:
        for k, v in _dict.items():
            dict_list[k].append(v)
    return dict(dict_list)


# def dict_list_to_list_dict(dict_list):
#     # For example,
#     # Input: {"agent_0": [1, 2], "agent_1": [3, 4]}
#     # Output: [{"agent_0": 1, "agent_1": 3}, {"agent_0": 2, "agent_1": 4}]
#     list_dict = [
#         {k: v[idx] for k, v in dict_list.items()}
#         for idx in range(len(list(dict_list.values())[0]))
#     ]
#     return list_dict


def dict_list_to_list_dict(dict_list):
    # For example,
    # Input: {"agent_0": [1, 2], "agent_1": [3]}
    # Output: [{"agent_0": 1, "agent_1": 3}, {"agent_0": 2}]
    # support not equal length list
    maxlen = max([len(v) for v in dict_list.values()])
    list_dict = [{} for _ in range(maxlen)]
    for k, v in dict_list.items():
        for idx in range(len(v)):
            list_dict[idx][k] = v[idx]
    return list_dict
