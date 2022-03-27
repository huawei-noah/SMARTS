# Example script experts from rl_swiss https://github.com/KamyarGh/rl_swiss
_pantry = {}


def get_scripted_policy(scripted_policy_name):
    return _pantry[scripted_policy_name]()
