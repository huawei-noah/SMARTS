from rlkit.env_creators.smarts.smarts_env import SmartsEnv


def get_env_cls(env_creator_name: str):
    return {
        "smarts": SmartsEnv,
    }[env_creator_name]
