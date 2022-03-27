from rlkit.env_creators import get_env_cls
from rlkit.envs.wrappers import ProxyEnv
from rlkit.envs.vecenvs import BaseVectorEnv, DummyVectorEnv, SubprocVectorEnv

__all__ = [
    "BaseVectorEnv",
    "DummyVectorEnv",
    "SubprocVectorEnv",
]


def get_env(env_specs, vehicle_ids=None):
    """
    env_specs:
        env_name: 'mujoco'
        scenario_name: 'halfcheetah'
        env_kwargs: {} # kwargs to pass to the env constructor call
    """
    try:
        env_class = get_env_cls(env_specs["env_creator"])
    except KeyError:
        print("Unknown env name: {}".format(env_specs["env_creator"]))

    env = env_class(vehicle_ids=vehicle_ids, **env_specs)

    return env


def get_envs(
    env_specs,
    env_wrapper=None,
    vehicle_ids_list=None,
    env_num=1,
    wait_num=None,
    auto_reset=False,
    seed=None,
    **kwargs,
):
    """
    env_specs:
        env_name: 'mujoco'
        scenario_name: 'halfcheetah'
        env_kwargs: {} # kwargs to pass to the env constructor call
    """

    if env_wrapper is None:
        env_wrapper = ProxyEnv

    try:
        env_class = get_env_cls(env_specs["env_creator"])
    except KeyError:
        print("Unknown env name: {}".format(env_specs["env_creator"]))

    if env_num == 1:
        print("\n WARNING: Single environment detected, wrap to DummyVectorEnv.\n")
        envs = DummyVectorEnv(
            [
                lambda i=i: env_wrapper(
                    env_class(vehicle_ids=vehicle_ids_list[i], **env_specs)
                )
                for i in range(env_num)
            ],
            auto_reset=auto_reset,
            **kwargs,
        )

    else:
        envs = SubprocVectorEnv(
            [
                lambda i=i: env_wrapper(
                    env_class(vehicle_ids=vehicle_ids_list[i], **env_specs)
                )
                for i in range(env_num)
            ],
            wait_num=wait_num,
            auto_reset=auto_reset,
            **kwargs,
        )

    envs.seed(seed)
    return envs
