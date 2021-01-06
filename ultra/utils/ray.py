import ray
from packaging import version


def default_ray_kwargs(**kwargs):
    ray_kwargs = {}
    _system_config = {
        # Needed to deal with cores that lock up for > 10 seconds
        "num_heartbeats_timeout": 10000,
        # "raylet_heartbeat_timeout_milliseconds": 10,
        # "object_timeout_milliseconds": 200,
    }
    if version.parse(ray.__version__) > version.parse("0.8"):
        ray_kwargs["_system_config"] = _system_config
    else:
        ray_kwargs["_internal_config"] = _system_config
    ray_kwargs.update(kwargs)

    return ray_kwargs


# TODO: Perhaps start cluster manually instead of `ray.init()`
# See https://github.com/ray-project/ray/blob/174bef56d452b6f86db167ecb80e7f23176079b6/python/ray/tests/conftest.py#L110
