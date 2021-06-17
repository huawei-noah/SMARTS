# ULTRA Environment

ULTRA's Gym and RLlib environments inherit from SMARTS' `smarts.env.hiway_env.HiWayEnv`
and `smarts.env.rllib_hiway_env.RLlibHiWayEnv` respectively.

ULTRA's Gym environment differs in two ways from SMARTS' HiWayEnv:
1. ULTRA's Gym environment takes a `scenario_info` parameter that is used to specify a
specific task and level that describe the type of scenarios the environment will use.
2. ULTRA's Gym environment performs automatic framestacking of certain parts of the
environment observation it receives from SMARTS. Currently, if an agent has a
`TopDownRGB` in its observation, the ULTRA environment will change the `data` attribute
of this class to still be a NumPy array, but with shape
`(_STACK_SIZE, HEIGHT, WIDTH, 3)` where `_STACK_SIZE` is the number of frames to stack
(its value can be found in `ultra.env.ultra_env`), `HEIGHT` is the height of the RGB
image, and `WIDTH` is the width of the RGB image.

ULTRA's RLlib environment differs in one way from SMARTS' RLlibHiWayEnv:
1. ULTRA's RLlib environment takes a `scenario_info` key as part of its config. This
`scenario_info` key is used to specify a specific task and level that describe the type
of scenarios the environment will use.
