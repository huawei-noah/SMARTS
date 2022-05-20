from typing import Any, Dict

from gym.wrappers.frame_stack import FrameStack
from merge import action as merge_action
from merge import observation as merge_observation
from merge import reward as merge_reward
from tf_agents.environments import (
    PyEnvironment,
    suite_gym,
    tf_py_environment,
    validate_py_environment,
)


def make(config: Dict[str, Any]) -> PyEnvironment:
    # Create TF environment.
    # Refer to https://www.tensorflow.org/agents/tutorials/2_environments_tutorial
    gym_reward_wrapper = merge_reward.Reward
    gym_action_wrapper = lambda env: merge_action.Action(
        env=env, space=config["action_wrapper"]
    )
    gym_obs_wrapper = merge_observation.RGB
    gym_frame_stack = lambda env: FrameStack(env=env, num_stack=config["num_stack"])
    gym_frame_concatenate = merge_observation.Concatenate
    pyenv = suite_gym.load(
        environment_name="Merge-v0",
        gym_env_wrappers=[
            gym_reward_wrapper,
            gym_action_wrapper,
            gym_obs_wrapper,
            gym_frame_stack,
            gym_frame_concatenate,
        ],
        gym_kwargs={
            "headless": not config["head"],  # If False, enables Envision display.
            "visdom": config["visdom"],  # If True, enables Visdom display.
            "sumo_headless": not config[
                "sumo_gui"
            ],  # If False, enables sumo-gui display.
            "img_meters": config["img_meters"],
            "img_pixels": config["img_pixels"],
        },
    )
    validate_py_environment(environment=pyenv)
    # (Optional) Manually verify Py env spaces
    # print('action_spec:', pyenv.action_spec())
    # print('time_step_spec.observation:', pyenv.time_step_spec().observation)
    # print('time_step_spec.step_type:', pyenv.time_step_spec().step_type)
    # print('time_step_spec.discount:', pyenv.time_step_spec().discount)
    # print('time_step_spec.reward:', pyenv.time_step_spec().reward)

    tfenv = tf_py_environment.TFPyEnvironment(pyenv)
    # (Optional) Manually verify TF env specs
    # print(isinstance(tfenv, tf_environment.TFEnvironment))
    # print("TimeStep Specs:", tfenv.time_step_spec())
    # print("Action Specs:", tfenv.action_spec())

    return tfenv
