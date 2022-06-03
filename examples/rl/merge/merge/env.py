from typing import Any, Dict

from smarts.env.wrappers.frame_stack import FrameStack
from merge import action as merge_action
from merge.observation import Concatenate, FilterObs
from merge.reward import Reward
from tf_agents.environments import suite_gym, tf_py_environment, validate_py_environment
from smarts.env.wrappers.format_obs import FormatObs
from smarts.env.wrappers.format_action import FormatAction
from smarts.core.controllers import ActionSpaceType
from smarts.env.wrappers.single_agent import SingleAgent

def make(config: Dict[str, Any]) -> tf_py_environment.TFPyEnvironment:
    # Create TF environment.
    # Refer to https://www.tensorflow.org/agents/tutorials/2_environments_tutorial
    gym_action_space = lambda env: FormatAction(env=env, space=ActionSpaceType[config["action_space"]])
    action_wrapper = lambda env: merge_action.Action(
        env=env, space=config["action_wrapper"]
    )
    frame_stack = lambda env: FrameStack(env=env, num_stack=config["num_stack"])
    pyenv = suite_gym.load(
        environment_name="smarts.env:merge-v0",
        gym_env_wrappers=[
            FormatObs,
            gym_action_space,
            Reward,
            action_wrapper,
            FilterObs,
            frame_stack,
            # Concatenate,
            SingleAgent,
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
