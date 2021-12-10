import gym
import pathlib
import numpy as np
from stable_baselines3 import PPO
from smarts.core import agent as smarts_agent
from smarts.core import agent_interface as smarts_agent_interface
from smarts.core import controllers as smarts_controllers
from smarts.env import hiway_env as smarts_hiway_env
import env.rgb_image as smarts_rgb_image
import env.single_agent as smarts_single_agent
import env.adapter as adapter
import env.action as action

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from examples.argument_parser import default_argument_parser
from ruamel.yaml import YAML

scenarios = ['examples/stable_baselines3/scenarios/loop']
yaml = YAML(typ="safe")

def create_env(config_env):

    vehicle_interface = smarts_agent_interface.AgentInterface(
        max_episode_steps=config_env["max_episode_steps"],
        rgb=smarts_agent_interface.RGB(
            width=config_env["rgb_pixels"],
            height=config_env["rgb_pixels"],
            resolution=config_env["rgb_meters"] / config_env["rgb_pixels"],
        ),
        action=getattr(
            smarts_controllers.ActionSpaceType,
            config_env["action_space_type"],
        ),
        done_criteria=smarts_agent_interface.DoneCriteria(
            collision=True,
            off_road=True,
            off_route=False,
            on_shoulder=False,
            wrong_way=False,
            not_moving=False,
        ),
    )

    agent_specs = {
        "agent": smarts_agent.AgentSpec(
            interface=vehicle_interface,
            agent_builder=None,
            reward_adapter=adapter.reward_adapter,
            info_adapter=adapter.info_adapter,
        )
    }

    env = smarts_hiway_env.HiWayEnv(
        scenarios=scenarios,
        agent_specs=agent_specs,
        headless=False,
        visdom=config_env["visdom"],
        seed=config_env["seed"],
        sim_name="env",
    )

    env = smarts_rgb_image.RGBImage(env=env, num_stack=1)
    env = action.Action(env=env)
    env = smarts_single_agent.SingleAgent(env)
    check_env(env, warn=True)

    return env

def main(args):

    name = "smarts"
    config_env = yaml.load(
        (pathlib.Path(__file__).absolute().parent / "config.yaml").read_text()
    )
    config_env = config_env[name]
    config_env["headless"] = not args.head
    print(config_env)

    if args.mode == 'evaluate':
        model = PPO.load(args.logdir)
        env = create_env(config_env)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
        print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    else:
        env = create_env()
        model = PPO("CnnPolicy", env, verbose=1)

        before_mean_reward, before_std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
        model.learn(total_timesteps=20000)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
        print(f"before_mean_reward:{before_mean_reward:.2f} +/- {before_std_reward:.2f}")
        print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

        # save trained model
        from datetime import datetime
        date_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        save_path = 'examples/stable_baselines3/logs/' + date_time
        model.save(save_path)

if __name__ == "__main__":

    parser = default_argument_parser("stable-baselines-3")
    parser.add_argument(
        "--mode",
        help="`train` or `evaluate`. Default is `train`.",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--logdir",
        help="Directory path to saved RL model. Required if `--mode=evaluate`, else optional.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--head", help="Run the simulation with display.", action="store_true"
    )
    args = parser.parse_args()
    
    main(args)