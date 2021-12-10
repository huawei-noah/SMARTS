import gym
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

scenarios = ['examples/stable_baselines3/scenarios/loop']

def create_env():

    vehicle_interface = smarts_agent_interface.AgentInterface(
        max_episode_steps=300,
        rgb=smarts_agent_interface.RGB(
            width=64,
            height=64,
            resolution=1,
        ),
        action=getattr(
            smarts_controllers.ActionSpaceType,
            "Continuous",
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
        visdom=False,
        seed=42,
        sim_name="env",
    )

    env = smarts_rgb_image.RGBImage(env=env, num_stack=1)
    env = action.Action(env=env)
    env = smarts_single_agent.SingleAgent(env)
    check_env(env, warn=True)

    return env

def main(evaluate=False, model_path=None):

    if evaluate:
        model = PPO.load(model_path)
        env = create_env()
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

    parser = default_argument_parser("single-agent-example")
    parser.add_argument(
        "--evaluate",
        help="Evaluate a trained model",
        action="store_true",
    )
    parser.add_argument(
        "--model-path",
        help="Path of trained model",
        type=str
    )
    args = parser.parse_args()
    
    main(
        evaluate=args.evaluate,
        model_path=args.model_path,
    )