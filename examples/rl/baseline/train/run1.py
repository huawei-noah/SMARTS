# https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo_atari.py

import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import sys
from pathlib import Path
# Required to load inference module
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Load inference module to register agent
import inference
# `contrib_policy` package is accessed from pip installed packages
from contrib_policy.utils import objdict

# from env import make as make_env
from smarts.zoo import registry
import yaml
from pathlib import Path

# from stable_baselines3.common.atari_wrappers import (  # isort:skip
#     ClipRewardEnv,
#     EpisodicLifeEnv,
#     FireResetEnv,
#     MaxAndSkipEnv,
#     NoopResetEnv,
# )

# def make_env(env_id, seed, idx, capture_video, run_name):
#     def thunk():
#         env = gym.make(env_id)
#         env = gym.wrappers.RecordEpisodeStatistics(env)
#         if capture_video:
#             if idx == 0:
#                 env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
#         env = NoopResetEnv(env, noop_max=30)
#         env = MaxAndSkipEnv(env, skip=4)
#         env = EpisodicLifeEnv(env)
#         if "FIRE" in env.unwrapped.get_action_meanings():
#             env = FireResetEnv(env)
#         env = ClipRewardEnv(env)
#         env = gym.wrappers.ResizeObservation(env, (84, 84))
#         env = gym.wrappers.GrayScaleObservation(env)
#         env = gym.wrappers.FrameStack(env, 4)
#         env.seed(seed)
#         env.action_space.seed(seed)
#         env.observation_space.seed(seed)
#         return env

#     return thunk


# def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
#     torch.nn.init.orthogonal_(layer.weight, std)
#     torch.nn.init.constant_(layer.bias, bias_const)
#     return layer


# class Agent(nn.Module):
#     def __init__(self, envs):
#         super(Agent, self).__init__()
#         self.network = nn.Sequential(
#             layer_init(nn.Conv2d(4, 32, 8, stride=4)),
#             nn.ReLU(),
#             layer_init(nn.Conv2d(32, 64, 4, stride=2)),
#             nn.ReLU(),
#             layer_init(nn.Conv2d(64, 64, 3, stride=1)),
#             nn.ReLU(),
#             nn.Flatten(),
#             layer_init(nn.Linear(64 * 7 * 7, 512)),
#             nn.ReLU(),
#         )
#         self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
#         self.critic = layer_init(nn.Linear(512, 1), std=1)

#     def get_value(self, x):
#         return self.critic(self.network(x / 255.0))

#     def get_action_and_value(self, x, action=None):
#         hidden = self.network(x / 255.0)
#         logits = self.actor(hidden)
#         probs = Categorical(logits=logits)
#         if action is None:
#             action = probs.sample()
#         return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


if __name__ == "__main__":
    # Load config file
    parent_dir = Path(__file__).resolve().parent
    config_file = yaml.safe_load((parent_dir / "config.yaml").read_text())
    config = objdict(config_file["smarts"])
    config.batch_size = int(config.num_envs * config.num_steps)
    config.minibatch_size = int(config.batch_size // config.num_minibatches)

    # Tensorboard
    run_name = f"{config.env_id}__{config.exp_name}__{config.seed}__{int(time.time())}"
    writer = SummaryWriter(f"{parent_dir}/logs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
    )

    # Seeding
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic

    # Torch device
    device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")

    # Create env
    agent_interface=registry.make(locator=config.agent_locator).interface
    env = gym.make(
        config.env_id,
        scenario=config.scenarios[0],
        agent_interface=agent_interface,
        seed=config.seed,
        sumo_headless=not config.sumo_gui,  # If False, enables sumo-gui display.
        headless=not config.head,  # If False, enables Envision display.
    ) 
 
    # # Wrap the environment
    # for wrapper in wrappers:
    #     env = wrapper(env)

    # Make agent
    agents = {
        agent_id: registry.make_agent(locator=config.agent_locator)
        for agent_id in env.agent_ids
    }

    env.close()


    # Env setup
    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(config.env_id, config.seed + i, i, config.capture_video, run_name) for i in range(config.num_envs)]
    # )
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "Only discrete action space is supported."

    # agent = Agent(envs).to(device)
    # optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # # ALGO Logic: Storage setup
    # obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    # actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    # logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # # TRY NOT TO MODIFY: start the game
    # global_step = 0
    # start_time = time.time()
    # next_obs = torch.Tensor(envs.reset()).to(device)
    # next_done = torch.zeros(args.num_envs).to(device)
    # num_updates = args.total_timesteps // args.batch_size

    # for update in range(1, num_updates + 1):
    #     # Annealing the rate if instructed to do so.
    #     if args.anneal_lr:
    #         frac = 1.0 - (update - 1.0) / num_updates
    #         lrnow = frac * args.learning_rate
    #         optimizer.param_groups[0]["lr"] = lrnow

    #     for step in range(0, args.num_steps):
    #         global_step += 1 * args.num_envs
    #         obs[step] = next_obs
    #         dones[step] = next_done

    #         # ALGO LOGIC: action logic
    #         with torch.no_grad():
    #             action, logprob, _, value = agent.get_action_and_value(next_obs)
    #             values[step] = value.flatten()
    #         actions[step] = action
    #         logprobs[step] = logprob

    #         # TRY NOT TO MODIFY: execute the game and log data.
    #         next_obs, reward, done, info = envs.step(action.cpu().numpy())
    #         rewards[step] = torch.tensor(reward).to(device).view(-1)
    #         next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

    #         for item in info:
    #             if "episode" in item.keys():
    #                 print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
    #                 writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
    #                 writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
    #                 break

    #     # bootstrap value if not done
    #     with torch.no_grad():
    #         next_value = agent.get_value(next_obs).reshape(1, -1)
    #         if args.gae:
    #             advantages = torch.zeros_like(rewards).to(device)
    #             lastgaelam = 0
    #             for t in reversed(range(args.num_steps)):
    #                 if t == args.num_steps - 1:
    #                     nextnonterminal = 1.0 - next_done
    #                     nextvalues = next_value
    #                 else:
    #                     nextnonterminal = 1.0 - dones[t + 1]
    #                     nextvalues = values[t + 1]
    #                 delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
    #                 advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
    #             returns = advantages + values
    #         else:
    #             returns = torch.zeros_like(rewards).to(device)
    #             for t in reversed(range(args.num_steps)):
    #                 if t == args.num_steps - 1:
    #                     nextnonterminal = 1.0 - next_done
    #                     next_return = next_value
    #                 else:
    #                     nextnonterminal = 1.0 - dones[t + 1]
    #                     next_return = returns[t + 1]
    #                 returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
    #             advantages = returns - values

    #     # flatten the batch
    #     b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    #     b_logprobs = logprobs.reshape(-1)
    #     b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    #     b_advantages = advantages.reshape(-1)
    #     b_returns = returns.reshape(-1)
    #     b_values = values.reshape(-1)

    #     # Optimizing the policy and value network
    #     b_inds = np.arange(args.batch_size)
    #     clipfracs = []
    #     for epoch in range(args.update_epochs):
    #         np.random.shuffle(b_inds)
    #         for start in range(0, args.batch_size, args.minibatch_size):
    #             end = start + args.minibatch_size
    #             mb_inds = b_inds[start:end]

    #             _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
    #             logratio = newlogprob - b_logprobs[mb_inds]
    #             ratio = logratio.exp()

    #             with torch.no_grad():
    #                 # calculate approx_kl http://joschu.net/blog/kl-approx.html
    #                 old_approx_kl = (-logratio).mean()
    #                 approx_kl = ((ratio - 1) - logratio).mean()
    #                 clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

    #             mb_advantages = b_advantages[mb_inds]
    #             if args.norm_adv:
    #                 mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

    #             # Policy loss
    #             pg_loss1 = -mb_advantages * ratio
    #             pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    #             pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    #             # Value loss
    #             newvalue = newvalue.view(-1)
    #             if args.clip_vloss:
    #                 v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
    #                 v_clipped = b_values[mb_inds] + torch.clamp(
    #                     newvalue - b_values[mb_inds],
    #                     -args.clip_coef,
    #                     args.clip_coef,
    #                 )
    #                 v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
    #                 v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
    #                 v_loss = 0.5 * v_loss_max.mean()
    #             else:
    #                 v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

    #             entropy_loss = entropy.mean()
    #             loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

    #             optimizer.zero_grad()
    #             loss.backward()
    #             nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
    #             optimizer.step()

    #         if args.target_kl is not None:
    #             if approx_kl > args.target_kl:
    #                 break

    #     y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    #     var_y = np.var(y_true)
    #     explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    #     # TRY NOT TO MODIFY: record rewards for plotting purposes
    #     writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
    #     writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    #     writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    #     writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    #     writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
    #     writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    #     writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    #     writer.add_scalar("losses/explained_variance", explained_var, global_step)
    #     print("SPS:", int(global_step / (time.time() - start_time)))
    #     writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    # envs.close()
    # writer.close()