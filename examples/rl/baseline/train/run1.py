# https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo_atari.py

import sys
import time
from pathlib import Path
# Required to load inference module
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathlib import Path
# Load inference module to register agent
import inference

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import yaml

from torch.utils.tensorboard import SummaryWriter

# `contrib_policy` package is accessed from pip installed packages
from contrib_policy.format_action import FormatAction
from contrib_policy.policy import Model
from contrib_policy.utils import objdict
from smarts.zoo import registry


def make_env(
    env_id, scenario, agent_interface, config, seed, idx, capture_video, run_name
):
    def thunk():
        env = gym.make(
            env_id,
            scenario=scenario,
            agent_interface=agent_interface,
            seed=seed,
            sumo_headless=not config.sumo_gui,  # If False, enables sumo-gui display.
            headless=not config.head,  # If False, enables Envision display.
        )
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        # if config.capture_video:
        #     if idx == 0:
        #         env = gym.wrappers.RecordVideo(env, f"{parent_dir}/videos/{run_name}")

        # env = NoopResetEnv(env, noop_max=30)
        # env = MaxAndSkipEnv(env, skip=4)
        # env = EpisodicLifeEnv(env)
        # if "FIRE" in env.unwrapped.get_action_meanings():
        # env = FireResetEnv(env)
        # env = ClipRewardEnv(env)
        # env = gym.wrappers.ResizeObservation(env, (84, 84))
        # env = gym.wrappers.GrayScaleObservation(env)
        # env = gym.wrappers.FrameStack(env, 4)

        return env

    return thunk


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
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
    )

    # Torch seeding and device
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic
    config.device = torch.device(
        "cuda" if torch.cuda.is_available() and config.cuda else "cpu"
    )

    # Build model
    config.action_space = FormatAction().action_space
    model = Model(in_channels=config.num_stack*3, out_actions=config.action_space.n).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)

    # Make agent specification
    agent_spec = registry.make(
        locator=config.agent_locator,
        agent_params={
            "config": config,
            "model": model,
        },
    )

    # Print model summary
    from torchsummary import summary
    summary(model, (config.num_stack*3,agent_spec.interface.top_down_rgb.height,agent_spec.interface.top_down_rgb.width))
    
    # Env setup
    scenario = config.scenarios[0]
    envs = make_env(
        env_id=config.env_id,
        scenario=scenario,
        agent_interface=agent_spec.interface,
        config=config,
        seed=config.seed + 1,
        idx=1,
        capture_video=config.capture_video,
        run_name=run_name,
    )()
    # assert isinstance(
    #     envs.action_space, gym.spaces.Discrete
    # ), "Only discrete action space is supported."

    # Build agents
    agents = {agent_id: agent_spec.build_agent() for agent_id in envs.agent_ids}

    # obs, info = envs.reset()
    # f = agents["Agent_0"].filter_obs.filter(obs["Agent_0"])
    # from matplotlib import pyplot as plt
    # plt.imshow(f)
    # plt.show()

    # Start driving
    global_step = 0
    start_time = time.time()
    obs, info = envs.reset()   
    next_obs = {}
    next_done = {}
    for agent_id, agent_obs in obs.items():
        next_obs[agent_id] = torch.Tensor(np.expand_dims(agents[agent_id].process(agent_obs),0)).to(config.device)
        next_done[agent_id] = torch.zeros(config.num_envs).to(config.device)
    num_updates = config.total_timesteps // config.batch_size

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if config.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * config.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Rollout
        for step in range(0, config.num_steps):
            global_step += 1 * config.num_envs

            # Store            
            actions = {}
            for agent_id, agent_obs in next_obs.items():
                agents[agent_id].store("obs",step,agent_obs)
                agents[agent_id].store("dones",step,next_done[agent_id])

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = model.get_action_and_value(x=agent_obs)
                    agents[agent_id].store("values",step,value.flatten())
                
                agents[agent_id].store("actions",step,action)
                agents[agent_id].store("logprobs",step,logprob)
                actions[agent_id] = agents[agent_id].format_action.format(action.cpu().numpy()[0])

            # TRY NOT TO MODIFY: execute the game and log data.
            next_done = {}
            next_obs, reward, terminated, truncated, info = envs.step(actions)

            for agent_id, agent_obs in next_obs.items():
                agent_reward_tensor = torch.tensor(reward[agent_id]).to(config.device).view(-1)
                agents[agent_id].store("rewards",step,agent_reward_tensor) 
                next_obs[agent_id] = torch.Tensor(np.expand_dims(agents[agent_id].process(agent_obs),0)).to(config.device)
                next_done[agent_id] = torch.Tensor([int(terminated[agent_id])]).to(config.device)  

            # for item in info:
            #     if "episode" in item.keys():
            #         print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
            #         writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
            #         writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
            #         break

        # # bootstrap value if not done
        # with torch.no_grad():
        #     next_value = agent.get_value(next_obs).reshape(1, -1)
        #     if config.gae:
        #         advantages = torch.zeros_like(rewards).to(config.device)
        #         lastgaelam = 0
        #         for t in reversed(range(config.num_steps)):
        #             if t == config.num_steps - 1:
        #                 nextnonterminal = 1.0 - next_done
        #                 nextvalues = next_value
        #             else:
        #                 nextnonterminal = 1.0 - dones[t + 1]
        #                 nextvalues = values[t + 1]
        #             delta = rewards[t] + config.gamma * nextvalues * nextnonterminal - values[t]
        #             advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
        #         returns = advantages + values
        #     else:
        #         returns = torch.zeros_like(rewards).to(config.device)
        #         for t in reversed(range(config.num_steps)):
        #             if t == config.num_steps - 1:
        #                 nextnonterminal = 1.0 - next_done
        #                 next_return = next_value
        #             else:
        #                 nextnonterminal = 1.0 - dones[t + 1]
        #                 next_return = returns[t + 1]
        #             returns[t] = rewards[t] + config.gamma * nextnonterminal * next_return
        #         advantages = returns - values

        # # flatten the batch
        # b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        # b_logprobs = logprobs.reshape(-1)
        # b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        # b_advantages = advantages.reshape(-1)
        # b_returns = returns.reshape(-1)
        # b_values = values.reshape(-1)

        # # Optimizing the policy and value network
        # b_inds = np.arange(config.batch_size)
        # clipfracs = []
        # for epoch in range(config.update_epochs):
        #     np.random.shuffle(b_inds)
        #     for start in range(0, config.batch_size, config.minibatch_size):
        #         end = start + config.minibatch_size
        #         mb_inds = b_inds[start:end]

        #         _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
        #         logratio = newlogprob - b_logprobs[mb_inds]
        #         ratio = logratio.exp()

        #         with torch.no_grad():
        #             # calculate approx_kl http://joschu.net/blog/kl-approx.html
        #             old_approx_kl = (-logratio).mean()
        #             approx_kl = ((ratio - 1) - logratio).mean()
        #             clipfracs += [((ratio - 1.0).abs() > config.clip_coef).float().mean().item()]

        #         mb_advantages = b_advantages[mb_inds]
        #         if config.norm_adv:
        #             mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        #         # Policy loss
        #         pg_loss1 = -mb_advantages * ratio
        #         pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
        #         pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        #         # Value loss
        #         newvalue = newvalue.view(-1)
        #         if config.clip_vloss:
        #             v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
        #             v_clipped = b_values[mb_inds] + torch.clamp(
        #                 newvalue - b_values[mb_inds],
        #                 -config.clip_coef,
        #                 config.clip_coef,
        #             )
        #             v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
        #             v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        #             v_loss = 0.5 * v_loss_max.mean()
        #         else:
        #             v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

        #         entropy_loss = entropy.mean()
        #         loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef

        #         optimizer.zero_grad()
        #         loss.backward()
        #         nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
        #         optimizer.step()

        #     if config.target_kl is not None:
        #         if approx_kl > config.target_kl:
        #             break

        # y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        # var_y = np.var(y_true)
        # explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # # TRY NOT TO MODIFY: record rewards for plotting purposes
        # writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        # writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        # writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        # writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        # writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        # writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        # writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        # writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # print("SPS:", int(global_step / (time.time() - start_time)))
        # writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
