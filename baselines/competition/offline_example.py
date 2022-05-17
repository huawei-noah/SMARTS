from pathlib import Path
from baselines.competition.offline_competition_env import OfflineCompetitionEnv
from smarts.core.agent import Agent


def act(obs, **conf):
    return (1, 1)


agent = Agent.from_function(agent_function=act)


def main(max_steps):
    env = OfflineCompetitionEnv(
        scenarios=["scenarios/straight"], max_episode_steps=max_steps
    )

    obs = env.reset()
    done = False

    while not done:
        obs["target_position"] = [0.4, 0, 0]  # can manually modify the observation here
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        # smarts_obs = info["env_obs"] # full observations for debugging if needed
        score = info["score"]  # TODO: metrics for environment score

    env.close()


if __name__ == "__main__":
    main(max_steps=10)
