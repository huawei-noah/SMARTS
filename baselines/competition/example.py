from baselines.competition.competition_env import CompetitionEnv
from smarts.core.agent import Agent


def act(obs, **conf):
    return (1, 1)


agent = Agent.from_function(agent_function=act)


def main(max_steps):
    env = CompetitionEnv()

    obs = env.reset()
    done = False

    steps = 0
    while not done and steps < max_steps:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        steps += 1

    env.close()


if __name__ == "__main__":
    main(max_steps=1000)
