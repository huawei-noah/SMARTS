import argparse

import gym

if __name__ == "__main__":
    parser = argparse.ArgumentParser("hiway-egoless-example")
    # parser.add_argument("replay_data", help="Replay data path", type=str)

    parser.add_argument(
        "scenarios",
        help="A list of scenarios. Each element can be either the scenario to run "
        "(see scenarios/ for some samples you can use) OR a directory of scenarios "
        "to sample from.",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--headless", help="run simulation in headless mode", action="store_true"
    )
    args = parser.parse_args()

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=args.scenarios,
        agent_specs={},
        headless=args.headless,
        visdom=False,
        timestep_sec=0.1,
    )

    for i in range(1):
        env.reset()

        for _ in range(600):
            print("step", _)
            env.step({})

    env.close()
