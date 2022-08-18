import logging
import os
import shutil
import stat
import sys

import gym
from argument_parser import default_argument_parser

from smarts.core.utils.episodes import episodes
from smarts.zoo.registry import make as zoo_make

logging.basicConfig(level=logging.INFO)

AGENT_ID = "Agent-007"


def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
        shutil.copystat(src, dst)
    lst = os.listdir(src)
    if ignore:
        excl = ignore(src, lst)
        lst = [x for x in lst if x not in excl]
    for item in lst:
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if symlinks and os.path.islink(s):
            if os.path.lexists(d):
                os.remove(d)
            os.symlink(os.readlink(s), d)
            try:
                st = os.lstat(s)
                mode = stat.S_IMODE(st.st_mode)
                os.lchmod(d, mode)
            except Exception as e:
                print(e)
                pass  # lchmod not available
        elif os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def copy_scenarios(save_dir, scenarios):
    for i in range(len(scenarios)):
        new_scenario_location = os.path.join(save_dir, scenarios[i])
        if not os.path.exists(new_scenario_location):
            copytree(scenarios[i], new_scenario_location)
        scenarios[i] = new_scenario_location


def main(scenarios, sim_name, headless, seed, speed, max_steps, save_dir, write):
    from zoo import policies

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    policies.replay_save_dir = save_dir
    policies.replay_read = not write

    # This is how you can wrap an agent in replay-agent-v0 wrapper to store and load its inputs and actions
    # and replay it
    agent_spec = zoo_make(
        "zoo.policies:replay-agent-v0",
        save_directory=save_dir,
        id="agent_007",
        wrapped_agent_locator="zoo.policies:keep-left-with-speed-agent-v0",
        wrapped_agent_params={"speed": speed},
    )
    # copy the scenarios to the replay directory to make sure it's not changed
    copy_scenarios(save_dir, scenarios)

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={AGENT_ID: agent_spec},
        sim_name=sim_name,
        headless=headless,
        visdom=False,
        timestep_sec=0.1,
        sumo_headless=True,
        seed=seed,
    )

    # Carry out the experiment
    episode = next(episodes(n=1))
    agent = agent_spec.build_agent()
    observations = env.reset()

    dones = {"__all__": False}
    MAX_STEPS = 2550
    i = 0
    try:
        while not dones["__all__"] and i < max_steps:
            agent_obs = observations[AGENT_ID]
            agent_action = agent.act(agent_obs)
            observations, rewards, dones, infos = env.step({AGENT_ID: agent_action})
            i += 1
            if i % 10 == 0:
                print("Step: ", i)
            episode.record_step(observations, rewards, dones, infos)
    except KeyboardInterrupt:
        # discard result
        i = MAX_STEPS
    finally:
        if dones["__all__"]:
            i = MAX_STEPS
        try:
            episode.record_scenario(env.scenario_log)
            env.close()
        finally:
            sys.exit(i // 10)


if __name__ == "__main__":
    parser = default_argument_parser("klws-agent-example")
    parser.add_argument(
        "--speed",
        help="The speed param for the vehicle.",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max-steps",
        help="The maximum number of steps.",
        type=int,
        default=1500,
    )

    # Along with any additional arguments these two arguments need to be added to pass the directory where the agent
    # inputs and actions will be store and whether to replay the agent or write out its action to the directory
    parser.add_argument(
        "--save-dir",
        help="The save directory location.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--write",
        help="Replay the agent else write the agent actions out to directory.",
        action="store_true",
    )

    args = parser.parse_args()

    main(
        scenarios=args.scenarios,
        sim_name=args.sim_name,
        headless=args.headless,
        seed=args.seed,
        speed=args.speed,
        max_steps=args.max_steps,
        save_dir=args.save_dir,
        write=args.write,
    )
