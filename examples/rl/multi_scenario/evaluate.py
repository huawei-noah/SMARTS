def evaluate(env, policy, config):
    total_return = 0.0
    for _ in range(config["eval"]["episodes"]):
        time_step = env.reset()
        ep_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = env.step(action_step.action)
            ep_return += time_step.reward

        # print(f"Eval episode {ep} return: {ep_return.numpy()[0]:.2f}")
        total_return += ep_return

    avg_return = total_return / config["eval"]["episodes"]

    print(f"Evaluating. Episode average return: {avg_return.numpy()[0]:.2f}")

    return


# for overtakeing scenario, the agent must start and stop at the same lane
#
