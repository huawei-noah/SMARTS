py_driver.PyDriver(
    env,
    py_tf_eager_policy.PyTFEagerPolicy(random_policy, use_tf_function=True),
    [rb_observer],
    max_steps=initial_collect_steps,
).run(train_py_env.reset())
