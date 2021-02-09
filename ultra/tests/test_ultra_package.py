import os, shutil, ray

from ultra.train import train
from ultra.scenarios.generate_scenarios import build_scenarios

if __name__ == '__main__':
    save_dir = "tests/scenarios/maps/no-traffic/"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    
    build_scenarios(
        task="task00",
        level_name="eval_test",
        stopwatcher_behavior=None,
        stopwatcher_route=None,
        root_path="tests/scenarios",
        save_dir=save_dir,
    )
    
    policy_class = "ultra.baselines.sac:sac-v0"

    ray.shutdown()
    try:
        ray.init(ignore_reinit_error=True)
        ray.wait(
            [
                train.remote(
                    scenario_info=("00", "eval_test"),
                    policy_class=policy_class,
                    num_episodes=1,
                    eval_info={"eval_rate": 1000, "eval_episodes": 2,},
                    timestep_sec=0.1,
                    headless=True,
                    seed=2,
                    log_dir="ultra/tests/logs",
                )
            ]
        )
        ray.shutdown()
    except ray.exceptions.WorkerCrashedError as err:
        print(err)
        ray.shutdown()

