try:
    import ray
except Exception as e:
    from smarts.core.utils.custom_exceptions import RayException

    raise RayException.required_to("stress_sumo.py")

from smarts.core.scenario import Scenario
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation


@ray.remote
def spawn_sumo(worker_idx, batch_id):
    sumo_sim = SumoTrafficSimulation(headless=True)

    scenarios_iterator = Scenario.scenario_variations(
        ["scenarios/sumo/loop"],
        ["Agent007"],
    )
    sumo_sim.setup(Scenario.next(scenarios_iterator, f"{batch_id}-{worker_idx}"))
    sumo_sim.teardown()


remotes_per_iteration = 32
ray.init()
for i in range(100):
    ray.wait(
        [spawn_sumo.remote(r, i) for r in range(remotes_per_iteration)],
        num_returns=remotes_per_iteration,
    )
ray.shutdown()
