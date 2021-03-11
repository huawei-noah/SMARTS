from envision.client import Client as Envision
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.utils.traffic_history_serivce import Traffic_history_service
from time import sleep

# for i in range(2):
#     smarts = SMARTS(
#         agent_interfaces={},
#         traffic_sim=SumoTrafficSimulation(headless=True, auto_start=True),
#         envision=Envision(),
#     )
#     smarts.destroy()


t = Traffic_history_service('/home/kyber/datasets/imitation-learning/scenarios/interaction_dataset_merging/traffic_history_000.json')
t.teardown()
t.teardown()