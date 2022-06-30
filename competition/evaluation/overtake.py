# class _Overtake():
#     def __init__(self, agent_names):
#         self._traj = {name: [[],[]] for name in agent_names}

#     def reinit(self):
#         self._traj = {name: [[],[]] for name in self._overtake.keys()}

#     def __call__(self, obs: Observation, agent_name: str):
#         lane_index = obs.ego_vehicle_state.lane_index
#         lane_index = obs.ego_vehicle_state.lane_index

#         self._traj[agent_name].append(lane_index)

#     def check(agent_name: str):
#         overtake = 0

#         return {"overtake": overtake}
