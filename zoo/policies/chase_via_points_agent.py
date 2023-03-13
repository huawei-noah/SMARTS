from smarts.core.agent import Agent
from smarts.core.observations import Observation


class ChaseViaPointsAgent(Agent):
    def act(self, obs: Observation):
        for ind, wp in enumerate(obs.waypoint_paths):
            print("+ Waypoint:", ind)
            print("  Waypoints= ",wp[0].pos,wp[0].lane_id)
            print("  Waypoints= ",wp[-1].pos,wp[-1].lane_id)
        print("+ Leader: ",obs.ego_vehicle_state.lane_id, obs.ego_vehicle_state.position)
        print("+ NVP= ",obs.via_data.near_via_points)
        print("+ Hit= ",obs.via_data.hit_via_points)        

        # if len(obs.via_data.near_via_points) < 1:
        #     # No nearby via points
        #     return (obs.waypoint_paths[0][0].speed_limit, 0)

        if (
            len(obs.via_data.near_via_points) < 1
            or obs.ego_vehicle_state.road_id != obs.via_data.near_via_points[0].road_id
        ):
            print("No via points or road id is different \n")
            input()
            return (obs.waypoint_paths[0][0].speed_limit, 0)

        nearest = obs.via_data.near_via_points[0]
        if nearest.lane_index == obs.ego_vehicle_state.lane_index:
            print("Nearest lane index matched ego road id \n")
            input()
            return (nearest.required_speed, 0)

        print("Changing lane \n")
        input()
        return (
            nearest.required_speed,
            1 if nearest.lane_index > obs.ego_vehicle_state.lane_index else -1,
        )
