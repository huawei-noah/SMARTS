from smarts.sstudio.types import CutIn, UTurn


class AgentBehaviour:
    """Agent behavior configuration and waypoint generation for uturn and cutin tasks"""

    def __init__(aggressiveness: int = 0):
        """The aggressiveness affects the waypoints of a mission task, in terms of
        the trigger timing.
        """
        self.aggressiveness = aggressiveness
        # Uturn specific parameters
        self._uturn_initial_heading = 0
        self._uturn_initial_distant = 0
        self._uturn_initial_velocity = 0
        self._uturn_initial_height = 0
        self._insufficient_initial_distant = False
        self._uturn_initial_position = 0
        self._uturn_is_initialized = False
        self._prev_kyber_x_position = None
        self._prev_kyber_y_position = None
        self._first_uturn = True
    
    def apply_custom_behaviour(self, sim, vehicle, mission):
        if isinstance(mission.task, UTurn):
            return self.uturn_waypoints(
                sim, vehicle.pose, vehicle
            )
        elif isinstance(mission.task, CutIn):
            return self.cut_in_waypoints(
                sim, vehicle.pose, vehicle
            )
    
    """Needs:
    self._road_network
    self._waypoints
    """
    def uturn_waypoints(self, sim, pose: Pose, vehicle):
        # TODO: 1. Need to revisit the approach to calculate the U-Turn trajectory.
        #       2. Wrap this method in a helper.

        ## the position of ego car is here: [x, y]
        ego_position = pose.position[:2]
        ego_lane = self._road_network.nearest_lane(ego_position)
        ego_wps = self._waypoints.waypoint_paths_on_lane_at(
            ego_position, ego_lane.getID(), 60
        )
        if self._mission.task.initial_speed is None:
            default_speed = ego_wps[0][0].speed_limit
        else:
            default_speed = self._mission.task.initial_speed
        ego_wps_des_speed = []
        for px in range(len(ego_wps[0])):
            new_wp = replace(ego_wps[0][px], speed_limit=default_speed)
            ego_wps_des_speed.append(new_wp)

        ego_wps_des_speed = [ego_wps_des_speed]
        neighborhood_vehicles = sim.neighborhood_vehicles_around_vehicle(
            vehicle=vehicle, radius=140
        )

        if not neighborhood_vehicles:
            return ego_wps_des_speed

        n_lane = self._road_network.nearest_lane(
            neighborhood_vehicles[0].pose.position[:2]
        )
        start_lane = self._road_network.nearest_lane(
            self._mission.start.position,
            include_junctions=False,
            include_special=False,
        )
        start_edge = self._road_network.road_edge_data_for_lane_id(start_lane.getID())
        oncoming_edge = start_edge.oncoming_edges[0]
        oncoming_lanes = oncoming_edge.getLanes()
        lane_id_list = []
        for idx in oncoming_lanes:
            lane_id_list.append(idx.getID())

        if n_lane.getID() not in lane_id_list:
            return ego_wps_des_speed
        # The aggressiveness is mapped from [0,10] to [0,0.8] domain which
        # represents the portion of intitial distantce which is used for
        # triggering the u-turn task.
        aggressiveness = 0.3 + 0.5 * self.aggressiveness / 10
        distance_threshold = 8

        if not self._uturn_is_initialized:
            self._uturn_initial_distant = (
                -vehicle.pose.position[0] + neighborhood_vehicles[0].pose.position[0]
            )

            self._uturn_initial_velocity = neighborhood_vehicles[0].speed
            self._uturn_initial_height = 1 * (
                neighborhood_vehicles[0].pose.position[1] - vehicle.pose.position[1]
            )

            if (1 * self._uturn_initial_height * 3.14 / 13.8) * neighborhood_vehicles[
                0
            ].speed + distance_threshold > self._uturn_initial_distant:
                self._insufficient_initial_distant = True
            self._uturn_is_initialized = True

        horizontal_distant = (
            -vehicle.pose.position[0] + neighborhood_vehicles[0].pose.position[0]
        )
        vertical_distant = (
            neighborhood_vehicles[0].pose.position[1] - vehicle.pose.position[1]
        )

        if self._insufficient_initial_distant is True:
            if horizontal_distant > 0:
                return ego_wps_des_speed
            else:
                self._task_is_triggered = True

        if (
            horizontal_distant > 0
            and self._task_is_triggered is False
            and horizontal_distant
            > (1 - aggressiveness) * (self._uturn_initial_distant - 1)
            + aggressiveness
            * (
                (1 * self._uturn_initial_height * 3.14 / 13.8)
                * neighborhood_vehicles[0].speed
                + distance_threshold
            )
        ):
            return ego_wps_des_speed

        if not neighborhood_vehicles and not self._task_is_triggered:
            return ego_wps_des_speed

        wp = self._waypoints.closest_waypoint(pose)
        current_edge = self._road_network.edge_by_lane_id(wp.lane_id)

        if self._task_is_triggered is False:
            self._uturn_initial_heading = pose.heading
            self._uturn_initial_position = pose.position[0]

        vehicle_heading_vec = radians_to_vec(pose.heading)
        initial_heading_vec = radians_to_vec(self._uturn_initial_heading)

        heading_diff = np.dot(vehicle_heading_vec, initial_heading_vec)

        lane = self._road_network.nearest_lane(vehicle.pose.position[:2])
        speed_limit = lane.getSpeed() / 1.5

        vehicle_dist = np.linalg.norm(
            vehicle.pose.position[:2] - neighborhood_vehicles[0].pose.position[:2]
        )
        if vehicle_dist < 5.5:
            speed_limit = 1.5 * lane.getSpeed()

        if (
            heading_diff < -0.95
            and pose.position[0] - self._uturn_initial_position < -2
        ):
            # Once it faces the opposite direction and pass the initial
            # uturn point for 2 meters, stop generating u-turn waypoints
            if (
                pose.position[0] - neighborhood_vehicles[0].pose.position[0] > 12
                or neighborhood_vehicles[0].pose.position[0] > pose.position[0]
            ):
                return ego_wps_des_speed
            else:
                speed_limit = neighborhood_vehicles[0].speed

        self._task_is_triggered = True

        target_lane_index = self._mission.task.target_lane_index
        target_lane_index = min(target_lane_index, len(oncoming_lanes) - 1)
        target_lane = oncoming_lanes[target_lane_index]

        offset = self._road_network.offset_into_lane(start_lane, pose.position[:2])
        oncoming_offset = max(0, target_lane.getLength() - offset)
        paths = self.paths_of_lane_at(target_lane, oncoming_offset, lookahead=30)

        target = paths[0][-1]

        heading = pose.heading
        target_heading = target.heading
        lane_width = target_lane.getWidth()
        lanes = (len(current_edge.getLanes())) + (
            len(oncoming_lanes) - target_lane_index
        )

        p0 = pose.position[:2]
        offset = radians_to_vec(heading) * lane_width
        p1 = np.array(
            [
                pose.position[0] + offset[0],
                pose.position[1] + offset[1],
            ]
        )
        offset = radians_to_vec(target_heading) * 5

        p3 = target.pos
        p2 = np.array([p3[0] - 5 * offset[0], p3[1] - 5 * offset[1]])

        p_x, p_y = bezier([p0, p1, p2, p3], 10)

        trajectory = []
        for i in range(len(p_x)):
            pos = np.array([p_x[i], p_y[i]])
            heading = Heading(vec_to_radians(target.pos - pos))
            lane = self._road_network.nearest_lane(pos)
            lane_id = lane.getID()
            lane_index = lane_id.split("_")[-1]
            width = lane.getWidth()

            wp = Waypoint(
                pos=pos,
                heading=heading,
                lane_width=width,
                speed_limit=speed_limit,
                lane_id=lane_id,
                lane_index=lane_index,
            )
            trajectory.append(wp)

        if self._first_uturn:
            uturn_activated_distance = math.sqrt(
                horizontal_distant ** 2 + vertical_distant ** 2
            )
            self._first_uturn = False

        return [trajectory]

    def cut_in_waypoints(self, sim, pose: Pose, vehicle):
        aggressiveness = self.aggressiveness or 0

        neighborhood_vehicles = sim.neighborhood_vehicles_around_vehicle(
            vehicle=vehicle, radius=850
        )

        position = pose.position[:2]
        lane = self._road_network.nearest_lane(position)

        if not neighborhood_vehicles or sim.elapsed_sim_time < 1:
            return []

        target_vehicle = neighborhood_vehicles[0]
        target_position = target_vehicle.pose.position[:2]

        if (self._prev_kyber_x_position is None) and (
            self._prev_kyber_y_position is None
        ):
            self._prev_kyber_x_position = target_position[0]
            self._prev_kyber_y_position = target_position[1]

        velocity_vector = np.array(
            [
                (-self._prev_kyber_x_position + target_position[0]) / sim.timestep_sec,
                (-self._prev_kyber_y_position + target_position[1]) / sim.timestep_sec,
            ]
        )
        target_velocity = np.dot(
            velocity_vector, radians_to_vec(target_vehicle.pose.heading)
        )

        self._prev_kyber_x_position = target_position[0]
        self._prev_kyber_y_position = target_position[1]

        target_lane = self._road_network.nearest_lane(target_position)

        offset = self._road_network.offset_into_lane(lane, position)
        target_offset = self._road_network.offset_into_lane(
            target_lane, target_position
        )

        # cut-in offset should consider the aggressiveness and the speed
        # of the other vehicle.

        cut_in_offset = np.clip(20 - aggressiveness, 10, 20)

        if (
            abs(offset - (cut_in_offset + target_offset)) > 1
            and lane.getID() != target_lane.getID()
            and self._task_is_triggered is False
        ):
            nei_wps = self._waypoints.waypoint_paths_on_lane_at(
                position, lane.getID(), 60
            )
            speed_limit = np.clip(
                np.clip(
                    (target_velocity * 1.1)
                    - 6 * (offset - (cut_in_offset + target_offset)),
                    0.5 * target_velocity,
                    2 * target_velocity,
                ),
                0.5,
                30,
            )
        else:
            self._task_is_triggered = True
            nei_wps = self._waypoints.waypoint_paths_on_lane_at(
                position, target_lane.getID(), 60
            )

            cut_in_speed = target_velocity * 2.3

            speed_limit = cut_in_speed

            # 1.5 m/s is the threshold for speed offset. If the vehicle speed
            # is less than target_velocity plus this offset then it will not
            # perform the cut-in task and instead the speed of the vehicle is
            # increased.
            if vehicle.speed < target_velocity + 1.5:
                nei_wps = self._waypoints.waypoint_paths_on_lane_at(
                    position, lane.getID(), 60
                )
                speed_limit = np.clip(target_velocity * 2.1, 0.5, 30)
                self._task_is_triggered = False

        p0 = position
        p_temp = nei_wps[0][len(nei_wps[0]) // 3].pos
        p1 = p_temp
        p2 = nei_wps[0][2 * len(nei_wps[0]) // 3].pos

        p3 = nei_wps[0][-1].pos
        p_x, p_y = bezier([p0, p1, p2, p3], 20)
        trajectory = []
        prev = position[:2]
        for i in range(len(p_x)):
            pos = np.array([p_x[i], p_y[i]])
            heading = Heading(vec_to_radians(pos - prev))
            prev = pos
            lane = self._road_network.nearest_lane(pos)
            if lane is None:
                continue
            lane_id = lane.getID()
            lane_index = lane_id.split("_")[-1]
            width = lane.getWidth()

            wp = Waypoint(
                pos=pos,
                heading=heading,
                lane_width=width,
                speed_limit=speed_limit,
                lane_id=lane_id,
                lane_index=lane_index,
            )
            trajectory.append(wp)
        return [trajectory]