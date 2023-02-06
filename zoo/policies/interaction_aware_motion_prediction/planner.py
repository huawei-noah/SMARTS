import matplotlib.pyplot as plt
import numpy as np
import torch

from smarts.core.road_map import Waypoint
from smarts.core.utils.math import (
    _gen_ego_frame_matrix,
    constrain_angle,
    radians_to_vec,
    signed_dist_to_line,
)

from .planner_utils import *


class Planner(object):
    def __init__(self, predictor=None, render=False):
        self.horizon = 3  # s
        self.dt = 0.1  # s
        self.cost_weights = np.array([1, 0.5, 0.1, 0.1, 1, 5, 10])
        self.render_obs = render
        self.predictor = predictor
        self.device = next(predictor.parameters()).device

    def generate_routes(self, paths):
        cs_routes = []
        if hasattr(self.ego_state.mission.goal, "position"):
            goal = np.array(self.ego_state.mission.goal.position)
        else:
            goal = self.ego_state.position
        self.routes = []
        self.speed_limit = paths[0][0].speed_limit

        for path in paths:
            # path extrapolation if not enough waypoints
            while len(path) < 51:
                ref_wp = path[-1]
                path.append(
                    Waypoint(
                        pos=np.array(
                            [
                                ref_wp.pos[0] + np.cos(ref_wp.heading + np.pi / 2),
                                ref_wp.pos[1] + np.sin(ref_wp.heading + np.pi / 2),
                            ]
                        ),
                        heading=ref_wp.heading,
                        lane_id=ref_wp.lane_id,
                        lane_width=ref_wp.lane_width,
                        speed_limit=ref_wp.speed_limit,
                        lane_index=ref_wp.lane_index,
                        lane_offset=ref_wp.lane_offset + 1,
                    )
                )

            # generate route
            path_x = [waypoint.pos[0] for waypoint in path]
            path_y = [waypoint.pos[1] for waypoint in path]
            route_x, route_y, route_dir, _, route_cs = generate_target_course(
                path_x, path_y
            )
            route = np.column_stack([route_x, route_y, route_dir])
            self.routes.append(route)
            cs_routes.append(route_cs)

        self.target_route = np.min(
            [
                np.min(np.linalg.norm(route[:, :2] - goal[None, :2], axis=-1))
                for route in self.routes
            ]
        )

        return cs_routes

    def generate_trajectories(self, routes):
        trajectories = {}  # key: high-level decision, value: trajectory

        for lane, route in enumerate(routes):
            for acc in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]:
                frenet_trajectory = self.generate_trajectory_on_frenet(
                    acc, self.routes[lane]
                )
                trajectory = self.calculate_global_trajectory(frenet_trajectory, route)
                trajectories[(lane, acc)] = trajectory

        self.emergency_traj = self.generate_emergency_trajectory(
            self.routes[lane], route
        )

        return trajectories

    def generate_emergency_trajectory(self, route, cs_route):
        fp = FrenetPath()
        fp.t = np.arange(self.dt, self.horizon + self.dt, self.dt)
        current_v = self.ego_frame_dynamics(
            self.ego_state.linear_velocity[:2], route[0][-1]
        )
        current_d = signed_dist_to_line(
            self.ego_state.position[:2],
            route[0, :2],
            radians_to_vec(route[0][-1] - math.pi / 2),
        )

        # max decelerate
        fp.s_d = [current_v[0] - 15 * t for t in fp.t]
        fp.s = np.cumsum(np.clip(fp.s_d, 0.01, 16) * 0.1)
        fp.d = [current_d for t in fp.t]
        fp.d_d = [current_v[1] for t in fp.t]

        # to global pos
        traj = self.calculate_global_trajectory(fp, cs_route)

        return traj

    def generate_trajectory_on_frenet(self, acc, route):
        fp = FrenetPath()
        fp.t = np.arange(self.dt, self.horizon + self.dt, self.dt)
        current_v = self.ego_frame_dynamics(
            self.ego_state.linear_velocity[:2], route[0][-1]
        )
        current_a = self.ego_frame_dynamics(
            self.ego_state.linear_acceleration[:2], route[0][-1]
        )

        # longitudinal
        current_vs = current_v[0]
        current_as = current_a[0]
        s_d, s = generate_lon_profile(current_vs, current_as, acc)
        fp.s_d, fp.s = s_d, s

        # lateral
        current_d = signed_dist_to_line(
            self.ego_state.position[:2],
            route[0, :2],
            radians_to_vec(route[0][-1] - math.pi / 2),
        )
        current_vd = current_v[1]
        d_d, d = generate_lat_profile(current_d, current_vd)
        fp.d, fp.d_d = d, d_d

        return fp

    def calculate_global_trajectory(self, fp, csp):
        # calc global positions
        for i in range(len(fp.s)):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                break
            i_yaw = csp.calc_yaw(fp.s[i])
            di = fp.d[i]
            fx = ix + di * np.cos(i_yaw + math.pi / 2.0)
            fy = iy + di * np.sin(i_yaw + math.pi / 2.0)
            fp.x.append(fx)
            fp.y.append(fy)

        # calc yaw and ds
        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(np.arctan2(dy, dx))
            fp.ds.append(np.hypot(dx, dy))

        fp.yaw.append(fp.yaw[-1])
        fp.ds.append(fp.ds[-1])

        return fp

    def get_other_agent_reaction(self, env_input):
        # query the model
        env_input = {
            key: torch.as_tensor(_obs).unsqueeze(0).float().to(self.device)
            for (key, _obs) in env_input.items()
        }

        with torch.no_grad():
            predictions, scores = self.predictor(env_input)
            best_mode = torch.argmax(scores, dim=-1)[0]
            N = predictions.shape[1]
            prediction = predictions[0][torch.arange(N)[:, None], best_mode[:, None]]
            prediction = prediction.cpu().numpy()

        # get valid agents
        valid_prediction = []
        for i in range(prediction.shape[0]):
            if env_input["neighbors_state"][0, i, -1].sum(-1) != 0:
                valid_prediction.append(prediction[i])

        return valid_prediction

    def calculate_cost(self, plan, prediction, route):
        d2g = self.check_dist_to_goal(route)
        speed = self.check_speed(plan) / self.speed_limit
        lon_jerk, lat_acc = self.check_comfort(plan)
        collision, ttc, d2a = self.check_collision(plan, prediction, route)
        cost = np.sum(
            self.cost_weights
            * np.stack([d2g, speed, lon_jerk, lat_acc, d2a, ttc, collision])
        )

        return cost, collision

    def plan(self, data):
        obs = data[0]
        env_input = data[1]

        # generate trajectories
        self.ego_state = obs.ego_vehicle_state
        routes = self.generate_routes(obs.waypoint_paths)
        trajectories = self.generate_trajectories(routes)

        # evaluate trajectories
        scores = {}
        response = {}
        prediction_ego_frame = self.get_other_agent_reaction(env_input)
        collision = {}

        for d, t in trajectories.items():
            prediction = self.project_to_world_position(
                prediction_ego_frame,
                self.ego_state.position,
                self.ego_state.heading + np.pi / 2,
            )
            response[d] = prediction
            cost, coll = self.calculate_cost(t, prediction, self.routes[d[0]])
            scores[d] = -cost
            collision[d] = True if coll == 1 else False

            if self.render_obs:
                self.render(t, response[d])

        self.response = response
        self.trajectories = trajectories

        if all(value for value in collision.values()):
            target_traj = self.emergency_traj
        else:
            scores = sorted(scores, key=scores.get, reverse=True)
            best = scores[0]
            target_traj = trajectories[best]

        # output target pose
        target_pose = [
            (
                target_traj.x[t],  # target global x
                target_traj.y[t],  # target global y
                constrain_angle(target_traj.yaw[t] - math.pi / 2),  # target heading
                0.1,
            )
            for t in range(len(target_traj.x))
        ]

        return target_pose

    def check_speed(self, traj):
        lon_speed = np.array(traj.s_d)
        speed_diff = np.abs(lon_speed - self.speed_limit)
        speed_diff = np.mean(speed_diff)

        return speed_diff

    def check_dist_to_goal(self, route):
        if hasattr(self.ego_state.mission.goal, "position"):
            goal = np.array(self.ego_state.mission.goal.position)
        else:
            goal = self.ego_state.position
        route_to_goal = np.linalg.norm(route[:, :2] - goal[None, :2], axis=-1)
        dist_to_goal = np.min(route_to_goal) - self.target_route

        return dist_to_goal

    def check_collision(self, plan, prediction, route):
        plan_frenet = np.column_stack([plan.s, plan.d])
        plan = np.column_stack([plan.x, plan.y])
        collision = 0
        ttc = 3
        dist = [100]

        # iterate through every timestep
        for t in range(plan.shape[0]):
            if collision == 1:
                break

            # iterate through every agent's every future
            for a in prediction:
                for f in range(len(a)):
                    if np.linalg.norm(plan[t][:2] - a[f][t]) > 5:
                        continue

                    agent_frenet = self.project_to_frenet(a[f][t], route)
                    if not agent_frenet:
                        continue

                    delta_s = np.abs(plan_frenet[t, 0] - agent_frenet[0])
                    delta_d = np.abs(plan_frenet[t, 1] - agent_frenet[1])

                    if delta_s < 5 and delta_d < 2.5:
                        collision = 1
                        ttc = (t + 1) / 10

                    if delta_d < 2.5:
                        dist.append(np.linalg.norm(plan[t] - a[f][t]))

        min_dist = np.clip(np.min(dist) - 3, 0, 20)
        d2a = np.exp(-0.3 * min_dist**2)
        ttc = 1 - ttc / 3

        return collision, ttc, d2a

    def check_comfort(self, traj):
        traj = np.column_stack([traj.x, traj.y, traj.yaw])
        v_x, v_y, theta = (
            np.diff(traj[:, 0]) / 0.1,
            np.diff(traj[:, 1]) / 0.1,
            traj[1:, 2],
        )
        lon_speed = v_x * np.cos(theta) + v_y * np.sin(theta)
        lat_speed = v_y * np.cos(theta) - v_x * np.sin(theta)
        acc = np.diff(lon_speed, n=1) / 0.1
        jerk = np.diff(acc, n=1) / 0.1
        lat_acc = np.diff(lat_speed) / 0.1
        jerk = np.mean(np.abs(jerk)) / 10
        lat_acc = np.mean(np.abs(lat_acc)) / 10

        return jerk, lat_acc

    @staticmethod
    def project_to_frenet(cartesian_pos, path):
        distance_to_ref = np.linalg.norm(cartesian_pos[None, :] - path[:, :2], axis=-1)
        k = np.argmin(distance_to_ref)

        if k == 0 or k == path.shape[0] - 1:
            return None

        s = 0.1 * k
        r = path[k]

        dx = cartesian_pos[0] - r[0]
        dy = cartesian_pos[1] - r[1]
        cross_rd_nd = np.cos(r[2]) * dy - np.sin(r[2]) * dx
        d = np.sign(cross_rd_nd) * np.sqrt(dx**2 + dy**2)

        return (s, d)

    @staticmethod
    def project_to_world_position(prediction, ego_pos, ego_heading):
        world_prediction = []

        transform_matrix = _gen_ego_frame_matrix(ego_heading)
        transform_matrix = np.linalg.inv(transform_matrix)

        for a in prediction:
            futures = []
            for f in a:
                f = np.concatenate([f[:, :2], np.zeros(shape=(f.shape[0], 1))], axis=-1)
                rot_traj = np.matmul(transform_matrix, f.T).T
                new_traj = rot_traj + np.asarray(ego_pos)
                futures.append(new_traj[:, :2])
            world_prediction.append(futures)

        return world_prediction

    @staticmethod
    def ego_frame_dynamics(v, theta):
        ego_v = v.copy()
        ego_v[0] = v[0] * np.cos(theta) + v[1] * np.sin(theta)
        ego_v[1] = v[1] * np.cos(theta) - v[0] * np.sin(theta)

        return ego_v

    @staticmethod
    def render(trajectory, prediction):
        for i, traj in enumerate([trajectory]):
            plt.plot(traj.x, traj.y, zorder=30 - i)

        for a in prediction:
            for f in range(len(a)):
                plt.plot(a[f][:, 0], a[f][:, 1], "k", zorder=30)

        plt.gca().set_aspect("equal")
        plt.show(block=False)
        plt.pause(0.5)
        plt.clf()
        # plt.show()
