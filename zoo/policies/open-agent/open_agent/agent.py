import json
import logging
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import casadi.casadi as cs
import numpy as np
import opengen as og

from smarts.core.agent import Agent
from smarts.core.controllers.trajectory_tracking_controller import (
    TrajectoryTrackingController,
    TrajectoryTrackingControllerState,
)
from smarts.core.coordinates import Heading

from .version import SOLVER_VERSION, VERSION

CONFIG_PATH = Path(__file__).parent / "config.json"


def angle_error(a, b):

    return cs.fabs(
        cs.if_else(
            a - b >= 0,
            cs.fmin((a - b), (-a + b + math.pi * 2.0)),
            cs.fmin(-a + b, (a - b + math.pi * 2.0)),
        )
    )


@dataclass
class Gain:
    theta: cs.SXElem
    position: cs.SXElem
    obstacle: cs.SXElem
    u_accel: cs.SXElem
    u_yaw_rate: cs.SXElem
    terminal: cs.SXElem
    impatience: cs.SXElem
    speed: cs.SXElem
    rate: cs.SXElem
    DOF = 9

    def __iter__(self):
        return iter(
            [
                self.theta,
                self.position,
                self.obstacle,
                self.u_accel,
                self.u_yaw_rate,
                self.terminal,
                self.impatience,
                self.speed,
                self.rate,
            ]
        )

    def setup_debug(self, plt):
        from matplotlib.widgets import Slider

        gains = [
            ("theta", self.theta, 0, 5e5),
            ("position", self.position, 0, 5e5),
            ("obstacle", self.obstacle, 0, 5e5),
            ("u_accel", self.u_accel, 0, 5e5),
            ("u_yaw_rate", self.u_yaw_rate, 0, 5e5),
            ("terminal", self.terminal, 0, 5e5),
            ("impatience", self.impatience, 0, 5e5),
            ("speed", self.speed, 0, 5e5),
            ("rate", self.rate, 0, 5e5),
        ]
        self.debug_sliders = {}
        for i, (gain_name, gain_value, min_v, max_v) in enumerate(reversed(gains)):
            gain_axes = plt.axes([0.25, 0.03 * i, 0.65, 0.03])
            gain_slider = Slider(gain_axes, gain_name, min_v, max_v, valinit=gain_value)
            self.debug_sliders[gain_name] = gain_slider
            gain_slider.on_changed(self.update_debug)

    def update_debug(self, val=None):
        for gain_name, slider in self.debug_sliders.items():
            if gain_name == "theta":
                self.theta = slider.val
            elif gain_name == "position":
                self.position = slider.val
            elif gain_name == "obstacle":
                self.obstacle = slider.val
            elif gain_name == "u_accel":
                self.u_accel = slider.val
            elif gain_name == "u_yaw_rate":
                self.u_yaw_rate = slider.val
            elif gain_name == "terminal":
                self.terminal = slider.val
            elif gain_name == "impatience":
                self.impatience = slider.val
            elif gain_name == "speed":
                self.speed = slider.val
            elif gain_name == "rate":
                self.rate = slider.val

    def persist(self, file_path):
        gains = {
            "theta": self.theta,
            "position": self.position,
            "obstacle": self.obstacle,
            "u_accel": self.u_accel,
            "u_yaw_rate": self.u_yaw_rate,
            "terminal": self.terminal,
            "impatience": self.impatience,
            "speed": self.speed,
            "rate": self.rate,
        }

        with open(file_path, "w") as fp:
            json.dump(gains, fp)

    @classmethod
    def load(cls, file_path):
        with open(file_path, "r") as fp:
            gains = json.load(fp)

        for gain_name, val in gains.items():
            if gain_name == "theta":
                theta = val
            elif gain_name == "position":
                position = val
            elif gain_name == "obstacle":
                obstacle = val
            elif gain_name == "u_accel":
                u_accel = val
            elif gain_name == "u_yaw_rate":
                u_yaw_rate = val
            elif gain_name == "terminal":
                terminal = val
            elif gain_name == "impatience":
                impatience = val
            elif gain_name == "speed":
                speed = val
            elif gain_name == "rate":
                rate = val

        return Gain(
            theta=theta,
            position=position,
            obstacle=obstacle,
            u_accel=u_accel,
            u_yaw_rate=u_yaw_rate,
            terminal=terminal,
            impatience=impatience,
            speed=speed,
            rate=rate,
        )


@dataclass
class Number:
    value: cs.SXElem
    DOF = 1


@dataclass
class VehicleModel:
    """
    Based on the vehicle model defined here:
    http://planning.cs.uiuc.edu/node658.html
    """

    x: cs.SXElem
    y: cs.SXElem
    theta: cs.SXElem
    speed: cs.SXElem
    LENGTH = 3.0
    MAX_SPEED = 14.0  # m/s roughly 50km/h
    MAX_ACCEL = 4.0  # m/s/s
    DOF = 4

    @property
    def as_xref(self):
        return XRef(x=self.x, y=self.y, theta=self.theta)

    def step(self, u: "U", ts):
        self.x += ts * self.speed * cs.cos(self.theta)
        self.y += ts * self.speed * cs.sin(self.theta)
        self.theta += ts * self.speed / self.LENGTH * u.yaw_rate
        self.speed += ts * self.MAX_ACCEL * u.accel


@dataclass
class XRef:
    x: cs.SXElem
    y: cs.SXElem
    theta: cs.SXElem
    DOF = 3

    def weighted_distance_to(self, other: "XRef", gain: Gain):
        theta_err = angle_error(self.theta, other.theta)
        pos_errx = (other.x - self.x) ** 2
        pos_erry = (other.y - self.y) ** 2
        heading_vector = [cs.cos(self.theta), cs.sin(self.theta)]

        lateral_error = (self.x - other.x) ** 2 + (self.y - other.y) ** 2
        pos_err_theta = gain.position * lateral_error ** 1
        return (
            gain.position * pos_errx,
            gain.position * pos_erry,
            gain.theta * theta_err,
            pos_err_theta,
        )


def min_cost_by_distance(xrefs: Sequence[XRef], point: XRef, gain: Gain):
    x_ref_iter = iter(xrefs)
    distant_to_first = next(x_ref_iter).weighted_distance_to(point, gain)
    min_xref_t_cost = sum(distant_to_first[:2])
    # This calculates the weighted combination of lateral error and
    # heading error, TODO: Define new variable or integrates the coefficents
    # into the default values.
    weighted_cost = 10 * distant_to_first[3] + 10 * cs.fabs(distant_to_first[2])
    for xref_t in x_ref_iter:

        distant_to_point = sum(xref_t.weighted_distance_to(point, gain)[:2])

        min_xref_t_cost = cs.if_else(
            distant_to_point <= min_xref_t_cost,
            sum(xref_t.weighted_distance_to(point, gain)[:2]),
            min_xref_t_cost,
        )
        weighted_cost = cs.if_else(
            distant_to_point <= min_xref_t_cost,
            10 * xref_t.weighted_distance_to(point, gain)[3]
            + 10 * cs.fabs(xref_t.weighted_distance_to(point, gain)[2]),
            weighted_cost,
        )

    return 5 * weighted_cost


@dataclass
class U:
    accel: cs.SXElem
    yaw_rate: cs.SXElem


class UTrajectory:
    def __init__(self, N):
        self.N = N
        self.u = cs.SX.sym("u", 2 * N)

    @property
    def symbolic(self):
        return self.u

    def __getitem__(self, i):
        assert 0 <= i < self.N
        return U(accel=self.u[i * 2], yaw_rate=self.u[i * 2 + 1])

    def integration_cost(self, gain: Gain):
        cost = 0
        for t in range(1, self.N):
            prev_u_t = self[t - 1]
            u_t = self[t]
            cost += 0.1 * gain.u_accel * u_t.accel ** 2
            cost += 0.1 * gain.u_yaw_rate * u_t.yaw_rate ** 2
            cost += 0.5 * gain.rate * (u_t.yaw_rate - prev_u_t.yaw_rate) ** 2

        return cost


def build_problem(N, SV_N, WP_N, ts):
    # Assumptions
    assert N >= 2, f"Must generate at least 2 trajectory points, got: {N}"
    assert SV_N >= 0, f"Must have non-negative # of sv's, got: {SV_N}"
    assert WP_N >= 1, f"Must have at lest 1 trajectory reference"

    z0_schema = [
        (1, Gain),
        (1, VehicleModel),  # Ego
        (SV_N, VehicleModel),  # SV's
        (WP_N, XRef),  # reference trajectory
        (1, Number),  # impatience
        (1, Number),  # target_speed
    ]

    z0_dimensions = sum(n * feature.DOF for n, feature in z0_schema)
    z0 = cs.SX.sym("z0", z0_dimensions)
    u_traj = UTrajectory(N)

    # parse z0 into features
    position = 0
    parsed = []
    for n, feature in z0_schema:
        feature_group = []
        for i in range(n):
            feature_group.append(
                feature(*z0[position : position + feature.DOF].elements())
            )
            position += feature.DOF
        if n > 1:
            parsed.append(feature_group)
        else:
            assert len(feature_group) == 1
            parsed += feature_group

    assert position == len(z0.elements())
    assert position == z0_dimensions

    gain, ego, social_vehicles, xref_traj, impatience, target_speed = parsed

    cost = 0

    for t in range(N):
        # Integrate the ego vehicle forward to the next trajectory point
        ego.step(u_traj[t], ts)

        # For the current pose, compute the smallest cost to any xref
        cost += min_cost_by_distance(xref_traj, ego.as_xref, gain)

        cost += gain.speed * (ego.speed - target_speed.value) ** 2 / t

        for sv in social_vehicles:
            # step the social vehicle assuming no change in velocity or heading
            sv.step(U(accel=0, yaw_rate=0), ts)

            min_dist = VehicleModel.LENGTH
            cost += gain.obstacle * cs.fmax(
                -1,
                min_dist ** 2 - ((ego.x - sv.x) ** 2 + 9 * (ego.y - sv.y) ** 2),
            )

    # To stabilize the trajectory, we attach a higher weight to the final x_ref
    cost += gain.terminal * sum(
        xref_traj[-1].weighted_distance_to(ego.as_xref, gain)[:3]
    )

    cost += u_traj.integration_cost(gain)

    # force acceleration when we become increasingly impatient
    cost += gain.impatience * ((u_traj[0].accel - 1.0) * impatience.value ** 2 * -(1.0))
    # cost=0

    bounds = og.constraints.Rectangle(
        xmin=[-1, -math.pi * 0.1] * N, xmax=[1, math.pi * 0.1] * N
    )

    return og.builder.Problem(u_traj.symbolic, z0, cost).with_constraints(bounds)


def compile_solver(output_dir, N=6, SV_N=4, WP_N=15, ts=0.1):
    build_dir = Path(output_dir)
    solver_name = "open_agent_solver"
    path_to_solver = build_dir / solver_name
    problem = build_problem(N, SV_N, WP_N, ts)
    build_config = (
        og.config.BuildConfiguration()
        .with_build_directory(build_dir)
        .with_build_mode("release")
        .with_build_python_bindings()
    )
    meta = (
        og.config.OptimizerMeta()
        .with_version(SOLVER_VERSION)
        .with_optimizer_name(solver_name)
    )
    solver_config = (
        og.config.SolverConfiguration()
        .with_tolerance(1e-15)
        .with_max_inner_iterations(555)
    )
    builder = og.builder.OpEnOptimizerBuilder(
        problem, meta, build_config, solver_config
    ).with_verbosity_level(3)
    builder.build()

    potentially_built_libs = [
        path_to_solver / f"{solver_name}.{ext}" for ext in ["so", "pyd"]
    ]
    built_libs = [lib.absolute() for lib in potentially_built_libs if lib.exists()]
    for built_lib in built_libs:
        shutil.copyfile(built_lib, Path(__file__).parent / built_lib.name)

    persist_config({"version": VERSION, "N": N, "SV_N": SV_N, "WP_N": WP_N, "ts": ts})

    return built_libs


def persist_config(config):
    with open(CONFIG_PATH, "w") as config_fp:
        json.dump(config, config_fp)


def load_config():
    with open(CONFIG_PATH, "r") as config_fp:
        return json.load(config_fp)


class OpEnAgent(Agent):
    def __init__(
        self,
        gains={
            "theta": 3.0 * 50,
            "position": 1.0 * 100,
            "obstacle": 3.0,
            "u_accel": 0.01,
            "u_yaw_rate": 0.1,
            "terminal": 0.01,
            "impatience": 0.01,
            "speed": 0.5 * 100,
            "rate": 0 * 10,
        },
        debug=False,
    ):
        self.log = logging.getLogger(self.__class__.__name__)
        self.debug = debug
        self.last_position = None
        self.steps_without_moving = 0
        config = load_config()

        assert (
            config["version"] == VERSION
        ), f'Extension not compiled for current version {config["version"]} != {VERSION}'

        self.N = config["N"]
        self.SV_N = config["SV_N"]
        self.WP_N = config["WP_N"]
        self.ts = config["ts"]

        self.gain_save_path = Path("gains.json")
        if self.gain_save_path.exists():
            print(f"Loading gains from {self.gain_save_path.absolute()}")
            self.gain = Gain.load(self.gain_save_path)
        else:
            self.gain = Gain(**gains)

        if self.debug:
            self._setup_debug_pannel()

        self.init_planner()

    def init_planner(self):
        try:
            import open_agent_solver
        except ImportError:
            raise "Can't import the solver, have you compiled it?"

        self.prev_solution = None
        self.solver = open_agent_solver.solver()

    def update_impatience(self, ego_state):
        if (
            self.last_position is not None
            and np.linalg.norm(ego_state.position - self.last_position) < 1e-1
        ):
            self.steps_without_moving += 1
        else:
            self.steps_without_moving = 0

        return self.steps_without_moving

    def act(self, obs):
        ego = obs.ego_vehicle_state
        lane_index = min(len(obs.waypoint_paths) - 1, ego.lane_index)
        wps = obs.waypoint_paths[lane_index]
        for wp_path in obs.waypoint_paths:
            if wp_path[0].lane_index == lane_index:
                wps = wp_path
                break
        # wps = min(
        #     obs.waypoint_paths,
        #     key=lambda wps: abs(wps[0].signed_lateral_error(ego.position)),
        # )
        num_trajectory_points = len(wps)
        trajectory = [
            [wps[i].pos[0] for i in range(num_trajectory_points)],
            [wps[i].pos[1] for i in range(num_trajectory_points)],
            [wps[i].heading for i in range(num_trajectory_points)],
        ]
        curvature = abs(
            TrajectoryTrackingController.curvature_calculation(
                trajectory, num_points=10
            )
        )

        gains = {
            "theta": 3.0 * 0.05,
            "position": 1.0 * 100,
            "obstacle": 0.0,
            "u_accel": 0.01,
            "u_yaw_rate": 0.1,
            "terminal": 0.01,
            "impatience": 0.0,
            "speed": 0.5 * 100,
            "rate": 1 * 10,
        }
        self.gain = Gain(**gains)

        if curvature < 100:
            gains = {
                "theta": 16 * 0.05,
                "position": 1 * 120,
                "obstacle": 3.0,
                "u_accel": 0.01,
                "u_yaw_rate": 0.1,
                "terminal": 0.0001,
                "impatience": 0.01,
                "speed": 0.5 * 100,
                "rate": 0.1 * 1,
            }
            self.gain = Gain(**gains)

        if curvature < 10:
            gains = {
                "theta": 1 * 5,
                "position": 40 * 6,
                "obstacle": 3.0,
                "u_accel": 0.1 * 1,
                "u_yaw_rate": 0.01,
                "terminal": 0.01,
                "impatience": 0.01,
                "speed": 0.3 * 400,
                "rate": 0.001,
            }
            self.gain = Gain(**gains)
        if curvature > 200:
            gains = {
                "theta": 0.1 * 0.05,
                "position": 1 * 10,
                "obstacle": 3.0,
                "u_accel": 0.1,
                "u_yaw_rate": 0.1,
                "terminal": 0.01,
                "impatience": 0.01,
                "speed": 0.5 * 250 * 2,
                "rate": 10,
            }
            self.gain = Gain(**gains)
        wps = wps[: self.WP_N]

        # repeat the last waypoint to fill out self.WP_N waypoints
        wps += [wps[-1]] * (self.WP_N - len(wps))
        flat_wps = [
            wp_param
            for wp in wps
            for wp_param in [wp.pos[0], wp.pos[1], float(wp.heading) + math.pi * 0.5]
        ]
        if self.SV_N == 0:
            flat_svs = []
        elif len(obs.neighborhood_vehicle_states) == 0 and self.SV_N > 0:
            # We have no social vehicles in the scene, create placeholders far away
            flat_svs = [
                ego.position[0] + 100000,
                ego.position[1] + 100000,
                0,
                0,
            ] * self.SV_N
        else:
            # Give the closest SV_N social vehicles to the planner
            social_vehicles = sorted(
                obs.neighborhood_vehicle_states,
                key=lambda sv: np.linalg.norm(sv.position - ego.position),
            )[: self.SV_N]

            # repeat the last social vehicle to ensure SV_N social vehicles
            social_vehicles += [social_vehicles[-1]] * (
                self.SV_N - len(social_vehicles)
            )
            flat_svs = [
                sv_param
                for sv in social_vehicles
                for sv_param in [
                    sv.position[0],
                    sv.position[1],
                    float(sv.heading) + math.pi * 0.5,
                    sv.speed,
                ]
            ]

        ego_params = [
            ego.position[0],
            ego.position[1],
            float(ego.heading) + math.pi * 0.5,
            ego.speed,
        ]

        impatience = self.update_impatience(ego)
        solver_params = (
            list(self.gain)
            + ego_params
            + flat_svs
            + flat_wps
            + [impatience, wps[0].speed_limit]
        )
        neutral_bias = [0 for i in range(12)]
        resp = self.solver.run(solver_params, initial_guess=neutral_bias)

        self.last_position = ego.position

        if resp is not None:
            u_star = resp.solution
            self.prev_solution = u_star
            ego_model = VehicleModel(*ego_params)
            xs = []
            ys = []
            headings = []
            speeds = []
            for u in zip(u_star[::2], u_star[1::2]):
                ego_model.step(U(*u), self.ts)
                headings.append(Heading(ego_model.theta - math.pi * 0.5))
                xs.append(ego_model.x)
                ys.append(ego_model.y)
                speeds.append(ego_model.speed)

            traj = [xs, ys, headings, speeds]
            if self.debug:
                self._draw_debug_panel(xs, ys, wps, flat_svs, ego, u_star)

            return traj
        else:
            # Failed to find a solution.
            # re-init the planner and stay still, hopefully once we've re-initialized, we can recover
            self.init_planner()
            return None

    def _setup_debug_pannel(self):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.plt.close()  # close any open plts from previous episodes
        self.gain.setup_debug(plt)
        self.plt.ion()

    def _draw_debug_panel(self, xs, ys, wps, flat_svs, ego, u_star):
        self.gain.persist(self.gain_save_path)

        subplot = self.plt.subplot(221)
        subplot.clear()

        self.plt.plot(xs, ys, "o-", color="xkcd:crimson", label="trajectory")
        wp_x = [wp.pos[0] for wp in wps]
        wp_y = [wp.pos[1] for wp in wps]
        self.plt.scatter(wp_x, wp_y, color="red", label="waypoint")

        sv_x = [sv_x for sv_x in flat_svs[:: VehicleModel.DOF]]
        sv_y = [sv_y for sv_y in flat_svs[1 :: VehicleModel.DOF]]
        self.plt.scatter(sv_x, sv_y, label="social vehicles")
        plt_radius = 50
        self.plt.axis(
            (
                ego.position[0] - plt_radius,
                ego.position[0] + plt_radius,
                ego.position[1] - plt_radius,
                ego.position[1] + plt_radius,
            )
        )
        self.plt.legend()

        subplot = self.plt.subplot(222)
        subplot.clear()
        u_accels = u_star[::2]
        u_thetas = u_star[1::2]
        ts = range(len(u_accels))
        self.plt.plot(ts, u_accels, "o-", color="gold", label="u_accel")
        self.plt.plot(ts, u_thetas, "o-", color="purple", label="u_theta")
        self.plt.legend()

        self.plt.draw()
        self.plt.pause(1e-6)
