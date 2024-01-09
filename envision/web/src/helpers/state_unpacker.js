// MIT License
//
// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

function multi_slice(array, width) {
  let a = [];
  for (let i = 0; i < array.length / width; i++) {
    let b = i * width;
    a[i] = array.slice(b, b + width);
  }
  return a;
}

const WorldState = Object.freeze({
  SCENARIO_ID: 1,
  SCENARIO_NAME: 2,
  TRAFFIC: 3,
  TRAFFIC_SIGNALS: 4,
  BUBBLES: 5,
  SCORES: 6,
  EGO_AGENT_IDS: 7,
});

const Traffic = Object.freeze({
  ACTOR_ID: 0,
  LANE_ID: 1,
  POSITION_BEGIN: 2,
  POSITION_END: 5,
  HEADING: 5,
  SPEED: 6,
  EVENTS: 7,
  WAYPOINT_PATHS: 8,
  DRIVEN_PATH: 9,
  POINT_CLOUD: 10,
  MISSION_ROUTE_GEOMETRY: 11,
  ACTOR_TYPE: 12,
  VEHICLE_TYPE: 13,
  INTEREST: 14,
});

const TrafficSignal = Object.freeze({
  SIGNAL_ID: 0,
  STATE: 1,
  POSITION_BEGIN: 2,
  POSITION_END: 5,
});

const POINT_2D_LENGTH = 2;
const POINT_3D_LENGTH = 3;

const BUBBLE_POINT_LENGTH = POINT_2D_LENGTH;
const DRIVEN_PATH_POINT_LENGTH = POINT_2D_LENGTH;
const LIDAR_POINT_LENGTH = POINT_3D_LENGTH;
const ROUTE_POINT_LENGTH = POINT_2D_LENGTH;

const Waypoint = Object.freeze({
  POSITION_BEGIN: 0,
  POSITION_END: 3,
  HEADING: 3,
  LANE_ID: 4,
  LANE_WIDTH: 5,
  SPEED_LIMIT: 6,
  LANE_INDEX: 7,
});

const AGENT_TYPE_MAP = Object.freeze({
  0: "social_vehicle",
  1: "social_agent",
  2: "agent",
});

const VEHICLE_TYPE_MAP = Object.freeze({
  0: "bus",
  1: "coach",
  2: "truck",
  3: "trailer",
  4: "car",
  5: "motorcycle",
  6: "pedestrian",
});

function unpack_bubbles(bubbles) {
  return bubbles.map((a) => multi_slice(a, BUBBLE_POINT_LENGTH));
}

function unpack_driven_path(driven_path) {
  return multi_slice(driven_path, DRIVEN_PATH_POINT_LENGTH);
}

function unpack_waypoints(lanes) {
  return lanes.map((lane) =>
    lane.map(function (wp) {
      return {
        pos: wp.slice(Waypoint.POSITION_BEGIN, Waypoint.POSITION_END),
        heading: wp[Waypoint.HEADING],
        lane_id: wp[Waypoint.LANE_ID],
        lane_width: wp[Waypoint.LANE_WIDTH],
        speed_limit: wp[Waypoint.SPEED_LIMIT],
        lane_index: wp[Waypoint.LANE_INDEX],
      };
    }),
  );
}

function unpack_point_cloud(point_cloud) {
  return multi_slice(point_cloud, LIDAR_POINT_LENGTH);
}

function unpack_route_geometry(route_geometry) {
  return route_geometry.map((a) => multi_slice(a, ROUTE_POINT_LENGTH));
}

function unpack_traffic(traffic) {
  let mapped_traffic = Object.assign(
    {},
    ...traffic.map((t) => ({
      [t[Traffic.ACTOR_ID]]: {
        actor_id: t[Traffic.ACTOR_ID],
        lane_id: t[Traffic.LANE_ID],
        position: t.slice(Traffic.POSITION_BEGIN, Traffic.POSITION_END),
        heading: t[Traffic.HEADING],
        speed: t[Traffic.SPEED],
        events: t[Traffic.EVENTS],
        waypoint_paths: unpack_waypoints(t[Traffic.WAYPOINT_PATHS]),
        driven_path: unpack_driven_path(t[Traffic.DRIVEN_PATH]),
        point_cloud: unpack_point_cloud(t[Traffic.POINT_CLOUD]),
        mission_route_geometry: unpack_route_geometry(
          t[Traffic.MISSION_ROUTE_GEOMETRY],
        ),
        actor_type: AGENT_TYPE_MAP[t[Traffic.ACTOR_TYPE]],
        vehicle_type: VEHICLE_TYPE_MAP[t[Traffic.VEHICLE_TYPE]],
        interest: t[Traffic.INTEREST],
      },
    })),
  );
  return mapped_traffic;
}

function unpack_signals(signals) {
  let mapped_signals = Object.assign(
    {},
    ...signals.map((t) => ({
      [t[TrafficSignal.SIGNAL_ID]]: {
        state: t[TrafficSignal.STATE],
        position: t.slice(
          TrafficSignal.POSITION_BEGIN,
          TrafficSignal.POSITION_END,
        ),
      },
    })),
  );
  return mapped_signals;
}

function get_attribute_map(unpacked_traffic, attr) {
  return Object.fromEntries(
    Object.entries(unpacked_traffic)
      .filter(([_, t]) => t.actor_type === AGENT_TYPE_MAP[2])
      .map(([n, t]) => [n, t[attr]]),
  );
}

export default function unpack_worldstate(formatted_state) {
  let unpacked_bubbles = unpack_bubbles(formatted_state[WorldState.BUBBLES]);
  let unpacked_traffic = unpack_traffic(formatted_state[WorldState.TRAFFIC]);
  let unpacked_signals = unpack_signals(
    formatted_state[WorldState.TRAFFIC_SIGNALS],
  );
  const worldstate = {
    traffic: unpacked_traffic,
    signals: unpacked_signals,
    scenario_id: formatted_state[WorldState.SCENARIO_ID],
    scenario_name: formatted_state[WorldState.SCENARIO_NAME],
    bubbles: unpacked_bubbles,
    scores: Object.fromEntries(formatted_state[WorldState.SCORES]),
    ego_agent_ids: formatted_state[WorldState.EGO_AGENT_IDS],
    position: get_attribute_map(unpacked_traffic, "position"),
    speed: get_attribute_map(unpacked_traffic, "speed"),
    heading: get_attribute_map(unpacked_traffic, "heading"),
    lane_ids: get_attribute_map(unpacked_traffic, "lane_id"),
  };
  return worldstate;
}
