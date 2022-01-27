// Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
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

import { Color3, Color4, StandardMaterial } from "@babylonjs/core";

import { useEffect, useRef } from "react";

import { ActorTypes } from "../enums.js";
import { vehicleMeshColor } from "../render_helpers.js";

export default function Waypoints({
  scene,
  worldState,
  egoWaypointModel,
  socialWaypointModel,
}) {
  if (scene == null) {
    return null;
  }

  const waypointGeometriesRef = useRef([]);
  let waypointGeometries = waypointGeometriesRef.current;

  useEffect(() => {
    for (const geom of waypointGeometries) {
      geom.dispose();
    }

    if (worldState.traffic.length == 0) {
      return;
    }

    if (egoWaypointModel.material == null) {
      egoWaypointModel.material = new StandardMaterial(
        "ego-waypoint-material",
        scene
      );
      egoWaypointModel.material.specularColor = new Color3(0, 0, 0);
      egoWaypointModel.material.diffuseColor = new Color4(
        ...worldState.scene_colors["ego_waypoint"]
      );
      egoWaypointModel.material.alpha =
        worldState.scene_colors["ego_waypoint"][3];
    }

    if (socialWaypointModel.material == null) {
      socialWaypointModel.material = new StandardMaterial(
        "social-waypoint-material",
        scene
      );
      socialWaypointModel.material.specularColor = new Color3(0, 0, 0);
      let color = vehicleMeshColor(
        ActorTypes.SOCIAL_AGENT,
        worldState.scene_colors
      );
      socialWaypointModel.material.diffuseColor = new Color4(...color);
      socialWaypointModel.material.alpha =
        worldState.scene_colors["ego_waypoint"][3];
    }

    let newWaypointGeometries = [];
    for (const [_, trafficActor] of Object.entries(worldState.traffic)) {
      for (const waypointPath of trafficActor.waypoint_paths) {
        for (const waypoint of waypointPath) {
          let wp_ = null;
          if (trafficActor.actor_type == ActorTypes.SOCIAL_AGENT) {
            wp_ = socialWaypointModel.createInstance("social-wp");
          } else {
            wp_ = egoWaypointModel.createInstance("ego-wp");
          }
          wp_.position.x = waypoint.pos[0];
          wp_.position.y = 0.15;
          wp_.position.z = waypoint.pos[1];
          newWaypointGeometries.push(wp_);
        }
      }
    }
    waypointGeometriesRef.current = newWaypointGeometries;
  }, [scene, worldState.traffic]);

  return null;
}
