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
import {
  Vector2,
  Vector3,
  Color3,
  StandardMaterial,
  Color4,
} from "@babylonjs/core";

import { useRef, useEffect } from "react";

import { ActorTypes } from "../enums.js";
import { vehicleMeshColor } from "../render_helpers.js";

// Driven path geometry
export default function DrivenPaths({
  scene,
  worldState,
  egoDrivenPathModel,
  socialDrivenPathModel,
}) {
  if (scene == null) {
    return null;
  }

  const drivenPathGeometriesRef = useRef({});
  let drivenPathGeometries = drivenPathGeometriesRef.current;

  useEffect(() => {
    if (worldState.traffic.length == 0) {
      return;
    }

    let newDrivenPathGeometries = {};

    // No need to re-create all driven path segments every frame,
    // cache geometries with unchanged positions across frames
    for (const vehicle_id of Object.keys(drivenPathGeometries)) {
      let drivenPath = drivenPathGeometries[vehicle_id];
      var i = 0;
      for (i = 0; i < drivenPath.length; i++) {
        if (
          !(vehicle_id in worldState.traffic) ||
          worldState.traffic[vehicle_id].driven_path.length < 2
        ) {
          drivenPath[i].dispose();
        } else {
          let geomPos = new Vector2(
            drivenPath[i].position.x,
            drivenPath[i].position.z
          );
          let newGeomPos = Vector2.Center(
            new Vector2(...worldState.traffic[vehicle_id].driven_path[0]),
            new Vector2(...worldState.traffic[vehicle_id].driven_path[1])
          );
          if (geomPos.equalsWithEpsilon(newGeomPos, 0.0001)) {
            newDrivenPathGeometries[vehicle_id] = [];
            break;
          }
          drivenPath[i].dispose();
        }
      }

      for (let j = i; j < drivenPath.length; j++) {
        newDrivenPathGeometries[vehicle_id].push(drivenPath[j]);
      }
    }

    if (egoDrivenPathModel.material == null) {
      egoDrivenPathModel.material = new StandardMaterial(
        "ego-driven-path-material",
        scene
      );
      egoDrivenPathModel.material.specularColor = new Color3(0, 0, 0);
      egoDrivenPathModel.material.diffuseColor = new Color4(
        ...worldState.scene_colors["ego_driven_path"]
      );
      egoDrivenPathModel.material.alpha =
        worldState.scene_colors["ego_driven_path"][3];
    }

    if (socialDrivenPathModel.material == null) {
      socialDrivenPathModel.material = new StandardMaterial(
        "social-driven-path-material",
        scene
      );
      socialDrivenPathModel.material.specularColor = new Color3(0, 0, 0);
      let color = vehicleMeshColor(
        ActorTypes.SOCIAL_AGENT,
        worldState.scene_colors
      );
      socialDrivenPathModel.material.diffuseColor = new Color4(...color);
      socialDrivenPathModel.material.alpha =
        worldState.scene_colors["ego_driven_path"][3];
    }

    // Add in new driven path segments
    let drivenPathOffsetY = 0.1;
    for (const [vehicle_id, trafficActor] of Object.entries(
      worldState.traffic
    )) {
      if (!(vehicle_id in newDrivenPathGeometries)) {
        newDrivenPathGeometries[vehicle_id] = [];
      }

      for (
        let i = newDrivenPathGeometries[vehicle_id].length;
        i < trafficActor.driven_path.length - 1;
        i++
      ) {
        let drivenPathSegment_ = null;
        if (trafficActor.actor_type == ActorTypes.SOCIAL_AGENT) {
          drivenPathSegment_ = socialDrivenPathModel.createInstance(
            "social-driven-path-segment"
          );
        } else {
          drivenPathSegment_ = egoDrivenPathModel.createInstance(
            "ego-driven-path-segment"
          );
        }

        let p0 = new Vector3(
          trafficActor.driven_path[i][0],
          drivenPathOffsetY,
          trafficActor.driven_path[i][1]
        );
        let p1 = new Vector3(
          trafficActor.driven_path[i + 1][0],
          drivenPathOffsetY,
          trafficActor.driven_path[i + 1][1]
        );

        drivenPathSegment_.position = Vector3.Center(p0, p1);

        let axis1 = p0.subtract(p1);
        let axis3 = new Vector3(0, 1, 0);
        let axis2 = Vector3.Cross(axis3, axis1);

        drivenPathSegment_.scaling.x = axis1.length() + 0.01;
        drivenPathSegment_.rotation = Vector3.RotationFromAxis(
          axis1,
          axis2,
          axis3
        );

        newDrivenPathGeometries[vehicle_id].push(drivenPathSegment_);
      }
    }

    drivenPathGeometriesRef.current = newDrivenPathGeometries;
  }, [worldState.traffic]);

  return null;
}
