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
  Vector3,
  Color3,
  StandardMaterial,
  Color4,
  MeshBuilder,
} from "@babylonjs/core";

import { useRef, useEffect } from "react";

// Mission route geometry
export default function MissionRoutes({ scene, worldState }) {
  if (scene == null) {
    return null;
  }

  let missionRoutes = {};
  let missionRouteStringify = [];

  let sortedTrafficKeys = Object.keys(worldState.traffic).sort();
  for (const vehicle_id of sortedTrafficKeys) {
    let actor = worldState.traffic[vehicle_id];
    if (actor.mission_route_geometry == null) {
      continue;
    }

    missionRoutes[vehicle_id] = actor.mission_route_geometry;
    missionRouteStringify.push(actor.mission_route_geometry);
  }

  const missionGeometryRef = useRef([]);
  let missionGeometry = missionGeometryRef.current;

  useEffect(() => {
    for (const geom of missionGeometryRef.current) {
      // doNotRecurse = false, disposeMaterialAndTextures = true
      geom.dispose(false, true);
    }

    missionGeometryRef.current = [];
    for (const [vehicle_id, agent_route] of Object.entries(missionRoutes)) {
      agent_route.forEach((route_shape, shape_id) => {
        let points = route_shape.map((p) => new Vector3(p[0], 0, p[1]));
        let polygon = MeshBuilder.CreatePolygon(
          `mission-route-shape-${vehicle_id}-${shape_id}`,
          { shape: points },
          scene
        );
        polygon.position.y = 0.1;
        polygon.material = new StandardMaterial(
          `mission-route-shape-${vehicle_id}-${shape_id}-material`,
          scene
        );
        polygon.material.diffuseColor = new Color4(
          ...worldState.scene_colors["mission_route"]
        );
        polygon.material.alpha = worldState.scene_colors["mission_route"][3];
        polygon.material.specularColor = new Color3(0, 0, 0);
        missionGeometryRef.current.push(polygon);
      });
    }
  }, [scene, JSON.stringify(missionRouteStringify)]);

  return null;
}
