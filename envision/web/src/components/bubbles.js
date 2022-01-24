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
  Color3,
  Color4,
  Mesh,
  MeshBuilder,
  StandardMaterial,
  Vector3,
} from "@babylonjs/core";

import { useEffect, useRef } from "react";

export default function Bubbles({ scene, worldState }) {
  if (scene == null) {
    return null;
  }

  const bubbleGeometryRef = useRef([]);

  useEffect(() => {
    for (const geom of bubbleGeometryRef.current) {
      // doNotRecurse = false, disposeMaterialAndTextures = true
      geom.dispose(false, true);
    }

    let newBubbleGeometry = worldState.bubbles.map((bubbleGeom, idx) => {
      let points = bubbleGeom.map((p) => new Vector3(p[0], 0, p[1]));
      let polygon = MeshBuilder.CreatePolygon(
        `bubble-${idx}`,
        {
          sideOrientation: Mesh.DOUBLESIDE,
          shape: points,
          depth: 5,
        },
        scene
      );
      polygon.position.y = 4;
      let material = new StandardMaterial(`bubble-${idx}-material`, scene);
      material.diffuseColor = new Color4(
        ...worldState.scene_colors["bubble_line"]
      );
      material.specularColor = new Color3(0, 0, 0);
      material.alpha = worldState.scene_colors["bubble_line"][3];
      polygon.material = material;
      return polygon;
    });

    bubbleGeometryRef.current = newBubbleGeometry;
  }, [scene, JSON.stringify(worldState.bubbles)]);
  return null;
}
