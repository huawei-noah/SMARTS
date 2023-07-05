// Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

import {
  Color3,
  Color4,
  MeshBuilder,
  StandardMaterial,
  Vector3,
  Space,
} from "@babylonjs/core";

import { useEffect, useRef } from "react";
import { SignalColors } from "../helpers/scene_colors";

export default function TrafficSignals({ scene, worldState }) {
  if (scene == null || worldState.signals == null) {
    return null;
  }

  const signalGeometryRef = useRef([]);
  const signalColorMap = Object.freeze({
    0: SignalColors.Unknown,
    1: SignalColors.Stop,
    2: SignalColors.Caution,
    3: SignalColors.Go,
  });

  useEffect(() => {
    for (const geom of signalGeometryRef.current) {
      // doNotRecurse = false, disposeMaterialAndTextures = true
      geom.dispose(false, true);
    }

    let newSignalGeometry = Object.keys(worldState.signals).map(
      (signalName) => {
        let state = worldState.signals[signalName].state;
        let pos = worldState.signals[signalName].position;
        let point = new Vector3(pos[0], 0.01, pos[1]);
        let mesh = MeshBuilder.CreateDisc(
          `signal-${signalName}`,
          { radius: 0.8 },
          scene,
        );
        mesh.position = point;
        let axis = new Vector3(1, 0, 0);
        mesh.rotate(axis, Math.PI / 2, Space.WORLD);

        let color = signalColorMap[state];
        let material = new StandardMaterial(
          `signal-${signalName}-material`,
          scene,
        );
        material.diffuseColor = new Color4(...color);
        material.specularColor = new Color3(0, 0, 0);
        mesh.material = material;
        mesh.isVisible = true;
        return mesh;
      },
    );

    signalGeometryRef.current = newSignalGeometry;
  }, [scene, worldState.signals]);
  return null;
}
