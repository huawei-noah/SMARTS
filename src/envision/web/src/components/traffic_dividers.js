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
import { Vector3, Color4, MeshBuilder } from "@babylonjs/core";

import { useRef, useEffect } from "react";

export default function TrafficDividers({
  scene,
  worldState,
  laneDividerPos,
  edgeDividerPos,
}) {
  if (scene == null) {
    return null;
  }

  const laneDividerGeometryRef = useRef([]);
  const edgeDividerGeometryRef = useRef(null);

  // Lane dividers
  useEffect(() => {
    if (laneDividerPos.length == 0) {
      return;
    }

    for (const geom of laneDividerGeometryRef.current) {
      geom.dispose();
    }

    let newLaneDividers = laneDividerPos.map((lines, idx) => {
      let points = lines.map((p) => new Vector3(p[0], 0.1, p[1]));
      let dashLine = MeshBuilder.CreateDashedLines(
        `lane-divider-${idx}`,
        { points: points, updatable: false, dashSize: 1, gapSize: 2 },
        scene
      );
      dashLine.color = new Color4(...worldState.scene_colors["lane_divider"]);
      return dashLine;
    });

    laneDividerGeometryRef.current = newLaneDividers;
  }, [scene, JSON.stringify(laneDividerPos)]);

  // Edge dividers
  useEffect(() => {
    if (edgeDividerPos.length == 0) {
      return;
    }

    if (edgeDividerGeometryRef.current != null) {
      edgeDividerGeometryRef.current.dispose();
    }

    let edgeDividerPoints = edgeDividerPos.map((lines) => {
      let points = lines.map((p) => new Vector3(p[0], 0.1, p[1]));
      return points;
    });

    let newEdgeDividers = MeshBuilder.CreateLineSystem(
      "edge-dividers",
      { lines: edgeDividerPoints, updatable: false },
      scene
    );
    newEdgeDividers.color = new Color4(
      ...worldState.scene_colors["edge_divider"]
    );

    edgeDividerGeometryRef.current = newEdgeDividers;
  }, [scene, JSON.stringify(edgeDividerPos)]);

  return null;
}
