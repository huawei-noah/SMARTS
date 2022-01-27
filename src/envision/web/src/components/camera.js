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
  ArcRotateCamera,
  Vector3,
  UniversalCamera,
  TransformNode,
} from "@babylonjs/core";

import { useRef, useEffect } from "react";

export default function Camera({ scene, roadNetworkBbox, egoView }) {
  if (scene == null) {
    return null;
  }

  const egoCamRootRef = useRef();
  const thirdPersonCameraRef = useRef();

  let canvas = scene.getEngine().getRenderingCanvas();

  if (!egoCamRootRef.current) {
    let egoCamRoot = new TransformNode("ego-camera-root");
    egoCamRoot.position = new Vector3.Zero(); // Set to the ego vehicle position during update
    let egoCamera = new UniversalCamera(
      "ego-camera",
      new Vector3(0, 5, -15), // Relative to camera root position
      scene
    );
    egoCamera.parent = egoCamRoot;
    egoCamRootRef.current = egoCamRoot;
  }

  if (!thirdPersonCameraRef.current) {
    let thirdPersonCamera = new ArcRotateCamera(
      "third-person-camera",
      -Math.PI / 2, // alpha
      0, // beta
      200, // radius
      new Vector3(0, 0, 0), // target
      scene
    );
    thirdPersonCamera.attachControl(canvas, true);
    thirdPersonCamera.panningSensibility = 50;
    thirdPersonCamera.lowerRadiusLimit = 5;
    scene.activeCamera = thirdPersonCamera; // default camera
    thirdPersonCameraRef.current = thirdPersonCamera;
  }

  // Update third person camera's pointing target and radius
  useEffect(() => {
    if (roadNetworkBbox.length != 4) {
      return;
    }

    let mapCenter = [
      (roadNetworkBbox[0] + roadNetworkBbox[2]) / 2,
      (roadNetworkBbox[1] + roadNetworkBbox[3]) / 2,
    ];
    let thirdPersonCamera = thirdPersonCameraRef.current;
    thirdPersonCamera.target.x = mapCenter[0];
    thirdPersonCamera.target.z = mapCenter[1];
    thirdPersonCamera.radius = Math.max(
      Math.abs(roadNetworkBbox[0] - roadNetworkBbox[2]),
      Math.abs(roadNetworkBbox[1] - roadNetworkBbox[3])
    );
  }, [JSON.stringify(roadNetworkBbox)]);

  // Set active camera
  useEffect(() => {
    let egoCamRoot = egoCamRootRef.current;
    let thirdPersonCamera = thirdPersonCameraRef.current;

    if (egoView) {
      let egoCamera = egoCamRoot.getChildren()[0];
      scene.activeCamera = egoCamera;

      // Disable mouse input to the third person camera during ego view
      thirdPersonCamera.detachControl(canvas);
    } else {
      thirdPersonCamera.attachControl(canvas, true);
      scene.activeCamera = thirdPersonCamera;
    }
  }, [egoView]);

  return null;
}
