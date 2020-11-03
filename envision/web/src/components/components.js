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
  Vector2,
  Vector3,
  Color3,
  Tools,
  SceneLoader,
  StandardMaterial,
  Quaternion,
  MeshBuilder,
  Mesh,
  Color4,
  BoundingInfo,
  UniversalCamera,
  TransformNode,
} from "@babylonjs/core";

import { useRef, useEffect } from "react";

import { ActorTypes } from "../enums.js";
import { intersection, difference } from "../math.js";
import {
  vehicleMeshFilename,
  vehicleMeshColor,
  buildLabel,
} from "../render_helpers.js";

let meshesLoaded = (vehicleMeshTemplates) => {
  for (const [_, mesh] of Object.entries(vehicleMeshTemplates)) {
    if (mesh == null) {
      return false;
    }
  }
  return true;
};

export function Camera({ scene, roadNetworkBbox, egoView }) {
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
    thirdPersonCameraRef.current.target.x = mapCenter[0];
    thirdPersonCameraRef.current.target.z = mapCenter[1];
    thirdPersonCameraRef.current.radius = Math.max(
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
      egoCamera.rotation = new Vector3.Zero(); //TODO: Not needed?
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

// Vehicles and agent labels
export function Vehicles({ scene, worldState, vehicleRootUrl, egoView }) {
  if (scene == null) {
    return null;
  }

  const vehicleMeshesRef = useRef({});
  const agentLabelGeometryRef = useRef({});
  const vehicleMeshTemplatesRef = useRef({});

  // Load mesh asynchronously
  useEffect(() => {
    for (const [vehicleFilename, meshTemplate] of Object.entries(
      vehicleMeshTemplatesRef.current
    )) {
      if (meshTemplate != null) {
        continue;
      }

      SceneLoader.ImportMesh(
        "",
        vehicleRootUrl,
        vehicleFilename,
        scene,
        (meshes) => {
          let rootMesh = meshes[0];
          rootMesh.isVisible = false;
          let rootMeshMin = new Vector3();
          let rootMeshMax = new Vector3();
          let childMeshes = rootMesh.getChildMeshes();
          for (let i = 0; i < childMeshes.length; i++) {
            let child = childMeshes[i];
            child.isVisible = false;

            let material = new StandardMaterial(
              `material-${vehicleFilename}`,
              scene
            );
            material.backFaceCulling = false;
            material.specularColor = new Color3(0, 0, 0);
            if (child.material) {
              // Currently only use flat shading, replace imported pbr material with standard material
              material.id = child.material.id;
              material.diffuseColor = child.material.albedoColor;
            }
            child.material = material;

            let childMin = child.getBoundingInfo().boundingBox.minimumWorld;
            let childMax = child.getBoundingInfo().boundingBox.maximumWorld;

            if (i == 0) {
              rootMeshMin = childMin;
              rootMeshMax = childMax;
            } else {
              rootMeshMin = Vector3.Minimize(rootMeshMin, childMin);
              rootMeshMax = Vector3.Maximize(rootMeshMax, childMax);
            }
          }

          rootMesh.setBoundingInfo(new BoundingInfo(rootMeshMin, rootMeshMax));

          vehicleMeshTemplatesRef.current[vehicleFilename] = rootMesh;
        }
      );
    }
    // This useEffect is triggered when the vehicleMeshTemplate's keys() change
  }, [Object.keys(vehicleMeshTemplatesRef.current).sort().join("-")]);

  useEffect(() => {
    if (!meshesLoaded(vehicleMeshTemplatesRef.current)) {
      return;
    }

    let vehicleMeshes = vehicleMeshesRef.current;
    let agentLabelGeometry = agentLabelGeometryRef.current;

    let nextVehicleMeshes = {};
    let nextAgentLabelGeometry = {};

    let nextVehicleMeshIds = new Set(Object.keys(worldState.traffic));
    let vehicleMeshIds = new Set(Object.keys(vehicleMeshes));

    let vehicleMeshIdsToRemove = difference(vehicleMeshIds, nextVehicleMeshIds);
    let vehicleMeshIdsToKeep = intersection(vehicleMeshIds, nextVehicleMeshIds);
    let vehicleMeshIdsToAdd = difference(nextVehicleMeshIds, vehicleMeshIds);

    // Vehicle model and color need to be updated when its agent actor type changes
    let meshIds = new Set(vehicleMeshIdsToKeep);
    for (const meshId of meshIds) {
      if (
        worldState.traffic[meshId].actor_type !=
        vehicleMeshes[meshId].metadata.actorType
      ) {
        vehicleMeshIdsToRemove.add(meshId);
        vehicleMeshIdsToKeep.delete(meshId);
        vehicleMeshIdsToAdd.add(meshId);
      }
    }

    // Dispose of stale meshes
    for (const meshId of vehicleMeshIdsToRemove) {
      vehicleMeshes[meshId].dispose();

      let label = agentLabelGeometry[meshId];
      if (label) {
        label.dispose();
      }
    }

    // Add back kept meshes
    for (const meshId of vehicleMeshIdsToKeep) {
      nextVehicleMeshes[meshId] = vehicleMeshes[meshId];
      nextAgentLabelGeometry[meshId] = agentLabelGeometry[meshId];
    }

    // Create new meshes
    for (const meshId of vehicleMeshIdsToAdd) {
      let state = worldState.traffic[meshId];
      // Vehicle mesh
      let filename = vehicleMeshFilename(state.actor_type, state.vehicle_type);
      if (!vehicleMeshTemplatesRef.current[filename]) {
        // Triggers loading the mesh according through the useEffect
        vehicleMeshTemplatesRef.current[filename] = null;
        continue;
      }

      let color = vehicleMeshColor(state.actor_type, worldState.scene_colors);
      let rootMesh = new Mesh(`root-mesh-${meshId}`, scene);
      let childMeshes = vehicleMeshTemplatesRef.current[
        filename
      ].getChildMeshes();
      for (const child of childMeshes) {
        let instancedSubMesh = child.createInstance(`${child.name}-${meshId}`);
        if (
          state.actor_type == ActorTypes.SOCIAL_VEHICLE ||
          instancedSubMesh.material.id == "body" || // Change the car body color based on actor type
          childMeshes.length == 1
        ) {
          instancedSubMesh.material.diffuseColor = new Color3(...color);
        }
        rootMesh.addChild(instancedSubMesh);
      }

      // Render bounding box for social vehicle
      if (state.actor_type == ActorTypes.SOCIAL_VEHICLE) {
        let boundingInfo = vehicleMeshTemplatesRef.current[
          filename
        ].getBoundingInfo();
        let boxSize = boundingInfo.boundingBox.extendSize.scale(2);
        let box = MeshBuilder.CreateBox(
          `boundingbox-${meshId}`,
          {
            height: boxSize.y + 0.1,
            width: boxSize.x + 0.1,
            depth: boxSize.z + 0.1,
          },
          scene
        );
        box.position = boundingInfo.boundingBox.center;

        let boxMaterial = new StandardMaterial(
          `boundingbox-${meshId}-material`,
          scene
        );
        boxMaterial.diffuseColor = new Color4(...color);
        boxMaterial.specularColor = new Color3(0, 0, 0);
        boxMaterial.alpha = 0.75;
        box.material = boxMaterial;
        rootMesh.addChild(box);
      }

      rootMesh.metadata = {};
      rootMesh.metadata.actorType = state.actor_type;

      nextVehicleMeshes[meshId] = rootMesh;

      // Agent label
      // Only show labels on agents
      let label = null;
      if (state.actor_type == ActorTypes.AGENT) {
        label = buildLabel(meshId, state.actor_id, scene);
      }
      nextAgentLabelGeometry[meshId] = label;
    }

    let firstEgoAgent = true;
    // Set mesh positions and orientations
    for (const meshId of nextVehicleMeshIds) {
      let state = worldState.traffic[meshId];
      let mesh = nextVehicleMeshes[meshId];
      // Unavailable until the mesh template has been loaded
      if (!mesh) {
        continue;
      }

      let pos = state.position;
      mesh.position = new Vector3(pos[0], pos[2], pos[1]);

      let axis = new Vector3(0, 1, 0);
      let angle = Math.PI - state.heading;
      let quaternion = new Quaternion.RotationAxis(axis, angle);
      mesh.rotationQuaternion = quaternion;

      let label = nextAgentLabelGeometry[meshId];
      if (label) {
        label.position = new Vector3(pos[0], pos[2] + 2, pos[1] - 4);
        label.isVisible = true;
      }

      // Ego camera follows the first ego agent in multi-agent case
      if (egoView && state.actor_type == ActorTypes.AGENT && firstEgoAgent) {
        label.isVisible = false;
        firstEgoAgent = false;
        let egoCamRoot = scene.activeCamera.parent;
        egoCamRoot.position = new Vector3(pos[0], pos[2], pos[1]);
        egoCamRoot.rotation = new Vector3(0, 2 * Math.PI - state.heading, 0);
      }
    }

    vehicleMeshesRef.current = nextVehicleMeshes;
    agentLabelGeometryRef.current = nextAgentLabelGeometry;
  }, [worldState.traffic]);

  return null;
}
