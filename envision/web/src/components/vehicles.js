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
// FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
import {
  Vector3,
  Color3,
  Color4,
  SceneLoader,
  StandardMaterial,
  Quaternion,
  MeshBuilder,
  Mesh,
  BoundingInfo,
  ActionManager,
  ExecuteCodeAction,
} from "@babylonjs/core";

import { useRef, useEffect } from "react";

import { ActorTypes } from "../enums.js";
import { SceneColors } from "../helpers/scene_colors.js";
import { intersection, difference } from "../math.js";
import {
  vehicleMeshFilename,
  vehicleMeshColor,
  buildLabel,
} from "../render_helpers.js";

const DEBUG_PROPS = [
  "actor_id",
  "position",
  "speed",
  "heading",
  "actor_type",
  "vehicle_type",
];

let meshesLoaded = (vehicleMeshTemplates) => {
  for (const [_, mesh] of Object.entries(vehicleMeshTemplates)) {
    if (mesh == null) {
      return false;
    }
  }
  return true;
};

// Vehicles and agent labels
export default function Vehicles({
  scene,
  worldState,
  vehicleRootUrl,
  egoView,
  setVehicleSelected,
  setDebugInfo,
}) {
  if (scene == null) {
    return null;
  }

  const vehicleMeshTemplatesRef = useRef({});
  const vehicleMeshesRef = useRef({});
  const agentLabelGeometryRef = useRef({});

  let vehicleMeshTemplates = vehicleMeshTemplatesRef.current;
  let vehicleMeshes = vehicleMeshesRef.current;
  let agentLabelGeometry = agentLabelGeometryRef.current;

  // Load mesh asynchronously
  useEffect(() => {
    for (const [vehicleFilename, meshTemplate] of Object.entries(
      vehicleMeshTemplates,
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
          let originalRootMesh = meshes[0];
          originalRootMesh.isVisible = false;
          let rootMeshMin = new Vector3();
          let rootMeshMax = new Vector3();
          let childMeshes = originalRootMesh.getChildMeshes();
          for (let i = 0; i < childMeshes.length; i++) {
            let child = childMeshes[i];
            child.isVisible = false;

            let material = new StandardMaterial(
              `material-${vehicleFilename}`,
              scene,
            );
            material.backFaceCulling = false;
            material.disableLighting = true;
            material.emissiveColor = Color3.White();
            // Instance buffers must be used because setting the diffuse color on the instance material
            // changes the color of all instance meshes because they share the same material reference.
            // See: https://doc.babylonjs.com/features/featuresDeepDive/mesh/copies/instances#custom-buffers
            child.registerInstancedBuffer("color", 4);
            if (child.material) {
              // Currently only use flat shading, replace imported pbr material with standard material
              material.id = child.material.id;
              child.instancedBuffers.color = Color4.FromColor3(
                child.material.albedoColor,
                1,
              );
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

          // Note bug: `gltf` sourced root nodes must be replaced, removed, or rescaled in order to instance
          // the mesh. The issue is vaguely described here:
          // https://forum.babylonjs.com/t/register-instanced-buffer-turns-mesh-instance-black/8445
          let newRoot = new Mesh(originalRootMesh.name);
          newRoot.isVisible = false;
          newRoot.metadata = originalRootMesh.metadata;
          for (let i = 0; i < childMeshes.length; i++) {
            let child = childMeshes[i];
            child.parent = newRoot;
          }

          newRoot.setBoundingInfo(new BoundingInfo(rootMeshMin, rootMeshMax));
          vehicleMeshTemplates[vehicleFilename] = newRoot;
        },
      );
    }
    // This useEffect is triggered when the vehicleMeshTemplate's keys() change
  }, [Object.keys(vehicleMeshTemplates).sort().join("-")]);

  useEffect(() => {
    if (!meshesLoaded(vehicleMeshTemplates)) {
      return;
    }

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
      delete vehicleMeshes[meshId]; // remove key from dict

      if (agentLabelGeometry[meshId]) {
        agentLabelGeometry[meshId].dispose();
      }
      delete agentLabelGeometry[meshId]; // remove key from dict
    }

    // Create new meshes
    for (const meshId of vehicleMeshIdsToAdd) {
      let state = worldState.traffic[meshId];

      // Vehicle mesh
      let filename = vehicleMeshFilename(state.actor_type, state.vehicle_type);
      if (!vehicleMeshTemplates[filename]) {
        // Triggers loading the mesh according through the useEffect
        vehicleMeshTemplates[filename] = null;
        continue;
      }

      let color = vehicleMeshColor(state.actor_type);
      let rootMesh = new Mesh(`root-mesh-${meshId}`, scene);
      let childMeshes = vehicleMeshTemplates[filename].getChildMeshes();
      for (const child of childMeshes) {
        let instancedSubMesh = child.createInstance(`${child.name}-${meshId}`);
        instancedSubMesh.alwaysSelectAsActiveMesh = true;
        if (state.interest && instancedSubMesh.material.id == "body") {
          instancedSubMesh.instancedBuffers.color = new Color4(
            ...SceneColors.Interest,
            1,
          );
        } else if (
          state.actor_type == ActorTypes.SOCIAL_VEHICLE ||
          instancedSubMesh.material.id == "body" || // Change the car body color based on actor type
          childMeshes.length == 1
        ) {
          instancedSubMesh.instancedBuffers.color = new Color4(...color, 1);
        }
        rootMesh.addChild(instancedSubMesh);
      }

      let boundingInfo = vehicleMeshTemplates[filename].getBoundingInfo();
      let boxSize = boundingInfo.boundingBox.extendSize.scale(2);
      let box = MeshBuilder.CreateBox(
        `boundingbox-${meshId}`,
        {
          height: boxSize.y + 0.1,
          width: boxSize.x + 0.1,
          depth: boxSize.z + 0.1,
        },
        scene,
      );

      let boxMaterial = new StandardMaterial(
        `boundingbox-${meshId}-material`,
        scene,
      );
      boxMaterial.diffuseColor = new Color4(...SceneColors.Selection);
      boxMaterial.specularColor = new Color3(0, 0, 0);
      boxMaterial.alpha = 0.0;

      box.material = boxMaterial;
      box.position = boundingInfo.boundingBox.center;
      box.actionManager = new ActionManager(scene);
      box.actionManager.registerAction(
        new ExecuteCodeAction(ActionManager.OnPointerOverTrigger, function (
          evt,
        ) {
          boxMaterial.alpha = 0.5;
          setVehicleSelected(true);
          setDebugInfo(evt.meshUnderPointer.parent.metadata.debugInfo);
        }),
      );
      box.actionManager.registerAction(
        new ExecuteCodeAction(ActionManager.OnPointerOutTrigger, function (
          evt,
        ) {
          boxMaterial.alpha = 0.0;
          setVehicleSelected(false);
          setDebugInfo({});
        }),
      );
      rootMesh.addChild(box);

      rootMesh.metadata = {};
      rootMesh.metadata.actorType = state.actor_type;

      vehicleMeshes[meshId] = rootMesh;

      // Agent label
      // Only show labels on agents
      let label = null;
      if (state.actor_type == ActorTypes.AGENT) {
        label = buildLabel(meshId, state.actor_id, scene);
      }
      agentLabelGeometry[meshId] = label;
    }

    let firstEgoAgent = true;
    // Set mesh positions and orientations
    for (const meshId of nextVehicleMeshIds) {
      let state = worldState.traffic[meshId];
      let mesh = vehicleMeshes[meshId];

      // Unavailable until the mesh template has been loaded
      if (!mesh) {
        continue;
      }

      let debugInfo = {};
      for (const prop of DEBUG_PROPS) {
        debugInfo[prop] = state[prop];
      }
      mesh.metadata.debugInfo = debugInfo;

      let pos = state.position;
      mesh.position = new Vector3(pos[0], pos[2], pos[1]);

      let axis = new Vector3(0, 1, 0);
      let angle = Math.PI - state.heading;
      let quaternion = new Quaternion.RotationAxis(axis, angle);
      mesh.rotationQuaternion = quaternion;

      let label = agentLabelGeometry[meshId];
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
  }, [worldState.traffic]);

  return null;
}
