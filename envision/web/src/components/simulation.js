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
import React, { useState, useEffect, useRef } from "react";

import {
  Vector3,
  Color3,
  Tools,
  SceneLoader,
  StandardMaterial,
  Quaternion,
  HemisphericLight,
  MeshBuilder,
  Color4,
} from "@babylonjs/core";

import { GLTFLoader } from "@babylonjs/loaders/glTF/2.0/glTFLoader";
import SceneComponent from "babylonjs-hook";

import Bubbles from "./bubbles.js";
import Camera from "./camera.js";
import Vehicles from "./vehicles.js";
import DrivenPaths from "./driven_paths.js";
import MissionRoutes from "./mission_routes.js";
import Waypoints from "./waypoints.js";
import TrafficDividers from "./traffic_dividers.js";
import { attrs, agentModes } from "./control_panel";

import InfoDisplay from "./InfoDisplay";
import earcut from "earcut";

// Required by Babylon.js
window.earcut = earcut;

export default function Simulation({
  simulationId,
  client,
  egoView,
  controlModes,
  canvasRef = null,
  onElapsedTimesChanged = (current, total) => {},
  style = {},
  playing = true,
}) {
  const [scene, setScene] = useState(null);

  const [egoWaypointModel, setEgoWaypointModel] = useState(null);
  const [socialWaypointModel, setSocialWaypointModel] = useState(null);
  const [egoDrivenPathModel, setEgoDrivenPathModel] = useState(null);
  const [socialDrivenPathModel, setSocialDrivenPathModel] = useState(null);

  const [roadNetworkBbox, setRoadNetworkBbox] = useState([]);
  const [laneDividerPos, setLaneDividerPos] = useState([]);
  const [edgeDividerPos, setEdgeDividerPos] = useState([]);

  const [worldState, setWorldState] = useState({
    traffic: [],
    scenario_id: null,
    bubbles: [],
    scene_colors: {},
    scores: [],
    ego_agent_ids: [],
    position: [],
    speed: [],
    heading: [],
    lane_ids: [],
  });

  const mapMeshesRef = useRef([]);

  // Parse extra data attached in glb file
  function LoadGLTFExtras(loader, scenario_id) {
    // Register loader locally under different names for different scenarios
    this.name = `load_gltf_extras_${scenario_id}`;
    this.enabled = true;

    if (loader.gltf["extras"]) {
      if (loader.gltf.extras["lane_dividers"]) {
        setLaneDividerPos(loader.gltf.extras["lane_dividers"]);
      }
      if (loader.gltf.extras["edge_dividers"]) {
        setEdgeDividerPos(loader.gltf.extras["edge_dividers"]);
      }
      if (loader.gltf.extras["bounding_box"]) {
        setRoadNetworkBbox(loader.gltf.extras["bounding_box"]);
      }
    }
  }

  const onSceneReady = (scene_) => {
    let canvas = scene_.getEngine().getRenderingCanvas();
    if (canvasRef) {
      canvasRef.current = canvas;
    }

    // Waypoint cylinder
    let cylinder_ = MeshBuilder.CreateCylinder(
      "waypoint",
      { diameterTop: 0.5, diameterBottom: 0.5, height: 0.01 },
      scene_
    );
    cylinder_.isVisible = false;

    setEgoWaypointModel(cylinder_.clone("ego-waypoint").makeGeometryUnique());
    setSocialWaypointModel(
      cylinder_.clone("social-waypoint").makeGeometryUnique()
    );

    // Driven path cuboid
    let cuboid_ = MeshBuilder.CreateBox(
      "drivenPath",
      { height: 0.3, width: 1, depth: 0.01 },
      scene_
    );
    cuboid_.isVisible = false;

    setEgoDrivenPathModel(
      cuboid_.clone("ego-driven-path").makeGeometryUnique()
    );
    setSocialDrivenPathModel(
      cuboid_.clone("social-driven-path").makeGeometryUnique()
    );

    // Light
    let light = new HemisphericLight("light", new Vector3(0, 1, 0), scene_);
    light.intensity = 1;

    // Scene
    scene_.ambientColor = new Color3(1, 1, 1);
    scene_.clearColor = new Color3(8 / 255, 25 / 255, 10 / 255);

    setScene(scene_);
  };

  // State subscription
  useEffect(() => {
    let stopPolling = false;
    (async () => {
      const it = client.worldstate(simulationId);
      let wstate_and_time = await it.next();
      while (!wstate_and_time.done && playing) {
        let wstate, elapsed_times;
        [wstate, elapsed_times] = wstate_and_time.value;
        if (!stopPolling) {
          setWorldState(wstate);
          onElapsedTimesChanged(...elapsed_times);
        }
        wstate_and_time = await it.next();
      }
    })();

    // Called when simulation ID changes
    return () => (stopPolling = true);
  }, [simulationId, playing]);

  // Load map
  useEffect(() => {
    if (scene == null || worldState.scenario_id == null) {
      return;
    }

    for (const mesh of mapMeshesRef.current) {
      // doNotRecurse = false, disposeMaterialAndTextures = true
      mesh.dispose(false, true);
    }

    let mapSourceUrl = `${client.endpoint.origin}/assets/maps/${worldState.scenario_id}.glb`;
    let mapRootUrl = Tools.GetFolderPath(mapSourceUrl);
    let mapFilename = Tools.GetFilename(mapSourceUrl);

    // Load extra information attached to the map glb
    GLTFLoader.RegisterExtension(
      `load_gltf_extras_${worldState.scenario_id}`,
      function (loader) {
        return new LoadGLTFExtras(loader, worldState.scenario_id);
      }
    );

    SceneLoader.ImportMesh("", mapRootUrl, mapFilename, scene, (meshes) => {
      // Revert root mesh's rotation to match Babylon's coordinate system
      let axis = new Vector3(0, 0, 1);
      let angle = 0;
      let quaternion = new Quaternion.RotationAxis(axis, angle);
      meshes[0].rotationQuaternion = quaternion; // root mesh

      // Update material for all child meshes
      // Currently only use flat shading, replace imported pbr material with standard material
      for (const child of meshes[0].getChildMeshes()) {
        let material = new StandardMaterial("material-map", scene);
        material.backFaceCulling = false;
        material.diffuseColor = new Color4(...worldState.scene_colors["road"]);
        material.specularColor = new Color3(0, 0, 0);
        child.material = material;
      }

      mapMeshesRef.current = meshes;

      GLTFLoader.UnregisterExtension(
        `load_gltf_extras_${worldState.scenario_id}`
      );
    });
  }, [scene, worldState.scenario_id]);

  return (
    <div
      style={{ position: "relative", width: "100%", height: "100%", ...style }}
    >
      <SceneComponent
        antialias
        onSceneReady={onSceneReady}
        style={{
          zIndex: "0",
          position: "absolute",
          top: "0",
          left: "0",
          bottom: "0",
          right: "0",
          width: "100%",
          height: "100%",
        }}
      />
      <Camera
        scene={scene}
        roadNetworkBbox={roadNetworkBbox}
        egoView={egoView}
      />
      <Vehicles
        scene={scene}
        worldState={worldState}
        vehicleRootUrl={`${client.endpoint.origin}/assets/models/`}
        egoView={egoView}
      />
      <Bubbles scene={scene} worldState={worldState} />
      <DrivenPaths
        scene={scene}
        worldState={worldState}
        egoDrivenPathModel={egoDrivenPathModel}
        socialDrivenPathModel={socialDrivenPathModel}
      />
      <MissionRoutes scene={scene} worldState={worldState} />
      <Waypoints
        scene={scene}
        worldState={worldState}
        egoWaypointModel={egoWaypointModel}
        socialWaypointModel={socialWaypointModel}
      />
      <TrafficDividers
        scene={scene}
        worldState={worldState}
        laneDividerPos={laneDividerPos}
        edgeDividerPos={edgeDividerPos}
      />
      <div
        style={{
          zIndex: "1",
          position: "absolute",
          top: "0",
          left: "0",
          maxWidth: "100%",
        }}
      >
        {controlModes[attrs.score] ? (
          <InfoDisplay
            data={worldState.scores}
            attrName="Score"
            data_formattter={(score) => parseFloat(score).toFixed(2)}
            ego_agent_ids={worldState.ego_agent_ids}
            ego_only={!controlModes[agentModes.socialObs]}
          />
        ) : null}
        {controlModes[attrs.speed] ? (
          <InfoDisplay
            data={worldState.speed}
            attrName="Speed"
            data_formattter={(speed) => parseFloat(speed).toFixed(2)}
            ego_agent_ids={worldState.ego_agent_ids}
            ego_only={!controlModes[agentModes.socialObs]}
          />
        ) : null}
        {controlModes[attrs.position] ? (
          <InfoDisplay
            data={worldState.position}
            attrName="Position"
            data_formattter={(position) =>
              `x: ${parseFloat(position[0]).toFixed(2)} y: ${parseFloat(
                position[1]
              ).toFixed(2)}`
            }
            ego_agent_ids={worldState.ego_agent_ids}
            ego_only={!controlModes[agentModes.socialObs]}
          />
        ) : null}
        {controlModes[attrs.heading] ? (
          <InfoDisplay
            data={worldState.heading}
            attrName="Heading"
            data_formattter={(heading) => parseFloat(heading).toFixed(2)}
            ego_agent_ids={worldState.ego_agent_ids}
            ego_only={!controlModes[agentModes.socialObs]}
          />
        ) : null}
        {controlModes[attrs.laneID] ? (
          <InfoDisplay
            data={worldState.lane_ids}
            attrName="Lane ID"
            data_formattter={(lane_id) => lane_id}
            ego_agent_ids={worldState.ego_agent_ids}
            ego_only={!controlModes[agentModes.socialObs]}
          />
        ) : null}
      </div>
    </div>
  );
}
