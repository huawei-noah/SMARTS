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
import React, { useState, useEffect } from "react";

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

import AgentScores from "./agent_scores";
import earcut from "earcut";

// Required by Babylon.js
window.earcut = earcut;

export default function Simulation({
  simulationId,
  client,
  showScores,
  egoView,
  canvasRef,
  onElapsedTimesChanged = (current, total) => {},
  style = {},
}) {
  const [scene, setScene] = useState(null);

  const [egoWaypointModel, setEgoWaypointModel] = useState(null);
  const [socialWaypointModel, setSocialWaypointModel] = useState(null);
  const [egoDrivenPathModel, setEgoDrivenPathModel] = useState(null);
  const [socialDrivenPathModel, setSocialDrivenPathModel] = useState(null);

  const [mapMeshes, setMapMeshes] = useState([]);

  const [roadNetworkBbox, setRoadNetworkBbox] = useState([]);
  const [laneDividerPos, setLaneDividerPos] = useState([]);
  const [edgeDividerPos, setEdgeDividerPos] = useState([]);
  const [laneDividerGeometry, setLaneDividerGeometry] = useState([]);
  const [edgeDividerGeometry, setEdgeDividerGeometry] = useState(null);

  const [worldState, setWorldState] = useState({
    traffic: [],
    scenario_id: null,
    bubbles: [],
    scene_colors: {},
    scores: [],
  });

  // Parse extra data attached in glb file
  function LoadGLTFExtras(loader) {
    this.name = "load_gltf_extras";
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
      for await (const [wstate, elapsed_times] of client.worldstate(
        simulationId
      )) {
        if (!stopPolling) {
          setWorldState(wstate);
          onElapsedTimesChanged(...elapsed_times);
        }
      }
    })();

    // Called when simulation ID changes
    return () => (stopPolling = true);
  }, [simulationId]);

  // Load map
  useEffect(() => {
    if (scene == null || worldState.scenario_id == null) {
      return;
    }

    for (const mesh of mapMeshes) {
      // doNotRecurse = false, disposeMaterialAndTextures = true
      mesh.dispose(false, true);
    }

    let mapSourceUrl = `${client.endpoint.origin}/assets/maps/${worldState.scenario_id}.glb`;
    let mapRootUrl = Tools.GetFolderPath(mapSourceUrl);
    let mapFilename = Tools.GetFilename(mapSourceUrl);

    // Load extra information attached to the map glb
    GLTFLoader.RegisterExtension("load_gltf_extras", function (loader) {
      return new LoadGLTFExtras(loader);
    });

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

      setMapMeshes(meshes);
      GLTFLoader.UnregisterExtension("load_gltf_extras");
    });
  }, [scene, worldState.scenario_id]);

  // Lane dividers
  useEffect(() => {
    if (scene == null || worldState.scenario_id == null) {
      return;
    }

    for (const geom of laneDividerGeometry) {
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

    setLaneDividerGeometry(newLaneDividers);
  }, [scene, JSON.stringify(laneDividerPos)]);

  // Edge dividers
  useEffect(() => {
    if (scene == null || worldState.scenario_id == null) {
      return;
    }

    if (edgeDividerGeometry != null) {
      edgeDividerGeometry.dispose();
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

    setEdgeDividerGeometry(newEdgeDividers);
  }, [scene, JSON.stringify(edgeDividerPos)]);

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
      <Bubbles scene={scene} worldState={worldState} />
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
      {showScores ? (
        <AgentScores
          style={{
            zIndex: "1",
            position: "absolute",
            top: "0",
            left: "0",
            maxWidth: "100%",
          }}
          scores={worldState.scores}
        />
      ) : null}
    </div>
  );
}
