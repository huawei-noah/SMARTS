import React, { useState, useEffect } from "react";

import {
  ArcRotateCamera,
  Vector2,
  Vector3,
  Color3,
  Tools,
  SceneLoader,
  StandardMaterial,
  Quaternion,
  HemisphericLight,
  MeshBuilder,
  Mesh,
  Color4,
  BoundingInfo,
  UniversalCamera,
  TransformNode,
} from "@babylonjs/core";

import { GLTFLoader } from "@babylonjs/loaders/glTF/2.0/glTFLoader";
import SceneComponent from "babylonjs-hook";
import { ActorTypes } from "../enums.js";
import AgentScores from "./agent_scores";
import earcut from "earcut";
import { intersection, difference } from "../math.js";
import {
  vehicleMeshFilename,
  vehicleMeshColor,
  buildLabel,
} from "../render_helpers.js";

// Required by Babylon.js
window.earcut = earcut;

export default ({ simulationId, client, showScores, egoView, canvasRef }) => {
  const [scene, setScene] = useState(null);
  const [camera, setCamera] = useState(null);
  const [thirdPersonCamera, setThirdPersonCamera] = useState(null);
  const [egoCamRoot, setEgoCamRoot] = useState(null);
  const [egoWaypointModel, setEgoWaypointModel] = useState(null);
  const [socialWaypointModel, setSocialWaypointModel] = useState(null);
  const [egoDrivenPathModel, setEgoDrivenPathModel] = useState(null);
  const [socialDrivenPathModel, setSocialDrivenPathModel] = useState(null);

  const [mapMeshes, setMapMeshes] = useState([]);
  const [vehicleMeshes, setVehicleMeshes] = useState({}); // {<id>: <mesh>}
  const [agentLabelGeometry, setAgentLabelGeometry] = useState({});
  const [bubbleGeometry, setBubbleGeometry] = useState([]); // List of mesh objects
  const [missionGeometry, setMissionGeometry] = useState([]);
  const [waypointGeometries, setWaypointGeometries] = useState([]);
  const [drivenPathGeometries, setDrivenPathGeometries] = useState({});
  const [vehicleMeshTemplates, setVehicleMeshTemplates] = useState({});

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

  let meshesLoaded = () => {
    for (const [_, mesh] of Object.entries(vehicleMeshTemplates)) {
      if (mesh == null) {
        return false;
      }
    }
    return true;
  };

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

    // Default third-person camera
    let thirdPersonCamera_ = new ArcRotateCamera(
      "third-person-camera",
      -Math.PI / 2, // alpha
      0, // beta
      200, // radius
      new Vector3(0, 0, 0), // target
      scene_
    );
    thirdPersonCamera_.attachControl(canvas, true);
    thirdPersonCamera_.panningSensibility = 50;
    thirdPersonCamera_.lowerRadiusLimit = 5;
    setThirdPersonCamera(thirdPersonCamera_);
    setCamera(thirdPersonCamera_);
    scene_.activeCamera = thirdPersonCamera_; // default camera

    // Ego-centric camera
    let egoCamRoot_ = new TransformNode("ego-camera-root");
    egoCamRoot_.position = new Vector3.Zero(); // Set to the ego vehicle position during update
    let egoCamera_ = new UniversalCamera(
      "ego-camera",
      new Vector3(0, 5, -15), // Relative to camera root position
      scene
    );
    egoCamera_.parent = egoCamRoot_;
    setEgoCamRoot(egoCamRoot_);

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
      for await (let wstate of client.worldstate(simulationId)) {
        if (!stopPolling) {
          setWorldState(wstate);
        }
      }
    })();

    // Called when simulation ID changes
    return () => (stopPolling = true);
  }, [simulationId]);

  // Set camera
  useEffect(() => {
    if (egoCamRoot == null || thirdPersonCamera == null) {
      return;
    }

    let canvas = scene.getEngine().getRenderingCanvas();
    if (egoView) {
      let egoCamera = egoCamRoot.getChildren()[0];
      egoCamera.rotation = new Vector3.Zero();
      scene.activeCamera = egoCamera;

      // Disable mouse input to the third person camera during ego view
      thirdPersonCamera.detachControl(canvas);

      setCamera(egoCamRoot);
    } else {
      thirdPersonCamera.attachControl(canvas, true);
      scene.activeCamera = thirdPersonCamera;
      setCamera(thirdPersonCamera);
    }
  }, [egoView]);

  // Load mesh asynchronously
  useEffect(() => {
    if (scene == null) {
      return;
    }

    let vehicleRootUrl = `${client.endpoint.origin}/assets/models/`;

    for (const [vehicleFilename, meshTemplate] of Object.entries(
      vehicleMeshTemplates
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

          setVehicleMeshTemplates((vehicleMeshTemplates) => ({
            ...vehicleMeshTemplates,
            [vehicleFilename]: rootMesh,
          }));
        }
      );
    }
    // This useEffect is triggered when the vehicleMeshTemplate's keys() change
  }, [scene, Object.keys(vehicleMeshTemplates).sort().join("-")]);

  // Update third person camera's pointing target and radius
  useEffect(() => {
    if (
      scene == null ||
      thirdPersonCamera == null ||
      worldState.scenario_id == null ||
      roadNetworkBbox.length != 4
    ) {
      return;
    }

    let mapCenter = [
      (roadNetworkBbox[0] + roadNetworkBbox[2]) / 2,
      (roadNetworkBbox[1] + roadNetworkBbox[3]) / 2,
    ];
    thirdPersonCamera.target.x = mapCenter[0];
    thirdPersonCamera.target.z = mapCenter[1];
    thirdPersonCamera.radius = Math.max(
      Math.abs(roadNetworkBbox[0] - roadNetworkBbox[2]),
      Math.abs(roadNetworkBbox[1] - roadNetworkBbox[3])
    );
  }, [scene, thirdPersonCamera, JSON.stringify(roadNetworkBbox)]);

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

  // Vehicles and agent labels
  useEffect(() => {
    if (!meshesLoaded()) {
      return;
    }

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
      if (!vehicleMeshTemplates[filename]) {
        // Triggers loading the mesh according through the useEffect
        setVehicleMeshTemplates((vehicleMeshTemplates) => ({
          ...vehicleMeshTemplates,
          [filename]: null,
        }));
        continue;
      }

      let color = vehicleMeshColor(state.actor_type, worldState.scene_colors);
      let rootMesh = new Mesh(`root-mesh-${meshId}`, scene);
      let childMeshes = vehicleMeshTemplates[filename].getChildMeshes();
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
        let boundingInfo = vehicleMeshTemplates[filename].getBoundingInfo();
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
        camera.position = new Vector3(pos[0], pos[2], pos[1]);
        camera.rotation = new Vector3(0, 2 * Math.PI - state.heading, 0);
      }
    }

    setVehicleMeshes(nextVehicleMeshes);
    setAgentLabelGeometry(nextAgentLabelGeometry);
  }, [scene, worldState.traffic]);

  // Bubble geometry
  useEffect(() => {
    if (scene == null) {
      return;
    }

    for (const geom of bubbleGeometry) {
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
    setBubbleGeometry(newBubbleGeometry);

    // Bubbles only change from scenario to scenario, this will prevent unnecessary work
  }, [scene, JSON.stringify(worldState.bubbles)]);

  // Mission route geometry
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

  useEffect(() => {
    if (scene == null) {
      return;
    }

    for (const geom of missionGeometry) {
      // doNotRecurse = false, disposeMaterialAndTextures = true
      geom.dispose(false, true);
    }

    let nextMissionGeometry = [];
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
        nextMissionGeometry.push(polygon);
      });
    }
    setMissionGeometry(nextMissionGeometry);
  }, [scene, JSON.stringify(missionRouteStringify)]);

  // Waypoints geometry
  useEffect(() => {
    if (scene == null) {
      return;
    }

    for (const geom of waypointGeometries) {
      geom.dispose();
    }

    if (worldState.traffic.length == 0) {
      return;
    }

    if (egoWaypointModel.material == null) {
      egoWaypointModel.material = new StandardMaterial(
        "ego-waypoint-material",
        scene
      );
      egoWaypointModel.material.specularColor = new Color3(0, 0, 0);
      egoWaypointModel.material.diffuseColor = new Color4(
        ...worldState.scene_colors["ego_waypoint"]
      );
      egoWaypointModel.material.alpha =
        worldState.scene_colors["ego_waypoint"][3];
    }

    if (socialWaypointModel.material == null) {
      socialWaypointModel.material = new StandardMaterial(
        "social-waypoint-material",
        scene
      );
      socialWaypointModel.material.specularColor = new Color3(0, 0, 0);
      let color = vehicleMeshColor(
        ActorTypes.SOCIAL_AGENT,
        worldState.scene_colors
      );
      socialWaypointModel.material.diffuseColor = new Color4(...color);
      socialWaypointModel.material.alpha =
        worldState.scene_colors["ego_waypoint"][3];
    }

    let newWaypointGeometries = [];
    for (const [_, trafficActor] of Object.entries(worldState.traffic)) {
      for (const waypointPath of trafficActor.waypoint_paths) {
        for (const waypoint of waypointPath) {
          let wp_ = null;
          if (trafficActor.actor_type == ActorTypes.SOCIAL_AGENT) {
            wp_ = socialWaypointModel.createInstance("social-wp");
          } else {
            wp_ = egoWaypointModel.createInstance("ego-wp");
          }
          wp_.position.x = waypoint.pos[0];
          wp_.position.y = 0.15;
          wp_.position.z = waypoint.pos[1];
          newWaypointGeometries.push(wp_);
        }
      }
    }
    setWaypointGeometries(newWaypointGeometries);
  }, [scene, worldState.traffic]);

  // Driven path geometry
  useEffect(() => {
    if (scene == null) {
      return;
    }

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
    setDrivenPathGeometries(newDrivenPathGeometries);
  }, [scene, worldState.traffic]);

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
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
};
