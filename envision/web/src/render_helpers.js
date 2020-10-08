import {
  Color3,
  StandardMaterial,
  Mesh,
  MeshBuilder,
  DynamicTexture,
} from "@babylonjs/core";

import { VehicleTypes, ActorTypes } from "./enums.js";

export function vehicleMeshFilename(actorType, vehicleType) {
  if (vehicleType == VehicleTypes.BUS) {
    return "bus.glb";
  }

  if (vehicleType == VehicleTypes.COACH) {
    return "coach.glb";
  }

  if (vehicleType == VehicleTypes.TRUCK) {
    return "truck.glb";
  }

  if (vehicleType == VehicleTypes.TRAILER) {
    return "trailer.glb";
  }

  if (vehicleType == VehicleTypes.CAR) {
    if (actorType == ActorTypes.SOCIAL_AGENT) {
      return "muscle_car_social_agent.glb";
    } else if (actorType == ActorTypes.AGENT) {
      return "muscle_car_agent.glb";
    }
  }

  return "simple_car.glb";
}

export function vehicleMeshColor(actorType, scene_colors) {
  if (actorType == ActorTypes.SOCIAL_AGENT) {
    return scene_colors["social_agent"];
  } else if (actorType == ActorTypes.AGENT) {
    return scene_colors["agent"];
  } else {
    return scene_colors["social_vehicle"];
  }
}

export function buildLabel(name, text, scene) {
  let width = 10;
  let height = 2;
  let plane = MeshBuilder.CreatePlane(
    name,
    { width: width, height: height },
    scene,
    false
  );
  plane.billboardMode = Mesh.BILLBOARDMODE_ALL;

  let texture = new DynamicTexture(
    `${name}-texture`,
    { width: width * 100, height: height * 100 },
    scene,
    true
  );
  texture.hasAlpha = true;
  texture.drawText(text, null, null, "bold 100px arial", "white");

  let material = new StandardMaterial(`${name}-material`, scene);
  material.diffuseTexture = texture;
  material.specularColor = new Color3(0, 0, 0);
  material.emissiveColor = new Color3(1, 1, 1);
  material.backFaceCulling = false;

  plane.material = material;
  return plane;
}
