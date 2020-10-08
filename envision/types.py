from enum import Enum
from typing import NamedTuple, Dict, Sequence, Tuple

from smarts.core.events import Events


class TrafficActorType(str, Enum):
    SocialVehicle = "social_vehicle"
    SocialAgent = "social_agent"
    Agent = "agent"


class VehicleType(str, Enum):
    Bus = "bus"
    Coach = "coach"
    Truck = "truck"
    Trailer = "trailer"
    Car = "car"


class TrafficActorState(NamedTuple):
    actor_type: TrafficActorType
    vehicle_type: VehicleType
    position: Tuple[float, float, float]
    heading: float
    speed: float
    name: str = ""
    actor_id: str = None
    events: Events = None
    waypoint_paths: Sequence = []
    driven_path: Sequence = []
    point_cloud: Sequence = []
    mission_route_geometry: Sequence[Sequence[Tuple[float, float]]] = None


class State(NamedTuple):
    traffic: Dict[str, TrafficActorState]
    scenario_id: str
    # sequence of x, y coordinates
    bubbles: Sequence[Sequence[Tuple[float, float]]]
    scene_colors: Dict[str, Tuple[float, float, float, float]]
    scores: Dict[str, float]


def format_actor_id(actor_id: str, vehicle_id: str, is_multi: bool):
    if is_multi:
        return f"{actor_id}-{{{vehicle_id[:4]}}}"
    return actor_id
