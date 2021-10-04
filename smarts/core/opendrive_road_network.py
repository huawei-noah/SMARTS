import logging

from lxml import etree
from opendrive2lanelet.opendriveparser.elements.opendrive import OpenDrive
from opendrive2lanelet.opendriveparser.elements.road import Road as RoadElement
from opendrive2lanelet.opendriveparser.parser import parse_opendrive

from smarts.core.road_map import RoadMap


class OpenDriveRoadNetwork(RoadMap):
    def __init__(self, network: OpenDrive, xodr_file: str):
        self._log = logging.getLogger(self.__class__.__name__)
        self._network = network
        self._xodr_file = xodr_file
        self._lanes = {}
        self._roads = {}
        self._lanepoints = None

    @classmethod
    def from_file(
        cls,
        xodr_file,
    ):
        with open(xodr_file, "r") as f:
            network = parse_opendrive(etree.parse(f).getroot())

        return cls(
            network,
            xodr_file,
        )

    @property
    def source(self) -> str:
        """ This is the .xodr file of the OpenDRIVE map. """
        return self._xodr_file

    class Road(RoadMap.Road):
        def __init__(self, road_id: str, road_elem: RoadElement, road_map: RoadMap):
            self._road_id = road_id
            self._road_elem = road_elem
            self._map = road_map

    def road_by_id(self, road_id: str) -> RoadMap.Road:
        road = self._roads.get(road_id)
        if road:
            return road
        road_elem = self._network.getRoad(int(road_id))
        if not road_elem:
            self._log.warning(
                f"OpenDriveRoadNetwork got request for unknown road_id '{road_id}'"
            )
            return None
        road = OpenDriveRoadNetwork.Road(road_id, road_elem, self)
        self._roads[road_id] = road
        return road
