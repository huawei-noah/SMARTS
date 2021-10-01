import logging
from lxml import etree
from opendrive2lanelet.opendriveparser.parser import parse_opendrive

from smarts.core.road_map import RoadMap


class OpenDriveRoadNetwork(RoadMap):
    def __init__(self, network, xodr_file):
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
