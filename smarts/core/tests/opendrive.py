from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import math
from os import path
from pathlib import Path
import matplotlib.pyplot as plt
from lxml import etree
from opendrive2lanelet.opendriveparser.elements.geometry import Line
from opendrive2lanelet.opendriveparser.elements.roadLanes import Lane as LaneElement
from opendrive2lanelet.opendriveparser.elements.roadLanes import LaneSection
from opendrive2lanelet.opendriveparser.elements.roadPlanView import PlanView
from opendrive2lanelet.opendriveparser.parser import parse_opendrive


def constrain_angle(angle):
    """Constrain to [-pi, pi]"""
    angle = angle % (2 * math.pi)
    if angle > math.pi:
        angle -= 2 * math.pi
    return angle


def t_angle(lane_id: int, s_heading: float):
    angle = (s_heading - math.pi / 2) if lane_id < 0 else (s_heading + math.pi / 2)
    return constrain_angle(angle)


@dataclass(frozen=True)
class CubicPolynomial:
    a: float
    b: float
    c: float
    d: float

    @classmethod
    def from_list(cls, coefficients: List[float]):
        return cls(
            a=coefficients[0],
            b=coefficients[1],
            c=coefficients[2],
            d=coefficients[3],
        )

    def eval(self, ds: float) -> float:
        return self.a + self.b * ds + self.c * ds * ds + self.d * ds * ds * ds


def refline_to_linear_segments(
    planview: PlanView, s_start: float, s_end: float
) -> List[float]:
    s_vals = []
    for geom in planview._geometries:
        if type(geom) == Line:
            s_vals.extend([s_start, s_end])
        else:
            num_segments = int((s_end - s_start) / 0.1)
            for seg in range(num_segments):
                s_vals.append(seg * 0.1)
    return sorted(s_vals)


@dataclass
class LaneBoundary:
    refline: PlanView
    inner: "LaneBoundary"
    poly: CubicPolynomial

    def calc_t(self, ds: float) -> float:
        if not self.inner:
            return 0
        return self.poly.eval(ds) + self.inner.calc_t(ds)

    def to_linear_segments(self, s_start: float, s_end: float):
        inner_s_vals = None
        outer_s_vals = None

        if self.inner:
            inner_s_vals = self.inner.to_linear_segments(s_start, s_end)
        else:
            return refline_to_linear_segments(self.refline, s_start, s_end)

        if self.poly.c == 0 and self.poly.d == 0:
            outer_s_vals = [s_start, s_end]
        else:
            num_segments = int((s_end - s_start) / 0.1)
            outer_s_vals = []
            for seg in range(num_segments):
                outer_s_vals.append(seg * 0.1)
        return sorted(set(inner_s_vals + outer_s_vals))


def width_id(road, section, lane, width):
    return f"{road.id}_{section.idx}_{lane.id}_{width.idx}"


def lane_vertices(
    lane: LaneElement,
    planview: PlanView,
    boundaries: Dict[str, Tuple[LaneBoundary, LaneBoundary]],
):
    xs, ys = [], []
    section: LaneSection = lane.lane_section
    section_len = section.length
    section_s_start = section.sPos
    section_s_end = section_s_start + section_len

    for width in lane.widths:
        w_id = width_id(lane.parentRoad, section, lane, width)
        (inner_boundary, outer_boundary) = boundaries[w_id]
        inner_s_vals = inner_boundary.to_linear_segments(section_s_start, section_s_end)
        outer_s_vals = outer_boundary.to_linear_segments(section_s_start, section_s_end)
        s_vals = sorted(set(inner_s_vals + outer_s_vals))

        xs_inner, ys_inner = [], []
        xs_outer, ys_outer = [], []
        for s in s_vals:
            print(s)
            ds = s - width.start_offset
            t_inner = inner_boundary.calc_t(ds)
            t_outer = outer_boundary.calc_t(ds)
            (x_ref, y_ref), heading = planview.calc(s)
            angle = t_angle(lane.id, heading)
            xs_inner.append(x_ref + t_inner * math.cos(angle))
            ys_inner.append(y_ref + t_inner * math.sin(angle))
            xs_outer.append(x_ref + t_outer * math.cos(angle))
            ys_outer.append(y_ref + t_outer * math.sin(angle))
        inner_boundary = outer_boundary
        xs.extend(xs_inner + xs_outer[::-1] + [xs_inner[0]])
        ys.extend(ys_inner + ys_outer[::-1] + [ys_inner[0]])
    return xs, ys


def plot_road(road_elem):
    planview = road_elem.planView

    # Create boundaries
    boundaries: Dict[str, Tuple[LaneBoundary, LaneBoundary]] = {}
    for section in road_elem.lanes.lane_sections:
        inner_boundary = LaneBoundary(planview, None, None)
        for lane in section.leftLanes:
            for width in lane.widths:
                poly = CubicPolynomial.from_list(width.polynomial_coefficients)
                outer_boundary = LaneBoundary(None, inner_boundary, poly)
                w_id = width_id(road_elem, section, lane, width)
                boundaries[w_id] = (inner_boundary, outer_boundary)
                inner_boundary = outer_boundary

        inner_boundary = LaneBoundary(planview, None, None)
        for lane in section.rightLanes:
            for width in lane.widths:
                poly = CubicPolynomial.from_list(width.polynomial_coefficients)
                outer_boundary = LaneBoundary(None, inner_boundary, poly)
                w_id = width_id(road_elem, section, lane, width)
                boundaries[w_id] = (inner_boundary, outer_boundary)
                inner_boundary = outer_boundary

    # Plot lane polygons
    for section in road_elem.lanes.lane_sections:
        for lane in section.leftLanes + section.rightLanes:
            xs, ys = lane_vertices(lane, planview, boundaries)
            plt.scatter(xs, ys, s=1)
            plt.plot(xs, ys, "k-")

    # Plot refline
    refline_s_vals = refline_to_linear_segments(planview, 0, planview.length)
    xs, ys = [], []
    for s in refline_s_vals:
        (x_ref, y_ref), heading = planview.calc(s)
        xs.append(x_ref)
        ys.append(y_ref)
    plt.scatter(xs, ys, s=1, c="r")
    plt.plot(xs, ys, "r-")


def view():
    fig, ax = plt.subplots()

    root = path.join(Path(__file__).parent.absolute(), "maps")
    with open(path.join(root, "Ex_Simple-LaneOffset.xodr"), "r") as f:
        od = parse_opendrive(etree.parse(f).getroot())

    for road in od.roads:
        plot_road(road)
    source = path.join(root, "Ex_Simple-LaneOffset.xodr")
    ax.set_title(f"{source}")
    ax.axis("equal")
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()


if __name__ == "__main__":
    view()