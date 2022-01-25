# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import math
from typing import List

import numpy as np
import trimesh
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.geometry.base import CAP_STYLE, JOIN_STYLE
from shapely.ops import triangulate


def buffered_shape(shape, width: float = 1.0) -> Polygon:
    """Generates a shape with a buffer of `width` around the original shape."""
    ls = LineString(shape).buffer(
        width / 2,
        1,
        cap_style=CAP_STYLE.flat,
        join_style=JOIN_STYLE.round,
        mitre_limit=5.0,
    )
    if isinstance(ls, MultiPolygon):
        # Sometimes it oddly outputs a MultiPolygon and then we need to turn it into a convex hull
        ls = ls.convex_hull
    elif not isinstance(ls, Polygon):
        raise RuntimeError("Shapely `object.buffer` behavior may have changed.")
    return ls


def triangulate_polygon(polygon: Polygon):
    """Attempts to convert a polygon into triangles."""
    # XXX: shapely.ops.triangulate current creates a convex fill of triangles.
    return [
        tri_face
        for tri_face in triangulate(polygon)
        if tri_face.centroid.within(polygon)
    ]


def generate_mesh_from_polygons(polygons: List[Polygon]) -> trimesh.Trimesh:
    """Creates a mesh out of a list of polygons."""
    vertices, faces = [], []
    point_dict = dict()
    current_point_index = 0

    # Trimesh's API require a list of vertices and a list of faces, where each
    # face contains three indexes into the vertices list. Ideally, the vertices
    # are all unique and the faces list references the same indexes as needed.
    # TODO: Batch the polygon processing.
    for poly in polygons:
        # Collect all the points on the shape to reduce checks by 3 times
        for x, y in poly.exterior.coords:
            p = (x, y, 0)
            if p not in point_dict:
                vertices.append(p)
                point_dict[p] = current_point_index
                current_point_index += 1
        triangles = triangulate_polygon(poly)
        for triangle in triangles:
            face = np.array(
                [point_dict.get((x, y, 0), -1) for x, y in triangle.exterior.coords]
            )
            # Add face if not invalid
            if -1 not in face:
                faces.append(face)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Trimesh doesn't support a coordinate-system="z-up" configuration, so we
    # have to apply the transformation manually.
    mesh.apply_transform(
        trimesh.transformations.rotation_matrix(math.pi / 2, [-1, 0, 0])
    )
    return mesh
