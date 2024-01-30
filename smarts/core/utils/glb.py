# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import math
import warnings
from pathlib import Path
from typing import Any, Dict, Final, List, Tuple, Union

import numpy as np
from shapely.geometry import Polygon

from smarts.core.coordinates import BoundingBox
from smarts.core.utils.geometry import triangulate_polygon

# Suppress trimesh deprecation warning
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.",
        category=DeprecationWarning,
    )
    import trimesh  # only suppress the warnings caused by trimesh
    from trimesh.exchange import gltf
    from trimesh.visual.material import PBRMaterial

OLD_TRIMESH: Final[bool] = tuple(int(d) for d in trimesh.__version__.split(".")) <= (
    3,
    9,
    29,
)


def _convert_camera(camera):
    result = {
        "name": camera.name,
        "type": "perspective",
        "perspective": {
            "aspectRatio": camera.fov[0] / camera.fov[1],
            "yfov": np.radians(camera.fov[1]),
            "znear": float(camera.z_near),
            # HACK: The trimesh gltf export doesn't include a zfar which Panda3D GLB
            #       loader expects. Here we override to make loading possible.
            "zfar": float(camera.z_near + 100),
        },
    }
    return result


gltf._convert_camera = _convert_camera


class GLBData:
    """Convenience class for writing GLB files."""

    def __init__(self, bytes_):
        self._bytes = bytes_

    def write_glb(self, output_path: Union[str, Path]):
        """Generate a geometry file."""
        with open(output_path, "wb") as f:
            f.write(self._bytes)


def _generate_meshes_from_polygons(
    polygons: List[Tuple[Polygon, Dict[str, Any]]]
) -> List[trimesh.Trimesh]:
    """Creates a mesh out of a list of polygons."""
    meshes = []

    # Trimesh's API require a list of vertices and a list of faces, where each
    # face contains three indexes into the vertices list. Ideally, the vertices
    # are all unique and the faces list references the same indexes as needed.
    # TODO: Batch the polygon processing.
    for poly, metadata in polygons:
        vertices, faces = [], []
        point_dict = dict()
        current_point_index = 0

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

        if not vertices or not faces:
            continue

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, metadata=metadata)

        # Trimesh doesn't support a coordinate-system="z-up" configuration, so we
        # have to apply the transformation manually.
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(math.pi / 2, [-1, 0, 0])
        )
        meshes.append(mesh)
    return meshes


def make_map_glb(
    polygons: List[Tuple[Polygon, Dict[str, Any]]],
    bbox: BoundingBox,
    lane_dividers,
    edge_dividers,
) -> GLBData:
    """Create a GLB file from a list of road polygons."""

    # Attach additional information for rendering as metadata in the map glb
    metadata = {
        "bounding_box": (
            bbox.min_pt.x,
            bbox.min_pt.y,
            bbox.max_pt.x,
            bbox.max_pt.y,
        ),
        "lane_dividers": lane_dividers,
        "edge_dividers": edge_dividers,
    }
    scene = trimesh.Scene(metadata=metadata)

    meshes = _generate_meshes_from_polygons(polygons)
    material = PBRMaterial("RoadDefault")
    for mesh in meshes:
        mesh.visual.material = material
        road_id = mesh.metadata["road_id"]
        lane_id = mesh.metadata.get("lane_id")
        name = str(road_id)
        if lane_id is not None:
            name += f"-{lane_id}"
        if OLD_TRIMESH:
            scene.add_geometry(mesh, name, extras=mesh.metadata)
        else:
            scene.add_geometry(mesh, name, geom_name=name, metadata=mesh.metadata)
    return GLBData(gltf.export_glb(scene, include_normals=True))


def make_road_line_glb(lines: List[List[Tuple[float, float]]]) -> GLBData:
    """Create a GLB file from a list of road/lane lines."""
    scene = trimesh.Scene()
    material = trimesh.visual.material.PBRMaterial()
    for line_pts in lines:
        vertices = [(*pt, 0.1) for pt in line_pts]
        point_cloud = trimesh.PointCloud(vertices=vertices)
        point_cloud.apply_transform(
            trimesh.transformations.rotation_matrix(math.pi / 2, [-1, 0, 0])
        )
        scene.add_geometry(point_cloud)
    return GLBData(gltf.export_glb(scene))
