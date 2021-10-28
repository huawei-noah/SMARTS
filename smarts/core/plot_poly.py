from shapely.geometry import Polygon
from centerline.geometry import Centerline

polygon = Polygon([[0, 0], [0, 4], [4, 4], [4, 0]])
attributes = {"id": 1, "name": "polygon", "valid": True}


centerline = Centerline(polygon, **attributes)
assert centerline.id == 1
print(centerline.name)
