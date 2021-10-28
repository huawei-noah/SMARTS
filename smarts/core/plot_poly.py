from shapely.geometry import Polygon
from centerline.geometry import Centerline
import matplotlib.pyplot as plt
import geopandas as gpd


polygon1 = Polygon([[0, 0], [0, 4], [4, 4], [4, 0]])
attributes = {"id": 1, "name": "polygon", "valid": True}

centerline = Centerline(polygon1, **attributes)
print(centerline.geoms)
p1 = gpd.GeoSeries(polygon1)
p1.plot()

plt.show()


