import geopandas as gpd
from shapely.geometry import box

# Charger le grid mondial
grid = gpd.read_file('grid.gpkg')

# Créer un rectangle Égypte directement
# Format shapely box : (minx, miny, maxx, maxy) = (West, South, East, North)
egypt_box = box(25, 22, 37, 32)
egypt_gdf = gpd.GeoDataFrame(geometry=[egypt_box], crs='EPSG:4326')

# Reprojeter si nécessaire
grid = grid.to_crs('EPSG:4326')

# Clipper
grid_egypt = gpd.clip(grid, egypt_gdf)

# Sauvegarder
grid_egypt.to_file('grid_egypt.gpkg', driver='GPKG')