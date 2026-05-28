from pathlib import Path

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.transform import from_bounds

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "Groundwater" / "a_raw"
OUT_DIR = BASE_DIR / "Groundwater" / "b_processed" / "bronze"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# SETTINGS
# Resolution in degrees — 0.05 = ~5km (maximum resolution of BGS data)
# Change to 0.1 for faster processing, 0.27 for coarser (original BGS resolution)
# =============================================================================
RESOLUTION = 0.05

# Africa bounding box
LON_MIN, LON_MAX = -20, 55
LAT_MIN, LAT_MAX = -35, 38

# =============================================================================
# HELPER FUNCTION — converts text categories to numeric midpoint values
# and saves as GeoTiff raster
# =============================================================================
def points_to_raster(df, x_col, y_col, value_col, mapping_mid, output_file):
    # Convert categories to numeric midpoints
    df = df.copy()
    df["VALUE_NUM"] = df[value_col].map(mapping_mid)
    df = df.dropna(subset=["VALUE_NUM"])

    # Create grid
    lon_grid = np.arange(LON_MIN, LON_MAX, RESOLUTION)
    lat_grid = np.arange(LAT_MIN, LAT_MAX, RESOLUTION)

    # Grid dimensions
    n_rows = len(lat_grid)
    n_cols = len(lon_grid)

    # Create empty grid filled with nodata
    grid_values = np.full((n_rows, n_cols), -9999.0)

    # Simply place each point value into the nearest grid cell
    # No interpolation — just direct assignment
    for _, row in df.iterrows():
        col_idx = int((row[x_col] - LON_MIN) / RESOLUTION)
        row_idx = int((LAT_MAX - row[y_col]) / RESOLUTION)

        if 0 <= col_idx < n_cols and 0 <= row_idx < n_rows:
            grid_values[row_idx, col_idx] = row["VALUE_NUM"]

    # Define raster transform
    transform = from_bounds(
        LON_MIN, LAT_MIN, LON_MAX, LAT_MAX,
        n_cols, n_rows
    )

    # Save as GeoTiff
    with rasterio.open(
        output_file,
        "w",
        driver="GTiff",
        height=n_rows,
        width=n_cols,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999
    ) as dst:
        dst.write(grid_values.astype("float32"), 1)

    print(f"Done: {output_file}")


# =============================================================================
# FILE 1: GROUNDWATER PRODUCTIVITY (xyzASCII_gwprod_v1)
# Unit: litres/second (L/s)
# Represents: average borehole yield — proxy for transmissivity
# =============================================================================
df_prod = pd.read_csv(RAW_DIR / "xyzASCII_gwprod_v1.txt", sep="\t")

# Midpoint values in L/s for raster assignment
prod_mid = {"VH": 30, "H": 12.5, "M": 3, "LM": 0.75, "L": 0.3, "VL": 0.05}

# Min/Max ranges kept in shapefile for Monte Carlo sampling
prod_min = {"VH": 20, "H": 5, "M": 1, "LM": 0.5, "L": 0.1, "VL": 0}
prod_max = {"VH": 999, "H": 20, "M": 5, "LM": 1, "L": 0.5, "VL": 0.1}

df_prod["PROD_MIN"] = df_prod["GWPROD_V2"].map(prod_min)
df_prod["PROD_MAX"] = df_prod["GWPROD_V2"].map(prod_max)

# Save shapefile with min/max for model
geometry_prod = [Point(xy) for xy in zip(df_prod["X"], df_prod["Y"])]
gdf_prod = gpd.GeoDataFrame(df_prod, geometry=geometry_prod, crs="EPSG:4326")
gdf_prod.to_file(OUT_DIR / "gwprod_africa.shp")

# Save raster for QGIS visualisation and spatial analysis
points_to_raster(
    df_prod,
    "X", "Y", "GWPROD_V2",
    prod_mid,
    OUT_DIR / "gwprod_africa.tif"
)


# =============================================================================
# FILE 2: GROUNDWATER STORAGE (xyzASCII_gwstor_v1)
# Unit: water depth in mm
# Represents: total volume of groundwater stored per unit area
# =============================================================================
df_stor = pd.read_csv(RAW_DIR / "xyzASCII_gwstor_v1.txt", sep="\t")

# Midpoint values in mm for raster assignment
stor_mid = {"0": 0, "L": 500, "LM": 5500, "M": 17500, "H": 37500, "VH": 75000}

# Min/Max ranges kept in shapefile for Monte Carlo sampling
stor_min = {"0": 0, "L": 0, "LM": 1000, "M": 10000, "H": 25000, "VH": 50000}
stor_max = {"0": 0, "L": 1000, "LM": 10000, "M": 25000, "H": 50000, "VH": 999999}

df_stor["STOR_MIN"] = df_stor["GWSTOR_V2"].map(stor_min)
df_stor["STOR_MAX"] = df_stor["GWSTOR_V2"].map(stor_max)

# Save shapefile with min/max for model
geometry_stor = [Point(xy) for xy in zip(df_stor["X"], df_stor["Y"])]
gdf_stor = gpd.GeoDataFrame(df_stor, geometry=geometry_stor, crs="EPSG:4326")
gdf_stor.to_file(OUT_DIR / "gwstor_africa.shp")

# Save raster for QGIS visualisation and spatial analysis
points_to_raster(
    df_stor,
    "X", "Y", "GWSTOR_V2",
    stor_mid,
    OUT_DIR / "gwstor_africa.tif"
)


# =============================================================================
# FILE 3: DEPTH TO GROUNDWATER (xyzASCII_gwdept_v2) renamed from dtwmap
# Unit: metres below ground level (mbgl)
# Represents: static water table depth
# =============================================================================
df_dtw = pd.read_csv(RAW_DIR / "xyzASCII_gwdept_v2.txt", sep="\t")

# Midpoint values in metres below ground level for raster assignment
dtw_mid = {"VS": 3.5, "S": 16, "SM": 37.5, "M": 75, "D": 175, "VD": 350}

# Min/Max ranges kept in shapefile for Monte Carlo sampling
dtw_min = {"VS": 0, "S": 7, "SM": 25, "M": 50, "D": 100, "VD": 250}
dtw_max = {"VS": 7, "S": 25, "SM": 50, "M": 100, "D": 250, "VD": 999}

df_dtw["DTW_MIN"] = df_dtw["DTWAFRICA_"].map(dtw_min)
df_dtw["DTW_MAX"] = df_dtw["DTWAFRICA_"].map(dtw_max)

# Save shapefile with min/max for model
geometry_dtw = [Point(xy) for xy in zip(df_dtw["X"], df_dtw["Y"])]
gdf_dtw = gpd.GeoDataFrame(df_dtw, geometry=geometry_dtw, crs="EPSG:4326")
gdf_dtw.to_file(OUT_DIR / "gwdept_africa.shp")

# Save raster for QGIS visualisation and spatial analysis
points_to_raster(
    df_dtw,
    "X", "Y", "DTWAFRICA_",
    dtw_mid,
    OUT_DIR / "gwdept_africa.tif"
)

print("\nAll files saved to:")
print(OUT_DIR)
