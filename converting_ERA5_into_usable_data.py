from pathlib import Path
import cdsapi
import zipfile

BASE_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "ERA5"/"b_processed"/"bronze"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ZIP_PATH = OUTPUT_DIR / "era5_egypt_monthly_2018_2022.zip"
NC_PATH  = OUTPUT_DIR / "era5_egypt_monthly_2018_2022.nc"

# =============================================================================
# STEP 1 — Download
# =============================================================================
client = cdsapi.Client()

dataset = "reanalysis-era5-land-monthly-means"

request = {
    "product_type": "monthly_averaged_reanalysis",
    
    "variable": [
        "2m_temperature",
        "2m_dewpoint_temperature",
        "surface_net_solar_radiation",
        "surface_net_thermal_radiation",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "surface_pressure",
        "total_precipitation",
    ],

    "year": [str(y) for y in range(2018, 2023)],

    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],

    "time": "00:00",

    # Egypt bounding box
    # CDS format: [North, West, South, East]
    # Egypt: North=32, West=25, South=22, East=37
    "area": [32, 25, 22, 37],

    "format": "netcdf"
}

print("Downloading ERA5 data...")
client.retrieve(dataset, request).download(ZIP_PATH)
print(f"Download complete — {ZIP_PATH}")

# =============================================================================
# STEP 2 — Unzip
# =============================================================================
print("Extracting...")
with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    contents = z.namelist()
    print(f"Files inside zip: {contents}")
    
    nc_files = [f for f in contents if f.endswith('.nc')]
    z.extract(nc_files[0], OUTPUT_DIR)
    extracted = OUTPUT_DIR / nc_files[0]
    extracted.rename(NC_PATH)

ZIP_PATH.unlink()
print(f"Done — NetCDF saved at: {NC_PATH}")