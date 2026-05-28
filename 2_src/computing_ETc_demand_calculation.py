import calendar
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray

from rasterio.enums import Resampling


# =============================================================================
# CONFIG
# =============================================================================

ERA5_PATH = Path("PROJECT/1_data/ERA5/era5_egypt_monthly_2018_2022.nc")
OUTPUT_PATH = Path("PROJECT/1_data/ERA5/outputs")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

ADJUST_KC_CLIMATE = False
RESCALE_STAGES_TO_CALENDAR = True

SAVE_NETCDF = True
SAVE_GEOTIFFS = True

TARGET_RESOLUTION = 0.05

# =============================================================================
# STEP 1 — Load ERA5 data
# =============================================================================

ds = xr.open_dataset(ERA5_PATH, engine="netcdf4")
print(ds)

TIME_DIM = "valid_time" if "valid_time" in ds.dims else "time"
LON_DIM = "longitude" if "longitude" in ds.coords else "lon"
LAT_DIM = "latitude" if "latitude" in ds.coords else "lat"

time_index = pd.to_datetime(ds[TIME_DIM].values)

days_in_month = xr.DataArray(
    time_index.days_in_month,
    coords={TIME_DIM: ds[TIME_DIM]},
    dims=[TIME_DIM],
)


# =============================================================================
# STEP 2 — Extract and convert ERA5 variables
# =============================================================================

Tmean = ds["t2m"] - 273.15
Tdew = ds["d2m"] - 273.15
P = ds["sp"] / 1000.0

if "mx2t" in ds:
    Tmax = ds["mx2t"] - 273.15
else:
    Tmax = Tmean

if "mn2t" in ds:
    Tmin = ds["mn2t"] - 273.15
else:
    Tmin = Tmean

u10 = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
u2 = u10 * (4.87 / np.log(67.8 * 10 - 5.42))

Rn_raw = (ds["ssr"] + ds["str"]) / 1e6

if float(Rn_raw.mean()) > 40:
    print("WARNING: Rn looks monthly accumulated. Dividing by days in month.")
    Rn = Rn_raw / days_in_month
else:
    Rn = Rn_raw


# =============================================================================
# STEP 3 — Compute FAO-56 Penman-Monteith ETo
# =============================================================================

def sat_vapour_pressure(T_c):
    return 0.6108 * np.exp((17.27 * T_c) / (T_c + 237.3))


es = (sat_vapour_pressure(Tmax) + sat_vapour_pressure(Tmin)) / 2
ea = sat_vapour_pressure(Tdew)
vpd = (es - ea).clip(min=0)

delta = (4098 * sat_vapour_pressure(Tmean)) / ((Tmean + 237.3) ** 2)
gamma = (1.013e-3 * P) / (0.622 * 2.45)

G = xr.zeros_like(Tmean)

numerator = (
    0.408 * delta * (Rn - G)
    + gamma * (900 / (Tmean + 273.15)) * u2 * vpd
)

denominator = delta + gamma * (1 + 0.34 * u2)

ETo = (numerator / denominator).clip(min=0)
ETo.name = "ETo"
ETo.attrs["units"] = "mm/day"
ETo.attrs["long_name"] = "Reference evapotranspiration FAO-56 Penman-Monteith"


# =============================================================================
# STEP 4 — Prepare optional climate adjustment variables
# =============================================================================

RHmin = 100 * ea / sat_vapour_pressure(Tmax)
RHmin = RHmin.clip(min=20, max=80)

u2_for_kc = u2.clip(min=1, max=6)


def adjust_kc_for_climate(kc_tab, h, template, u2_stage=None, RHmin_stage=None):
    if not ADJUST_KC_CLIMATE:
        return xr.zeros_like(template) + kc_tab

    if h <= 0.1:
        return xr.zeros_like(template) + kc_tab

    return (
        kc_tab
        + (0.04 * (u2_stage - 2) - 0.004 * (RHmin_stage - 45))
        * (h / 3) ** 0.3
    )


# =============================================================================
# STEP 5 — Define crop library
# =============================================================================

def annual_crop(
    tier,
    calendar_source,
    kc_source,
    planting_month,
    planting_day,
    Lini,
    Ldev,
    Lmid,
    Llate,
    Kc_ini,
    Kc_mid,
    Kc_end,
    h,
    harvest_month=None,
    harvest_day=None,
    proxy_for=None,
):
    crop = {
        "tier": tier,
        "calendar_source": calendar_source,
        "kc_source": kc_source,
        "planting_month": planting_month,
        "planting_day": planting_day,
        "Lini": Lini,
        "Ldev": Ldev,
        "Lmid": Lmid,
        "Llate": Llate,
        "Kc_ini": Kc_ini,
        "Kc_mid": Kc_mid,
        "Kc_end": Kc_end,
        "h": h,
    }

    if harvest_month is not None and harvest_day is not None:
        crop["harvest_month"] = harvest_month
        crop["harvest_day"] = harvest_day

    if proxy_for is not None:
        crop["proxy_for"] = proxy_for

    return crop


def constant_crop(tier, calendar_source, kc_source, constant_kc, h, proxy_for=None):
    crop = {
        "tier": tier,
        "calendar_source": calendar_source,
        "kc_source": kc_source,
        "constant_kc": constant_kc,
        "h": h,
    }

    if proxy_for is not None:
        crop["proxy_for"] = proxy_for

    return crop


def monthly_crop(tier, calendar_source, kc_source, monthly_kc, h, proxy_for=None):
    crop = {
        "tier": tier,
        "calendar_source": calendar_source,
        "kc_source": kc_source,
        "monthly_kc": monthly_kc,
        "h": h,
    }

    if proxy_for is not None:
        crop["proxy_for"] = proxy_for

    return crop


CROPS = {}

# =============================================================================
# STEP 5A — Tier 1 crops: Gabr calendar + FAO-56 Kc
# =============================================================================

CROPS["maize"] = annual_crop(
    tier=1,
    calendar_source="Gabr Egypt calendar",
    kc_source="FAO56 maize grain",
    planting_month=4,
    planting_day=20,
    harvest_month=8,
    harvest_day=22,
    Lini=30,
    Ldev=40,
    Lmid=50,
    Llate=30,
    Kc_ini=0.30,
    Kc_mid=1.20,
    Kc_end=0.60,
    h=2.0,
)

CROPS["rice"] = annual_crop(
    tier=1,
    calendar_source="Gabr Egypt calendar",
    kc_source="FAO56 rice",
    planting_month=5,
    planting_day=15,
    harvest_month=9,
    harvest_day=11,
    Lini=30,
    Ldev=30,
    Lmid=60,
    Llate=30,
    Kc_ini=1.05,
    Kc_mid=1.20,
    Kc_end=0.90,
    h=1.0,
)

CROPS["wheat"] = annual_crop(
    tier=1,
    calendar_source="Gabr Egypt calendar",
    kc_source="FAO56 winter wheat, non-frozen soils",
    planting_month=11,
    planting_day=1,
    harvest_month=3,
    harvest_day=10,
    Lini=15,
    Ldev=25,
    Lmid=50,
    Llate=30,
    Kc_ini=0.70,
    Kc_mid=1.15,
    Kc_end=0.25,
    h=1.0,
)

CROPS["potato"] = annual_crop(
    tier=1,
    calendar_source="Gabr Egypt calendar",
    kc_source="FAO56 potato, semi-arid climate",
    planting_month=9,
    planting_day=1,
    harvest_month=1,
    harvest_day=8,
    Lini=25,
    Ldev=30,
    Lmid=45,
    Llate=30,
    Kc_ini=0.50,
    Kc_mid=1.15,
    Kc_end=0.75,
    h=0.6,
)

CROPS["tomato_winter"] = annual_crop(
    tier=1,
    calendar_source="Gabr Egypt calendar",
    kc_source="FAO56 tomato, arid region",
    planting_month=11,
    planting_day=15,
    harvest_month=4,
    harvest_day=8,
    Lini=35,
    Ldev=45,
    Lmid=70,
    Llate=30,
    Kc_ini=0.60,
    Kc_mid=1.15,
    Kc_end=0.80,
    h=0.6,
)

CROPS["tomato_summer"] = annual_crop(
    tier=1,
    calendar_source="Gabr Egypt calendar",
    kc_source="FAO56 tomato, Mediterranean",
    planting_month=6,
    planting_day=6,
    harvest_month=10,
    harvest_day=28,
    Lini=30,
    Ldev=40,
    Lmid=45,
    Llate=30,
    Kc_ini=0.60,
    Kc_mid=1.15,
    Kc_end=0.80,
    h=0.6,
)

CROPS["sugarbeet"] = annual_crop(
    tier=1,
    calendar_source="Gabr Egypt calendar",
    kc_source="FAO56 sugar beet, arid region",
    planting_month=8,
    planting_day=1,
    harvest_month=1,
    harvest_day=7,
    Lini=35,
    Ldev=60,
    Lmid=70,
    Llate=40,
    Kc_ini=0.35,
    Kc_mid=1.20,
    Kc_end=0.70,
    h=0.5,
)

CROPS["sugarcane"] = annual_crop(
    tier=1,
    calendar_source="Gabr Egypt calendar",
    kc_source="FAO56 sugar cane, low latitudes",
    planting_month=9,
    planting_day=1,
    harvest_month=8,
    harvest_day=31,
    Lini=35,
    Ldev=60,
    Lmid=190,
    Llate=120,
    Kc_ini=0.40,
    Kc_mid=1.25,
    Kc_end=0.75,
    h=3.0,
)

CROPS["cotton"] = annual_crop(
    tier=1,
    calendar_source="Gabr Egypt calendar",
    kc_source="FAO56 cotton, Egypt/Pakistan/California",
    planting_month=5,
    planting_day=1,
    harvest_month=11,
    harvest_day=11,
    Lini=30,
    Ldev=50,
    Lmid=60,
    Llate=55,
    Kc_ini=0.35,
    Kc_mid=1.15,
    Kc_end=0.70,
    h=1.2,
)

CROPS["bean"] = annual_crop(
    tier=1,
    calendar_source="Gabr Egypt calendar",
    kc_source="FAO56 dry beans",
    planting_month=10,
    planting_day=1,
    harvest_month=1,
    harvest_day=18,
    Lini=20,
    Ldev=30,
    Lmid=40,
    Llate=20,
    Kc_ini=0.40,
    Kc_mid=1.15,
    Kc_end=0.35,
    h=0.4,
)

CROPS["clover"] = annual_crop(
    tier=1,
    calendar_source="Gabr Egypt calendar",
    kc_source="FAO56 berseem clover hay, averaged cutting effects",
    planting_month=10,
    planting_day=15,
    harvest_month=5,
    harvest_day=27,
    Lini=10,
    Ldev=30,
    Lmid=150,
    Llate=35,
    Kc_ini=0.40,
    Kc_mid=0.90,
    Kc_end=0.85,
    h=0.6,
)


# =============================================================================
# STEP 5B — Tier 2 crops: FAO-56 direct calendar + FAO-56 Kc
# =============================================================================

CROPS["sorghum"] = annual_crop(
    tier=2,
    calendar_source="FAO56 arid region",
    kc_source="FAO56 sorghum grain",
    planting_month=3,
    planting_day=15,
    Lini=20,
    Ldev=35,
    Lmid=45,
    Llate=30,
    Kc_ini=0.30,
    Kc_mid=1.05,
    Kc_end=0.55,
    h=1.5,
)

CROPS["orange"] = annual_crop(
    tier=2,
    calendar_source="FAO56 perennial full year",
    kc_source="FAO56 citrus, 70% canopy, no ground cover",
    planting_month=1,
    planting_day=1,
    harvest_month=12,
    harvest_day=31,
    Lini=60,
    Ldev=90,
    Lmid=120,
    Llate=95,
    Kc_ini=0.70,
    Kc_mid=0.65,
    Kc_end=0.70,
    h=4.0,
)

CROPS["onion"] = annual_crop(
    tier=2,
    calendar_source="FAO56 onion dry, arid region",
    kc_source="FAO56 onion dry",
    planting_month=10,
    planting_day=1,
    Lini=20,
    Ldev=35,
    Lmid=110,
    Llate=45,
    Kc_ini=0.70,
    Kc_mid=1.05,
    Kc_end=0.75,
    h=0.4,
)

CROPS["grape"] = annual_crop(
    tier=2,
    calendar_source="FAO56 grapes, low latitudes",
    kc_source="FAO56 grapes table/raisin",
    planting_month=4,
    planting_day=1,
    Lini=20,
    Ldev=40,
    Lmid=120,
    Llate=60,
    Kc_ini=0.30,
    Kc_mid=0.85,
    Kc_end=0.45,
    h=2.0,
)

CROPS["olive"] = monthly_crop(
    tier=2,
    calendar_source="FAO56 olive monthly coefficients, Spain analogue",
    kc_source="FAO56 olive, 40-60% canopy",
    monthly_kc={
        1: 0.50, 2: 0.50, 3: 0.65, 4: 0.60,
        5: 0.55, 6: 0.50, 7: 0.45, 8: 0.45,
        9: 0.55, 10: 0.60, 11: 0.65, 12: 0.50,
    },
    h=4.0,
)

CROPS["date"] = constant_crop(
    tier=2,
    calendar_source="FAO56 perennial full year",
    kc_source="FAO56 date palms",
    constant_kc=0.95,
    h=8.0,
)

CROPS["groundnut"] = annual_crop(
    tier=2,
    calendar_source="FAO56 Mediterranean",
    kc_source="FAO56 groundnut",
    planting_month=5,
    planting_day=1,
    Lini=35,
    Ldev=45,
    Lmid=35,
    Llate=25,
    Kc_ini=0.40,
    Kc_mid=1.15,
    Kc_end=0.60,
    h=0.4,
)

CROPS["eggplant"] = annual_crop(
    tier=2,
    calendar_source="FAO56 arid region",
    kc_source="FAO56 egg plant",
    planting_month=10,
    planting_day=1,
    Lini=30,
    Ldev=40,
    Lmid=40,
    Llate=20,
    Kc_ini=0.60,
    Kc_mid=1.05,
    Kc_end=0.90,
    h=0.8,
)

CROPS["watermelon"] = annual_crop(
    tier=2,
    calendar_source="FAO56 Mediterranean",
    kc_source="FAO56 watermelon",
    planting_month=4,
    planting_day=1,
    Lini=20,
    Ldev=30,
    Lmid=30,
    Llate=30,
    Kc_ini=0.40,
    Kc_mid=1.00,
    Kc_end=0.75,
    h=0.4,
)

CROPS["barley"] = annual_crop(
    tier=2,
    calendar_source="FAO56 small grains",
    kc_source="FAO56 barley",
    planting_month=11,
    planting_day=1,
    Lini=15,
    Ldev=25,
    Lmid=50,
    Llate=30,
    Kc_ini=0.30,
    Kc_mid=1.15,
    Kc_end=0.25,
    h=1.0,
)

CROPS["sesame"] = annual_crop(
    tier=2,
    calendar_source="FAO56 sesame",
    kc_source="FAO56 sesame",
    planting_month=6,
    planting_day=1,
    Lini=20,
    Ldev=30,
    Lmid=40,
    Llate=20,
    Kc_ini=0.35,
    Kc_mid=1.10,
    Kc_end=0.25,
    h=1.0,
)

CROPS["banana"] = annual_crop(
    tier=2,
    calendar_source="FAO56 banana 2nd year, Mediterranean",
    kc_source="FAO56 banana 2nd year",
    planting_month=2,
    planting_day=1,
    harvest_month=1,
    harvest_day=31,
    Lini=120,
    Ldev=60,
    Lmid=180,
    Llate=5,
    Kc_ini=1.00,
    Kc_mid=1.20,
    Kc_end=1.10,
    h=4.0,
)

CROPS["cucumberetc"] = annual_crop(
    tier=2,
    calendar_source="FAO56 cucumber arid region",
    kc_source="FAO56 cucumber fresh market",
    planting_month=6,
    planting_day=1,
    Lini=20,
    Ldev=30,
    Lmid=40,
    Llate=15,
    Kc_ini=0.60,
    Kc_mid=1.00,
    Kc_end=0.75,
    h=0.3,
)

CROPS["pumpkinetc"] = annual_crop(
    tier=2,
    calendar_source="FAO56 pumpkin/winter squash Mediterranean",
    kc_source="FAO56 pumpkin/winter squash",
    planting_month=3,
    planting_day=1,
    Lini=20,
    Ldev=30,
    Lmid=30,
    Llate=20,
    Kc_ini=0.50,
    Kc_mid=1.00,
    Kc_end=0.80,
    h=0.4,
)

CROPS["cabbage"] = annual_crop(
    tier=2,
    calendar_source="FAO56 cabbage Mediterranean/desert",
    kc_source="FAO56 cabbage",
    planting_month=9,
    planting_day=1,
    Lini=35,
    Ldev=45,
    Lmid=40,
    Llate=15,
    Kc_ini=0.70,
    Kc_mid=1.05,
    Kc_end=0.95,
    h=0.4,
)

CROPS["alfalfa"] = annual_crop(
    tier=2,
    calendar_source="FAO56 forage full-year proxy",
    kc_source="FAO56 alfalfa hay, averaged cutting effects",
    planting_month=1,
    planting_day=1,
    harvest_month=12,
    harvest_day=31,
    Lini=10,
    Ldev=30,
    Lmid=290,
    Llate=35,
    Kc_ini=0.40,
    Kc_mid=0.95,
    Kc_end=0.90,
    h=0.7,
)

CROPS["garlic"] = annual_crop(
    tier=2,
    calendar_source="FAO56 onion/garlic winter proxy",
    kc_source="FAO56 garlic",
    planting_month=10,
    planting_day=1,
    Lini=20,
    Ldev=35,
    Lmid=110,
    Llate=45,
    Kc_ini=0.70,
    Kc_mid=1.00,
    Kc_end=0.70,
    h=0.3,
)

CROPS["sweetpotato"] = annual_crop(
    tier=2,
    calendar_source="FAO56 sweet potato Mediterranean",
    kc_source="FAO56 sweet potato",
    planting_month=4,
    planting_day=1,
    Lini=20,
    Ldev=30,
    Lmid=60,
    Llate=40,
    Kc_ini=0.50,
    Kc_mid=1.15,
    Kc_end=0.65,
    h=0.4,
)

CROPS["soybean"] = annual_crop(
    tier=2,
    calendar_source="FAO56 soybean",
    kc_source="FAO56 soybean",
    planting_month=5,
    planting_day=1,
    Lini=20,
    Ldev=35,
    Lmid=60,
    Llate=25,
    Kc_ini=0.40,
    Kc_mid=1.15,
    Kc_end=0.50,
    h=0.8,
)

CROPS["sunflower"] = annual_crop(
    tier=2,
    calendar_source="FAO56 sunflower Mediterranean",
    kc_source="FAO56 sunflower",
    planting_month=4,
    planting_day=15,
    Lini=25,
    Ldev=35,
    Lmid=45,
    Llate=25,
    Kc_ini=0.35,
    Kc_mid=1.10,
    Kc_end=0.35,
    h=2.0,
)

CROPS["melonetc"] = annual_crop(
    tier=2,
    calendar_source="FAO56 sweet melons arid region",
    kc_source="FAO56 sweet melons",
    planting_month=12,
    planting_day=15,
    Lini=30,
    Ldev=45,
    Lmid=65,
    Llate=20,
    Kc_ini=0.50,
    Kc_mid=1.05,
    Kc_end=0.75,
    h=0.4,
)

CROPS["flax"] = annual_crop(
    tier=2,
    calendar_source="FAO56 flax",
    kc_source="FAO56 flax",
    planting_month=10,
    planting_day=1,
    Lini=30,
    Ldev=40,
    Lmid=100,
    Llate=50,
    Kc_ini=0.35,
    Kc_mid=1.10,
    Kc_end=0.25,
    h=1.2,
)

CROPS["carrot"] = annual_crop(
    tier=2,
    calendar_source="FAO56 carrot arid region",
    kc_source="FAO56 carrots",
    planting_month=10,
    planting_day=1,
    Lini=20,
    Ldev=30,
    Lmid=50,
    Llate=20,
    Kc_ini=0.70,
    Kc_mid=1.05,
    Kc_end=0.95,
    h=0.3,
)

CROPS["lettuce"] = annual_crop(
    tier=2,
    calendar_source="FAO56 lettuce Mediterranean",
    kc_source="FAO56 lettuce",
    planting_month=11,
    planting_day=1,
    Lini=30,
    Ldev=40,
    Lmid=25,
    Llate=10,
    Kc_ini=0.70,
    Kc_mid=1.00,
    Kc_end=0.95,
    h=0.3,
)

CROPS["cauliflower"] = annual_crop(
    tier=2,
    calendar_source="FAO56 cauliflower desert",
    kc_source="FAO56 cauliflower",
    planting_month=9,
    planting_day=1,
    Lini=35,
    Ldev=50,
    Lmid=40,
    Llate=15,
    Kc_ini=0.70,
    Kc_mid=1.05,
    Kc_end=0.95,
    h=0.4,
)


# =============================================================================
# STEP 5C — Tier 3 crops: annual proxy Kc by crop group
# =============================================================================

TIER3_PROXY_KC = {
    "orchard": 0.70,
    "citrus": 0.68,
    "annual_vegetable": 0.85,
    "root_tuber": 0.80,
    "pulse": 0.65,
    "forage": 0.85,
    "oilseed_spice": 0.70,
    "cereal_proxy": 0.75,
    "other": 0.70,
}


TIER3_ASSIGNMENT = {
    # Fodder / forage
    "vegfor": "forage",
    "mixedgrass": "forage",
    "vetch": "forage",

    # Fruit trees / orchards
    "mango": "orchard",
    "fruitnes": "orchard",
    "fig": "orchard",
    "apple": "orchard",
    "peachetc": "orchard",
    "apricot": "orchard",
    "pear": "orchard",
    "walnut": "orchard",
    "plum": "orchard",
    "almond": "orchard",
    "stonefruitnes": "orchard",
    "tropicalnes": "orchard",
    "nutnes": "orchard",
    "avocado": "orchard",
    "carob": "orchard",
    "cherry": "orchard",
    "persimmon": "orchard",

    # Citrus residuals
    "tangetc": "citrus",
    "lemonlime": "citrus",
    "citrusnes": "citrus",
    "grapefruitetc": "citrus",

    # Vegetables / aggregated vegetables
    "vegetablenes": "annual_vegetable",
    "greenpea": "annual_vegetable",
    "greenbean": "annual_vegetable",
    "greenbroadbean": "annual_vegetable",
    "greenonion": "annual_vegetable",
    "okra": "annual_vegetable",
    "pimento": "annual_vegetable",
    "chilleetc": "annual_vegetable",

    # Roots / tubers
    "rootnes": "root_tuber",
    "taro": "root_tuber",

    # Pulses
    "pulsenes": "pulse",
    "broadbean": "pulse",
    "cowpea": "pulse",
    "lupin": "pulse",
    "pea": "pulse",

    # Oilseeds / spices / minor industrial crops
    "aniseetc": "oilseed_spice",
    "linseed": "oilseed_spice",
    "oilseednes": "oilseed_spice",
    "poppy": "oilseed_spice",

    # Cereal-like residuals
    "rye": "cereal_proxy",
    "oats": "cereal_proxy",
    "greencorn": "cereal_proxy",

    # Other minor crops
    "jute": "other",
    "berr yn es": "other",
    "berrynes": "other",
    "spicenes": "other",
}

for crop_name, proxy_group in TIER3_ASSIGNMENT.items():
    if crop_name in CROPS:
        continue

    CROPS[crop_name] = constant_crop(
        tier=3,
        calendar_source="Annual proxy",
        kc_source=f"Tier 3 annual mean Kc proxy: {proxy_group}",
        constant_kc=TIER3_PROXY_KC[proxy_group],
        h=1.0,
        proxy_for=crop_name,
    )

# =============================================================================
# Composite crops
# =============================================================================

COMPOSITE_CROPS = {
    "tomato": {
        "tomato_winter": 0.50,
        "tomato_summer": 0.50,
    }
}


# =============================================================================
# STEP 6 — Kc curve helpers
# =============================================================================

def safe_timestamp(year, month, day):
    last_day = calendar.monthrange(year, month)[1]
    return pd.Timestamp(year, month, min(day, last_day))


def season_bounds_for_date(date, crop):
    if "planting_month" not in crop:
        return None

    start = safe_timestamp(
        date.year,
        crop["planting_month"],
        crop["planting_day"],
    )

    if "harvest_month" in crop and "harvest_day" in crop:
        end = safe_timestamp(
            date.year,
            crop["harvest_month"],
            crop["harvest_day"],
        )

        if end < start:
            end = safe_timestamp(
                date.year + 1,
                crop["harvest_month"],
                crop["harvest_day"],
            )
    else:
        raw_total = crop["Lini"] + crop["Ldev"] + crop["Lmid"] + crop["Llate"]
        end = start + pd.Timedelta(days=int(round(raw_total)) - 1)

    if date < start:
        start = safe_timestamp(
            date.year - 1,
            crop["planting_month"],
            crop["planting_day"],
        )

        if "harvest_month" in crop and "harvest_day" in crop:
            end = safe_timestamp(
                date.year - 1,
                crop["harvest_month"],
                crop["harvest_day"],
            )
            if end < start:
                end = safe_timestamp(
                    date.year,
                    crop["harvest_month"],
                    crop["harvest_day"],
                )
        else:
            raw_total = crop["Lini"] + crop["Ldev"] + crop["Lmid"] + crop["Llate"]
            end = start + pd.Timedelta(days=int(round(raw_total)) - 1)

    if start <= date <= end:
        return start, end

    return None


def stage_lengths_for_season(crop, start, end):
    raw = np.array(
        [crop["Lini"], crop["Ldev"], crop["Lmid"], crop["Llate"]],
        dtype=float,
    )

    if (
        RESCALE_STAGES_TO_CALENDAR
        and "harvest_month" in crop
        and "harvest_day" in crop
    ):
        target_total = (end - start).days + 1
        return raw * (target_total / raw.sum())

    return raw


def kc_terms_for_date(date, crop):
    date = pd.Timestamp(date)

    if "constant_kc" in crop:
        return crop["constant_kc"], 0.0, 0.0

    if "monthly_kc" in crop:
        return crop["monthly_kc"][date.month], 0.0, 0.0

    bounds = season_bounds_for_date(date, crop)

    if bounds is None:
        return 0.0, 0.0, 0.0

    start, end = bounds
    Lini, Ldev, Lmid, Llate = stage_lengths_for_season(crop, start, end)

    d = (date - start).days + 1

    if d <= Lini:
        return crop["Kc_ini"], 0.0, 0.0

    if d <= Lini + Ldev:
        frac = (d - Lini) / Ldev
        a = crop["Kc_ini"] * (1 - frac)
        return a, frac, 0.0

    if d <= Lini + Ldev + Lmid:
        return 0.0, 1.0, 0.0

    if d <= Lini + Ldev + Lmid + Llate:
        frac = (d - Lini - Ldev - Lmid) / Llate
        return 0.0, 1 - frac, frac

    return 0.0, 0.0, 0.0


def build_monthly_kc(crop_name, crop, ETo, u2_for_kc, RHmin):
    kc_slices = []

    for i, t in enumerate(ETo[TIME_DIM].values):
        template = ETo.isel({TIME_DIM: i})

        month_start = pd.Timestamp(t).to_period("M").to_timestamp()
        month_end = month_start + pd.offsets.MonthEnd(0)
        daily_dates = pd.date_range(month_start, month_end, freq="D")

        terms = np.array([kc_terms_for_date(d, crop) for d in daily_dates])
        a_mean, b_mid_mean, b_end_mean = terms.mean(axis=0)

        if "Kc_mid" in crop:
            u2_t = u2_for_kc.isel({TIME_DIM: i})
            RHmin_t = RHmin.isel({TIME_DIM: i})

            kc_mid_adj = adjust_kc_for_climate(
                crop["Kc_mid"],
                crop["h"],
                template,
                u2_stage=u2_t,
                RHmin_stage=RHmin_t,
            )

            if crop["Kc_end"] > 0.45:
                kc_end_adj = adjust_kc_for_climate(
                    crop["Kc_end"],
                    crop["h"],
                    template,
                    u2_stage=u2_t,
                    RHmin_stage=RHmin_t,
                )
            else:
                kc_end_adj = xr.zeros_like(template) + crop["Kc_end"]

            kc_t = a_mean + b_mid_mean * kc_mid_adj + b_end_mean * kc_end_adj

        else:
            kc_t = xr.zeros_like(template) + a_mean

        kc_t = kc_t.clip(min=0)
        kc_slices.append(kc_t)

    Kc = xr.concat(kc_slices, dim=TIME_DIM)
    Kc = Kc.assign_coords({TIME_DIM: ETo[TIME_DIM]})
    Kc.name = f"Kc_{crop_name}"
    Kc.attrs["units"] = "-"
    Kc.attrs["long_name"] = f"FAO-56 single crop coefficient for {crop_name}"
    Kc.attrs["tier"] = crop.get("tier", -1)
    Kc.attrs["calendar_source"] = crop.get("calendar_source", "")
    Kc.attrs["kc_source"] = crop.get("kc_source", "")

    return Kc


# =============================================================================
# STEP 7 — Compute Kc and ETc
# =============================================================================

Kc_all = {}
ETc_all = {}

for crop_name, crop in CROPS.items():
    print(f"Calculating Kc and ETc for: {crop_name}")

    Kc = build_monthly_kc(crop_name, crop, ETo, u2_for_kc, RHmin)

    ETc_mm_day = Kc * ETo
    ETc_mm_day.name = f"ETc_mm_day_{crop_name}"
    ETc_mm_day.attrs["units"] = "mm/day"

    ETc_mm_month = ETc_mm_day * days_in_month
    ETc_mm_month.name = f"ETc_mm_month_{crop_name}"
    ETc_mm_month.attrs["units"] = "mm/month"

    Kc_all[crop_name] = Kc
    ETc_all[crop_name] = {
        "ETc_mm_day": ETc_mm_day,
        "ETc_mm_month": ETc_mm_month,
    }


for crop_name, weights in COMPOSITE_CROPS.items():
    print(f"Calculating composite crop: {crop_name}")

    Kc = None

    for profile_name, weight in weights.items():
        if Kc is None:
            Kc = weight * Kc_all[profile_name]
        else:
            Kc = Kc + weight * Kc_all[profile_name]

    Kc.name = f"Kc_{crop_name}"
    Kc.attrs["units"] = "-"
    Kc.attrs["long_name"] = f"Composite FAO-56 Kc for {crop_name}"
    Kc.attrs["tier"] = "composite"
    Kc.attrs["calendar_source"] = "Weighted seasonal composite"
    Kc.attrs["kc_source"] = "Weighted crop profiles"

    ETc_mm_day = Kc * ETo
    ETc_mm_day.name = f"ETc_mm_day_{crop_name}"
    ETc_mm_day.attrs["units"] = "mm/day"

    ETc_mm_month = ETc_mm_day * days_in_month
    ETc_mm_month.name = f"ETc_mm_month_{crop_name}"
    ETc_mm_month.attrs["units"] = "mm/month"

    Kc_all[crop_name] = Kc
    ETc_all[crop_name] = {
        "ETc_mm_day": ETc_mm_day,
        "ETc_mm_month": ETc_mm_month,
    }


# =============================================================================
# STEP 8 — Sanity checks
# =============================================================================

print("\n=== ETo SANITY CHECK ===")
print(f"ETo   min: {float(ETo.min()):.2f}  max: {float(ETo.max()):.2f}  mean: {float(ETo.mean()):.2f} mm/day")
print(f"Rn    min: {float(Rn.min()):.2f}   max: {float(Rn.max()):.2f}   mean: {float(Rn.mean()):.2f} MJ/m²/day")
print(f"es    min: {float(es.min()):.2f}   max: {float(es.max()):.2f}   mean: {float(es.mean()):.2f} kPa")
print(f"ea    min: {float(ea.min()):.2f}   max: {float(ea.max()):.2f}   mean: {float(ea.mean()):.2f} kPa")
print(f"vpd   min: {float(vpd.min()):.2f}  max: {float(vpd.max()):.2f}  mean: {float(vpd.mean()):.2f} kPa")
print(f"u2    min: {float(u2.min()):.2f}   max: {float(u2.max()):.2f}   mean: {float(u2.mean()):.2f} m/s")
print(f"gamma min: {float(gamma.min()):.4f} max: {float(gamma.max()):.4f} mean: {float(gamma.mean()):.4f} kPa/°C")

print("\n=== CROP SANITY CHECK ===")
for crop_name in ["maize", "rice", "wheat", "sorghum", "mango", "tomato"]:
    if crop_name in Kc_all:
        print(
            f"{crop_name:15s} "
            f"Kc mean: {float(Kc_all[crop_name].mean()):.2f} "
            f"Kc max: {float(Kc_all[crop_name].max()):.2f} "
            f"ETc mean: {float(ETc_all[crop_name]['ETc_mm_day'].mean()):.2f} mm/day"
        )


# =============================================================================
# STEP 9 — Save NetCDF outputs
# =============================================================================

if SAVE_NETCDF:
    ETo.to_netcdf(OUTPUT_PATH / "ETo_egypt_monthly.nc")
    print("\nSaved: ETo_egypt_monthly.nc")

    crop_nc_dir = OUTPUT_PATH / "crop_netcdf"
    crop_nc_dir.mkdir(parents=True, exist_ok=True)

    for crop_name in ETc_all:
        ds_crop = xr.Dataset(
            {
                "Kc": Kc_all[crop_name],
                "ETc_mm_day": ETc_all[crop_name]["ETc_mm_day"],
                "ETc_mm_month": ETc_all[crop_name]["ETc_mm_month"],
            }
        )

        ds_crop.to_netcdf(crop_nc_dir / f"ETc_{crop_name}_egypt_monthly.nc")
        print(f"Saved: ETc_{crop_name}_egypt_monthly.nc")


# =============================================================================
# STEP 10 — Save GeoTIFF outputs
# =============================================================================

def prepare_for_geotiff(da, target_resolution=None):
    da = da.rio.set_spatial_dims(x_dim=LON_DIM, y_dim=LAT_DIM)
    da = da.rio.write_crs("EPSG:4326")

    if da[LAT_DIM][0] < da[LAT_DIM][-1]:
        da = da.sortby(LAT_DIM, ascending=False)

    if target_resolution is not None:
        da = da.rio.reproject(
            "EPSG:4326",
            resolution=target_resolution,
            resampling=Resampling.bilinear,
        )

    return da


if SAVE_GEOTIFFS:
    geotiff_dir = OUTPUT_PATH / "crop_geotiffs"
    geotiff_dir.mkdir(parents=True, exist_ok=True)

    for crop_name in ETc_all:
        crop_dir = geotiff_dir / crop_name
        crop_dir.mkdir(parents=True, exist_ok=True)

        ETc_annual_depth = (
            ETc_all[crop_name]["ETc_mm_month"]
            .groupby(f"{TIME_DIM}.year")
            .sum(dim=TIME_DIM)
            .mean(dim="year")
        )

        ETc_annual_depth = prepare_for_geotiff(ETc_annual_depth, target_resolution=TARGET_RESOLUTION)
        ETc_annual_depth.rio.to_raster(
            crop_dir / f"ETc_annual_depth_{crop_name}_egypt.tif"
        )

        ETc_monthly_clim = (
            ETc_all[crop_name]["ETc_mm_month"]
            .groupby(f"{TIME_DIM}.month")
            .mean(dim=TIME_DIM)
        )

        for month in range(1, 13):
            ETc_m = ETc_monthly_clim.sel(month=month)
            ETc_m = prepare_for_geotiff(ETc_m, target_resolution=TARGET_RESOLUTION)

            ETc_m.rio.to_raster(
                crop_dir / f"ETc_monthly_depth_{crop_name}_{month:02d}_egypt.tif"
            )

        print(f"Saved GeoTIFFs for {crop_name}")


# =============================================================================
# STEP 11 — Save summary tables
# =============================================================================

summary_rows = []

for crop_name in ETc_all:
    Kc = Kc_all[crop_name]
    ETc_day = ETc_all[crop_name]["ETc_mm_day"]
    ETc_month = ETc_all[crop_name]["ETc_mm_month"]

    monthly_clim = ETc_month.groupby(f"{TIME_DIM}.month").mean(dim=TIME_DIM)
    annual_depth = monthly_clim.sum(dim="month")

    crop = CROPS.get(crop_name, {})
    if crop_name in COMPOSITE_CROPS:
        tier = "composite"
        calendar_source = "Weighted seasonal composite"
        kc_source = "Weighted crop profiles"
    else:
        tier = crop.get("tier", "")
        calendar_source = crop.get("calendar_source", "")
        kc_source = crop.get("kc_source", "")

    summary_rows.append(
        {
            "crop": crop_name,
            "tier": tier,
            "calendar_source": calendar_source,
            "kc_source": kc_source,
            "Kc_mean_all_months": float(Kc.mean()),
            "Kc_max": float(Kc.max()),
            "ETc_mean_mm_day": float(ETc_day.mean()),
            "ETc_max_mm_day": float(ETc_day.max()),
            "ETc_mean_mm_month": float(ETc_month.mean()),
            "ETc_max_mm_month": float(ETc_month.max()),
            "ETc_annual_depth_mean_mm": float(annual_depth.mean()),
            "ETc_annual_depth_max_mm": float(annual_depth.max()),
        }
    )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_PATH / "ETc_crop_summary.csv", index=False)
print("\nSaved: ETc_crop_summary.csv")

metadata_rows = []

for crop_name, crop in CROPS.items():
    row = {
        "crop": crop_name,
        "tier": crop.get("tier"),
        "calendar_source": crop.get("calendar_source"),
        "kc_source": crop.get("kc_source"),
        "proxy_for": crop.get("proxy_for", ""),
        "h": crop.get("h", ""),
        "constant_kc": crop.get("constant_kc", ""),
        "Kc_ini": crop.get("Kc_ini", ""),
        "Kc_mid": crop.get("Kc_mid", ""),
        "Kc_end": crop.get("Kc_end", ""),
        "Lini": crop.get("Lini", ""),
        "Ldev": crop.get("Ldev", ""),
        "Lmid": crop.get("Lmid", ""),
        "Llate": crop.get("Llate", ""),
        "planting_month": crop.get("planting_month", ""),
        "planting_day": crop.get("planting_day", ""),
        "harvest_month": crop.get("harvest_month", ""),
        "harvest_day": crop.get("harvest_day", ""),
    }

    metadata_rows.append(row)

metadata_df = pd.DataFrame(metadata_rows)
metadata_df.to_csv(OUTPUT_PATH / "crop_kc_metadata.csv", index=False)
print("Saved: crop_kc_metadata.csv")

print("\nAll done.")