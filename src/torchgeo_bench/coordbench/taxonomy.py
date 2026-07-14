"""Semantic task families for aggregating CoordBench results.

Location encoders tend to specialize: a model strong on climate/elevation may be
weak on socioeconomic targets and vice versa. Averaging one number over the whole
suite hides that, so benchmarks are grouped into families and reported per family.

Three families, keyed by benchmark name:

* ``socio`` — socioeconomic / health / human-activity targets (income, population,
  DHS indices, CDC health prevalences, nightlights, roads).
* ``env`` — physical-environment targets (climate, elevation, soil, biomass,
  evapotranspiration, tree cover).
* ``land`` — land-cover / land-use / crop-type and biome/ecoregion/country
  classification.

Regression scores are floored at :data:`R2_FLOOR` before averaging: R² values far
below zero are numerically unstable, not informative.
"""

# Socioeconomic / health / human-activity.
SOCIO = (
    {
        "pdfm-conus27",
        "california-housing",
        "mosaiks-population",
        "mosaiks-nightlights",
        "mosaiks-income",
        "mosaiks-housing",
        "mosaiks-roads",
        "satclip-population",
        "bt-population",
        "bt-distroad",
    }
    | {
        f"sustainbench-{k}"
        for k in ("asset", "water", "sanitation", "child_mortality", "women_edu", "women_bmi")
    }
    | {
        f"places-{k}"
        for k in (
            "asthma",
            "cancer",
            "chd",
            "checkup",
            "copd",
            "diabetes",
            "high_chol",
            "mental_health",
            "obesity",
            "phys_health",
            "sleep_lt7",
            "smoking",
        )
    }
)

# Physical environment: climate, elevation, soil, biomass, tree cover.
ENV = {
    "worldclim-bio1",
    "worldclim-bio12",
    "soilgrids-soc",
    "soilgrids-phh2o",
    "dm-aster_ged",
    "dm-openet_ensemble",
    "mosaiks-elevation",
    "mosaiks-treecover",
    "satclip-elevation",
    "satclip-air-temp",
    "bt-biomass",
    "bt-bioclim",
}

# Land cover / land use / crop type + biome/ecoregion/country classification.
LAND = {
    f"dm-{k}"
    for k in (
        "africa_crop_mask",
        "canada_crops_coarse",
        "canada_crops_fine",
        "descals",
        "ethiopia_crops",
        "glance",
        "lcmap_lc",
        "lcmap_lcc",
        "lcmap_lu",
        "lcmap_luc",
        "lucas_lc",
        "lucas_lu",
        "us_trees",
    )
} | {
    "satclip-biome",
    "satclip-country",
    "satclip-ecoregion",
    "country",
    "ecoregions",
    "bt-cropharvest",
    "bt-landcover",
}

FAMILIES = ("socio", "env", "land")

# Floor per-benchmark R^2 before averaging; deep-negative values are unstable noise.
R2_FLOOR = -1.0


def family_of(name: str) -> str:
    """Return the family (``socio`` / ``env`` / ``land``) for a benchmark name.

    Unknown names return ``"other"`` so they surface rather than silently joining a
    family.
    """
    if name in SOCIO:
        return "socio"
    if name in ENV:
        return "env"
    if name in LAND:
        return "land"
    return "other"
