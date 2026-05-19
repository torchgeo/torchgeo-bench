"""Generate the OlmoEarth-specific GeoBench sweep job list.

OlmoEarth was pretrained on 12-channel Sentinel-2 (no B10/cirrus) at native
resolution.  Datasets vary in what's available:

* 12 S2 bands ready -> use them in OlmoEarth's expected order.
* Fewer S2 bands or non-S2 sensor -> RGB-only zero-fill fallback.

The wrapper auto-detects the input scale (DN vs reflectance vs uint8) and
rescales to S2 DN before calling OlmoEarth's internal Normalizer, so we
don't need to set ``dataset.normalization`` per task.
"""

from pathlib import Path

# Per-dataset S2 band list, ordered to match OlmoEarth's
# (B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09).
TWELVE_BAND_S2: dict[str, list[str]] = {
    # GeoBench V1 (semantic lowercase names)
    "m-eurosat": [
        "blue",
        "green",
        "red",
        "nir",
        "red_edge_1",
        "red_edge_2",
        "red_edge_3",
        "red_edge_4",
        "swir_1",
        "swir_2",
        "coastal_aerosol",
        "water_vapour",
    ],
    "m-bigearthnet": [
        "blue",
        "green",
        "red",
        "nir",
        "red_edge_1",
        "red_edge_2",
        "red_edge_3",
        "red_edge_4",
        "swir_1",
        "swir_2",
        "coastal_aerosol",
        "water_vapour",
    ],
    "m-brick-kiln": [
        "blue",
        "green",
        "red",
        "nir",
        "red_edge_1",
        "red_edge_2",
        "red_edge_3",
        "red_edge_4",
        "swir_1",
        "swir_2",
        "coastal_aerosol",
        "water_vapour",
    ],
    # GeoBench V2 (b02-style names)
    "benv2": ["b02", "b03", "b04", "b08", "b05", "b06", "b07", "b8a", "b11", "b12", "b01", "b09"],
    "treesatai": [
        "b02",
        "b03",
        "b04",
        "b08",
        "b05",
        "b06",
        "b07",
        "b8a",
        "b11",
        "b12",
        "b01",
        "b09",
    ],
}

# 10-band S2 mode (no B01/B09) for datasets missing those bands.  The wrapper
# zero-fills positions 10 (B01) and 11 (B09) internally.  Caller passes
# bands in OlmoEarth's first-10 order.
TEN_BAND_S2: dict[str, list[str]] = {
    "m-so2sat": [
        "blue",
        "green",
        "red",
        "nir",
        "red_edge_1",
        "red_edge_2",
        "red_edge_3",
        "red_edge_4",
        "swir_1",
        "swir_2",
    ],
    "so2sat": ["b02", "b03", "b04", "b08", "b05", "b06", "b07", "b8a", "b11", "b12"],
}

# RGB-only fallback for non-S2 sensors and S2 datasets with <10 bands.
RGB_ONLY = [
    "m-forestnet",  # Landsat (6 bands, uint8)
    "m-pv4ger",  # NAIP aerial RGB
    "forestnet",  # S2 6 bands (V2), uint8 — fewer than 10 in OlmoEarth's order
]

MODELS = ["olmoearth_nano", "olmoearth_tiny", "olmoearth_base", "olmoearth_large"]


def main() -> None:
    lines: list[str] = []
    for model in MODELS:
        for ds, band_list in TWELVE_BAND_S2.items():
            lines.append(f"{model} {ds} {','.join(band_list)} null")
        for ds, band_list in TEN_BAND_S2.items():
            lines.append(f"{model} {ds} {','.join(band_list)} null")
        for ds in RGB_ONLY:
            lines.append(f"{model} {ds} rgb null")
    out = Path("scripts/slurm/olmoearth_sweep.jobs")
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} jobs to {out}")
    print(f"  models: {len(MODELS)} · datasets: {len(TWELVE_BAND_S2) + len(RGB_ONLY)}")


if __name__ == "__main__":
    main()
