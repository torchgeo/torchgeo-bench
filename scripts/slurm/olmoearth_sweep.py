"""Generate the OlmoEarth-specific GeoBench sweep job list.

OlmoEarth was pretrained on 12-channel Sentinel-2, Sentinel-1 (2 SAR), and
Landsat-8 (11 channels) at native resolution.  Datasets vary in what's
available:

* 12 S2 bands ready -> use them in OlmoEarth's expected order.
* Mixed S2 + SAR (m-so2sat) -> pass all bands; wrapper routes each sensor.
* Landsat (m-forestnet) -> pass all 6 Landsat bands; wrapper maps to 11-ch.
* Fewer S2 bands or unsupported sensor -> RGB-only zero-fill fallback.

The wrapper auto-detects the input scale (DN vs reflectance vs uint8) per
sensor group and rescales accordingly, so dataset.normalization isn't needed.
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

# 10-band S2 (no B01/B09).  Wrapper zero-fills positions 10–11 internally.
TEN_BAND_S2: dict[str, list[str]] = {
    "so2sat": ["b02", "b03", "b04", "b08", "b05", "b06", "b07", "b8a", "b11", "b12"],
}

# Mixed S2 + SAR datasets.  Pass all bands; the wrapper routes S2 bands to
# SENTINEL2_L2A and SAR bands to SENTINEL1, populating both fields of
# MaskedOlmoEarthSample simultaneously.
MIXED_S2_SAR: dict[str, list[str]] = {
    # m-so2sat: 10 S2 + 8 SAR bands (interleaved in dataset order).
    # Wrapper groups by sensor automatically.
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
        "vh_real",
        "vh_imag",
        "vv_real",
        "vv_imag",
        "vh_lee",
        "vv_lee",
        "vh_lee_real",
        "vv_lee_imag",
    ],
}

# Landsat datasets.  Wrapper uses LANDSAT modality (11-ch layout, input_res=30).
LANDSAT_BANDS: dict[str, list[str]] = {
    # m-forestnet: Landsat-8, 6 bands (blue/green/red/nir/swir_1/swir_2), uint8.
    "m-forestnet": ["blue", "green", "red", "nir", "swir_1", "swir_2"],
}

# RGB-only fallback for NAIP/aerial and S2 datasets with <10 usable bands.
RGB_ONLY = [
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
        for ds, band_list in MIXED_S2_SAR.items():
            lines.append(f"{model} {ds} {','.join(band_list)} null")
        for ds, band_list in LANDSAT_BANDS.items():
            lines.append(f"{model} {ds} {','.join(band_list)} null")
        for ds in RGB_ONLY:
            lines.append(f"{model} {ds} rgb null")
    total_ds = (
        len(TWELVE_BAND_S2)
        + len(TEN_BAND_S2)
        + len(MIXED_S2_SAR)
        + len(LANDSAT_BANDS)
        + len(RGB_ONLY)
    )
    out = Path("scripts/slurm/olmoearth_sweep.jobs")
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} jobs to {out}")
    print(f"  models: {len(MODELS)} · datasets: {total_ds}")


if __name__ == "__main__":
    main()
