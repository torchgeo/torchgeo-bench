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
    "m-eurosat":     ["blue","green","red","nir","red_edge_1","red_edge_2","red_edge_3","red_edge_4","swir_1","swir_2","coastal_aerosol","water_vapour"],
    "m-bigearthnet": ["blue","green","red","nir","red_edge_1","red_edge_2","red_edge_3","red_edge_4","swir_1","swir_2","coastal_aerosol","water_vapour"],
    "m-brick-kiln":  ["blue","green","red","nir","red_edge_1","red_edge_2","red_edge_3","red_edge_4","swir_1","swir_2","coastal_aerosol","water_vapour"],
    # GeoBench V2 (b02-style names)
    "benv2":      ["b02","b03","b04","b08","b05","b06","b07","b8a","b11","b12","b01","b09"],
    "treesatai":  ["b02","b03","b04","b08","b05","b06","b07","b8a","b11","b12","b01","b09"],
}

# RGB-only fallback for the rest.  m-so2sat/so2sat have 10 S2 bands but
# lack B01/B09 — RGB keeps the comparison clean; a 10-band-with-zero-fill
# mode is a separate follow-up.
RGB_ONLY = [
    "m-forestnet",   # Landsat (6 bands, uint8)
    "m-so2sat",      # S2 10 bands, no B01/B09; reflectance-scaled
    "m-pv4ger",      # NAIP aerial RGB
    "forestnet",     # S2 6 bands (V2), uint8
    "so2sat",        # S2 10 bands; reflectance-scaled
]

MODELS = ["olmoearth_nano", "olmoearth_tiny", "olmoearth_base", "olmoearth_large"]


def main() -> None:
    lines: list[str] = []
    for model in MODELS:
        for ds, band_list in TWELVE_BAND_S2.items():
            lines.append(f"{model} {ds} {','.join(band_list)} null")
        for ds in RGB_ONLY:
            lines.append(f"{model} {ds} rgb null")
    out = Path("scripts/slurm/olmoearth_sweep.jobs")
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} jobs to {out}")
    print(f"  models: {len(MODELS)} · datasets: {len(TWELVE_BAND_S2) + len(RGB_ONLY)}")


if __name__ == "__main__":
    main()
