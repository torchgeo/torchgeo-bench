"""Heavy runtime for dataset downloads."""

from torchgeo_bench.download import (
    download_eurosat,
    download_geobench_v1,
    download_geobench_v2,
    download_resisc45,
)

__all__ = ("download_eurosat", "download_geobench_v1", "download_geobench_v2", "download_resisc45")
