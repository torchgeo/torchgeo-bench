"""Repack a GeoBench V1 classification dataset from per-sample HDF5 files into
WebDataset tar shards.

V1 ships ~22k tiny ``id_*.hdf5`` files per dataset, which makes the dataloader
NFS-bound (one ``open()`` round-trip per sample).  Repacking into tar shards
of ~1000 samples each cuts file-opens by 1000x and yields 5–10x dataloader
throughput on the same data.

Usage::

    python experiments/scripts/repack_geobench_v1.py m-eurosat
    python experiments/scripts/repack_geobench_v1.py m-bigearthnet --shard-size 1000
    python experiments/scripts/repack_geobench_v1.py m-eurosat --validate

Each output sample is::

    <id>.bands.npz   per-band float arrays keyed by their source name
    <id>.meta.pkl    raw bytes of the original HDF5 ``pickle`` attribute
    <id>.label       UTF-8 string of the integer/multilabel class

Partition JSON files are copied verbatim into the output dir so the new
loader can read them without changes.
"""

import argparse
import io
import logging
import pickle
import shutil
from pathlib import Path

import h5py
import numpy as np
import webdataset as wds

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class _StubUnpickler(pickle.Unpickler):
    """Match GeoBenchv1._load_sample_metadata: stub geobench module classes."""

    def find_class(self, module: str, name: str) -> type:  # type: ignore[override]
        if module == "geobench.dataset":
            return type(name, (), {})
        return super().find_class(module, name)


def _safe_unpickle(b: bytes) -> dict:
    try:
        return pickle.loads(b)
    except (ModuleNotFoundError, AttributeError):
        return _StubUnpickler(io.BytesIO(b)).load()


def _read_sample(hdf5_path: Path) -> tuple[dict[str, np.ndarray], bytes]:
    """Return ``(bands_dict, raw_pickle_bytes)`` for one V1 HDF5 sample."""
    with h5py.File(hdf5_path, "r") as f:
        bands = {k: f[k][:] for k in f}
        raw_pickle = f.attrs["pickle"]
    return bands, raw_pickle


def _resolve_pickle_bytes(raw: object) -> bytes:
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, str):
        # The upstream V1 distribution stored the bytes as a Python repr
        # of a bytestring (``"b'...'"``), so eval the string back.
        return eval(raw)  # noqa: S307 — trusted dataset payload
    raise TypeError(f"Unexpected pickle attr type: {type(raw)}")


def repack(dataset_dir: Path, out_dir: Path, shard_size: int = 1000) -> int:
    sample_paths = sorted(dataset_dir.glob("*.hdf5"))
    if not sample_paths:
        raise FileNotFoundError(f"No id_*.hdf5 files in {dataset_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "shard_%05d.tar")

    logger.info("Packing %d samples into shards of %d...", len(sample_paths), shard_size)
    written = 0
    with wds.ShardWriter(pattern, maxcount=shard_size, encoder=False) as sink:
        for hp in sample_paths:
            sid = hp.stem
            bands, raw_pickle = _read_sample(hp)
            bands_buf = io.BytesIO()
            np.savez(bands_buf, **bands)
            sink.write(
                {
                    "__key__": sid,
                    "bands.npz": bands_buf.getvalue(),
                    "meta.pkl": _resolve_pickle_bytes(raw_pickle),
                }
            )
            written += 1
            if written % 1000 == 0:
                logger.info("  packed %d / %d", written, len(sample_paths))

    # Carry partition + metadata files over so the new loader can find them
    # in the same place.
    for sidecar in dataset_dir.iterdir():
        if sidecar.suffix in (".json", ".pkl") or sidecar.name in (
            "LICENSE",
            "README",
            "README.md",
        ):
            shutil.copy2(sidecar, out_dir / sidecar.name)

    logger.info("Wrote %d samples to %s", written, out_dir)
    return written


def validate(dataset_dir: Path, out_dir: Path, n_samples: int = 50) -> None:
    """Cross-check the first ``n_samples`` between original HDF5 and shards."""
    import random

    sample_paths = sorted(dataset_dir.glob("*.hdf5"))
    rng = random.Random(0)
    rng.shuffle(sample_paths)
    targets = {p.stem: p for p in sample_paths[:n_samples]}

    shard_paths = sorted(out_dir.glob("shard_*.tar"))
    if not shard_paths:
        raise FileNotFoundError(f"No shards in {out_dir}")

    logger.info("Validating %d samples against %d shards...", len(targets), len(shard_paths))
    # Index shards directly (matches the runtime loader's logic for sample
    # IDs that contain ``.``) instead of relying on wds.WebDataset's
    # first-dot key split.
    import tarfile

    index: dict[str, dict[str, tuple]] = {}
    for path in shard_paths:
        with tarfile.open(path, "r") as t:
            for m in t.getmembers():
                for ext in ("bands.npz", "meta.pkl"):
                    suffix = "." + ext
                    if m.name.endswith(suffix):
                        base = m.name[: -len(suffix)]
                        index.setdefault(base, {})[ext] = (path, m.offset_data, m.size)
                        break

    def _read(ref):
        path, offset, size = ref
        with open(path, "rb") as f:
            f.seek(offset)
            return f.read(size)

    found = 0
    for sid in list(targets):
        parts = index.get(sid)
        if not parts:
            continue
        new_bands = dict(np.load(io.BytesIO(_read(parts["bands.npz"]))))
        new_meta = _safe_unpickle(_read(parts["meta.pkl"]))

        # Reference HDF5
        ref_bands, ref_pkl = _read_sample(targets[sid])
        ref_meta = _safe_unpickle(_resolve_pickle_bytes(ref_pkl))

        assert set(new_bands) == set(ref_bands), f"{sid}: band keys differ"
        for k in ref_bands:
            assert np.array_equal(new_bands[k], ref_bands[k]), f"{sid}: band '{k}' differs"
        assert np.array_equal(
            np.asarray(ref_meta.get("label")), np.asarray(new_meta.get("label"))
        ), f"{sid}: label differs"
        assert ref_meta.get("bands_order") == new_meta.get("bands_order"), (
            f"{sid}: bands_order differs"
        )
        found += 1

    if found != len(targets):
        raise RuntimeError(f"only {found}/{len(targets)} samples checked — missing in shards")
    logger.info("OK — %d samples bit-equal between HDF5 and shards", found)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="V1 dataset name (e.g. m-eurosat)")
    parser.add_argument(
        "--root", default="data/classification_v1.0", help="GeoBench V1 collection root"
    )
    parser.add_argument(
        "--out-root",
        default="data/classification_v1.0_wds",
        help="Output root for sharded copies",
    )
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument(
        "--validate", action="store_true", help="run a 50-sample bit-equality check"
    )
    args = parser.parse_args()

    dataset_dir = Path(args.root) / args.dataset
    out_dir = Path(args.out_root) / args.dataset
    repack(dataset_dir, out_dir, shard_size=args.shard_size)
    if args.validate:
        validate(dataset_dir, out_dir)


if __name__ == "__main__":
    main()
