"""Compare GeoBenchDataset vs geobench library reference implementation.

This test validates that the lightweight GeoBenchDataset implementation
produces the same results as the original geobench library implementation.
"""

import logging
import os

import pytest
import torch
from torch.utils.data import DataLoader

from torchgeo_bench.datasets import get_datasets as new_get_datasets

# Only import reference implementation if available
try:
    from torchgeo_bench.reference_datasets import get_datasets as ref_get_datasets

    HAS_REFERENCE = True
except ImportError:
    HAS_REFERENCE = False
    pytestmark = pytest.mark.skip(
        reason="Reference implementation (geobench library) not available"
    )


logger = logging.getLogger(__name__)


def compare_tensors(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    name: str,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> tuple[bool, str]:
    """Compare two tensors and report differences.

    Args:
        tensor1: First tensor (reference)
        tensor2: Second tensor (new implementation)
        name: Name for logging
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Tuple of (passed, message)
    """
    if tensor1.shape != tensor2.shape:
        return False, f"{name} shape mismatch: {tensor1.shape} vs {tensor2.shape}"

    if not torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol):
        diff = torch.abs(tensor1 - tensor2)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        msg = (
            f"{name} value mismatch: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}\n"
            f"  Reference: min={tensor1.min():.6f}, max={tensor1.max():.6f}, mean={tensor1.mean():.6f}\n"
            f"  New:       min={tensor2.min():.6f}, max={tensor2.max():.6f}, mean={tensor2.mean():.6f}"
        )
        return False, msg

    return True, f"{name} matches (shape={tensor1.shape})"


@pytest.mark.skipif(not HAS_REFERENCE, reason="Reference implementation not available")
class TestCompareImplementations:
    """Compare GeoBenchDataset with geobench library reference."""

    @pytest.fixture(autouse=True)
    def setup_geobench_env(self, geobench_root):
        """Setup environment variable for reference implementation."""
        # Reference implementation expects parent directory (without classification_v1.0)
        parent_dir = os.path.dirname(geobench_root)
        # Store original value to restore later
        original_env = os.environ.get("GEO_BENCH_DIR")
        os.environ["GEO_BENCH_DIR"] = parent_dir
        yield
        # Restore original environment
        if original_env is not None:
            os.environ["GEO_BENCH_DIR"] = original_env
        elif "GEO_BENCH_DIR" in os.environ:
            del os.environ["GEO_BENCH_DIR"]

    @pytest.mark.parametrize(
        "dataset_name",
        [
            "m-eurosat",
            "m-forestnet",
            "m-so2sat",
            "m-pv4ger",
            "m-brick-kiln",
        ],
    )
    def test_dataset_sizes(self, dataset_name: str, geobench_root: str):
        """Test that dataset sizes match between implementations."""
        partition_name = "0.01x_train"
        normalization = "mean_stdev"

        # Load reference datasets
        ref_train, ref_valid, ref_test = ref_get_datasets(
            dataset_name=dataset_name,
            partition_name=partition_name,
            normalization=normalization,
            only_return_datasets=True,
            return_val=True,
        )

        # Load new datasets
        new_train, new_valid, new_test = new_get_datasets(
            dataset_name=dataset_name,
            partition_name=partition_name,
            normalization=normalization,
            only_return_datasets=True,
            return_val=True,
            geobench_root=geobench_root,
        )

        # Compare sizes
        assert len(ref_train) == len(new_train), (
            f"Train size mismatch: {len(ref_train)} vs {len(new_train)}"
        )
        assert len(ref_valid) == len(new_valid), (
            f"Valid size mismatch: {len(ref_valid)} vs {len(new_valid)}"
        )
        assert len(ref_test) == len(new_test), (
            f"Test size mismatch: {len(ref_test)} vs {len(new_test)}"
        )

    @pytest.mark.parametrize(
        "dataset_name",
        [
            "m-eurosat",
            "m-forestnet",
            "m-so2sat",
            "m-pv4ger",
            "m-brick-kiln",
        ],
    )
    def test_sample_ids(self, dataset_name: str, geobench_root: str):
        """Test that sample IDs match between implementations."""
        partition_name = "0.01x_train"
        normalization = "mean_stdev"

        # Load reference datasets
        ref_train, ref_valid, ref_test = ref_get_datasets(
            dataset_name=dataset_name,
            partition_name=partition_name,
            normalization=normalization,
            only_return_datasets=True,
            return_val=True,
        )

        # Load new datasets
        new_train, new_valid, new_test = new_get_datasets(
            dataset_name=dataset_name,
            partition_name=partition_name,
            normalization=normalization,
            only_return_datasets=True,
            return_val=True,
            geobench_root=geobench_root,
        )

        # Compare sample IDs for each split
        for split_name, ref_ds, new_ds in [
            ("train", ref_train, new_train),
            ("valid", ref_valid, new_valid),
            ("test", ref_test, new_test),
        ]:
            # Get sample IDs from reference dataset
            if hasattr(ref_ds, "active_partition") and hasattr(
                ref_ds.active_partition, "partition_dict"
            ):
                ref_ids = ref_ds.active_partition.partition_dict[split_name]
            elif hasattr(ref_ds, "sample_ids"):
                ref_ids = ref_ds.sample_ids
            else:
                pytest.skip(f"Cannot extract sample IDs from reference {split_name} dataset")

            # Get sample IDs from new dataset
            if not hasattr(new_ds, "sample_ids"):
                pytest.fail(f"New {split_name} dataset missing sample_ids attribute")

            new_ids = new_ds.sample_ids

            # Compare IDs
            assert ref_ids == new_ids, (
                f"{split_name} sample IDs mismatch for {dataset_name}\n"
                f"  First 5 ref: {ref_ids[:5]}\n"
                f"  First 5 new: {new_ids[:5]}"
            )

    @pytest.mark.parametrize(
        "dataset_name",
        [
            "m-eurosat",
            "m-forestnet",
            "m-so2sat",
            "m-pv4ger",
            "m-brick-kiln",
        ],
    )
    def test_train_samples(self, dataset_name: str, geobench_root: str):
        """Test that train samples match between implementations."""
        partition_name = "0.01x_train"
        normalization = "mean_stdev"
        num_samples = 5  # Compare first 5 samples

        # Load reference datasets
        ref_train, _, _ = ref_get_datasets(
            dataset_name=dataset_name,
            partition_name=partition_name,
            normalization=normalization,
            only_return_datasets=True,
            return_val=True,
        )

        # Load new datasets
        new_train, _, _ = new_get_datasets(
            dataset_name=dataset_name,
            partition_name=partition_name,
            normalization=normalization,
            only_return_datasets=True,
            return_val=True,
            geobench_root=geobench_root,
        )

        # Compare first N samples
        for i in range(min(num_samples, len(new_train))):
            ref_sample = ref_train[i]
            new_sample = new_train[i]

            # Compare images
            ref_image = ref_sample["image"]
            new_image = new_sample["image"]

            passed, msg = compare_tensors(
                ref_image,
                new_image,
                f"Sample {i} image",
                rtol=1e-4,
                atol=1e-4,
            )
            assert passed, f"{dataset_name}: {msg}"

            # Compare labels
            ref_label = ref_sample["label"]
            new_label = new_sample["label"]

            if isinstance(ref_label, torch.Tensor):
                ref_label = ref_label.item()
            if isinstance(new_label, torch.Tensor):
                new_label = new_label.item()

            assert ref_label == new_label, (
                f"{dataset_name} sample {i} label mismatch: {ref_label} vs {new_label}"
            )

    @pytest.mark.parametrize(
        "dataset_name",
        [
            "m-eurosat",
            "m-forestnet",
            "m-so2sat",
            "m-pv4ger",
            "m-brick-kiln",
        ],
    )
    def test_validation_samples(self, dataset_name: str, geobench_root: str):
        """Test that validation samples match between implementations."""
        partition_name = "0.01x_train"
        normalization = "mean_stdev"
        num_samples = 3  # Compare first 3 validation samples

        # Load reference datasets
        _, ref_valid, _ = ref_get_datasets(
            dataset_name=dataset_name,
            partition_name=partition_name,
            normalization=normalization,
            only_return_datasets=True,
            return_val=True,
        )

        # Load new datasets
        _, new_valid, _ = new_get_datasets(
            dataset_name=dataset_name,
            partition_name=partition_name,
            normalization=normalization,
            only_return_datasets=True,
            return_val=True,
            geobench_root=geobench_root,
        )

        # Compare first N validation samples
        for i in range(min(num_samples, len(new_valid))):
            ref_sample = ref_valid[i]
            new_sample = new_valid[i]

            ref_image = ref_sample["image"]
            new_image = new_sample["image"]

            passed, msg = compare_tensors(
                ref_image,
                new_image,
                f"Validation sample {i}",
                rtol=1e-4,
                atol=1e-4,
            )
            assert passed, f"{dataset_name}: {msg}"

    @pytest.mark.parametrize(
        "dataset_name",
        [
            "m-eurosat",
            "m-forestnet",
            "m-so2sat",
            "m-pv4ger",
            "m-brick-kiln",
        ],
    )
    def test_batch_statistics(self, dataset_name: str, geobench_root: str):
        """Test that batch statistics match between implementations."""
        partition_name = "0.01x_train"
        normalization = "mean_stdev"
        batch_size = 32

        # Load reference datasets
        ref_train, _, _ = ref_get_datasets(
            dataset_name=dataset_name,
            partition_name=partition_name,
            normalization=normalization,
            only_return_datasets=True,
            return_val=True,
        )

        # Load new datasets
        new_train, _, _ = new_get_datasets(
            dataset_name=dataset_name,
            partition_name=partition_name,
            normalization=normalization,
            only_return_datasets=True,
            return_val=True,
            geobench_root=geobench_root,
        )

        # Create dataloaders
        ref_loader = DataLoader(
            ref_train,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        new_loader = DataLoader(
            new_train,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Compare first batch
        ref_batch = next(iter(ref_loader))
        new_batch = next(iter(new_loader))

        # Compare batch images
        passed, msg = compare_tensors(
            ref_batch["image"],
            new_batch["image"],
            "Batch images",
            rtol=1e-4,
            atol=1e-4,
        )
        assert passed, f"{dataset_name}: {msg}"

        # Compare batch labels
        passed, msg = compare_tensors(
            ref_batch["label"],
            new_batch["label"],
            "Batch labels",
        )
        assert passed, f"{dataset_name}: {msg}"

    @pytest.mark.parametrize(
        "dataset_name,normalization",
        [
            ("m-eurosat", "mean_stdev"),
            ("m-eurosat", "min_max"),
            ("m-eurosat", "none"),
        ],
    )
    def test_normalizations(self, dataset_name: str, normalization: str, geobench_root: str):
        """Test different normalization methods."""
        partition_name = "0.01x_train"

        try:
            # Load reference datasets
            ref_train, _, _ = ref_get_datasets(
                dataset_name=dataset_name,
                partition_name=partition_name,
                normalization=normalization,
                only_return_datasets=True,
                return_val=True,
            )
        except Exception as e:
            pytest.skip(f"Reference implementation failed with {normalization}: {e}")

        # Load new datasets
        new_train, _, _ = new_get_datasets(
            dataset_name=dataset_name,
            partition_name=partition_name,
            normalization=normalization,
            only_return_datasets=True,
            return_val=True,
            geobench_root=geobench_root,
        )

        # Compare first sample
        ref_sample = ref_train[0]
        new_sample = new_train[0]

        passed, msg = compare_tensors(
            ref_sample["image"],
            new_sample["image"],
            f"Sample with {normalization}",
            rtol=1e-4,
            atol=1e-4,
        )
        assert passed, msg


@pytest.mark.skipif(not HAS_REFERENCE, reason="Reference implementation not available")
def test_all_datasets_pass(geobench_root: str):
    """Integration test: verify all datasets pass comparison.

    This is a convenience test that runs a quick check on all datasets
    to ensure the implementations are compatible.
    """
    datasets = ["m-eurosat", "m-forestnet", "m-so2sat", "m-pv4ger", "m-brick-kiln"]
    partition_name = "0.01x_train"
    normalization = "mean_stdev"

    results = {}

    for dataset_name in datasets:
        try:
            # Load both implementations
            ref_train, _, _ = ref_get_datasets(
                dataset_name=dataset_name,
                partition_name=partition_name,
                normalization=normalization,
                only_return_datasets=True,
                return_val=True,
            )

            new_train, _, _ = new_get_datasets(
                dataset_name=dataset_name,
                partition_name=partition_name,
                normalization=normalization,
                only_return_datasets=True,
                return_val=True,
                geobench_root=geobench_root,
            )

            # Quick check: compare first sample
            ref_sample = ref_train[0]
            new_sample = new_train[0]

            passed, _ = compare_tensors(
                ref_sample["image"],
                new_sample["image"],
                f"{dataset_name}",
                rtol=1e-4,
                atol=1e-4,
            )

            results[dataset_name] = passed

        except Exception as e:
            results[dataset_name] = False
            logger.error(f"Failed to compare {dataset_name}: {e}")

    # Check that all passed
    failed = [name for name, passed in results.items() if not passed]
    assert not failed, f"Datasets failed comparison: {failed}"
