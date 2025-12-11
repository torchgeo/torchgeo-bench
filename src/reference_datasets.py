import geobench
import torch
from geobench.dataset import Sample
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples
from torchvision import transforms as T

NUM_CLASSES_PER_DATASET = {
    "m-forestnet": 12,
    "m-eurosat": 10,
    "m-pv4ger": 2,
    "m-brick-kiln": 2,
    "m-so2sat": 17,
    # "m-bigearthnet": None,  # TODO: Handle BigEarthNet separately
}

PARTITION_NAMES = [
    "0.01x_train",
    "0.02x_train",
    "0.05x_train",
    "0.10x_train",
    "0.20x_train",
    "0.50x_train",
    "1.00x_train",
    "default",
]


def get_transform(means, stdevs, band_names):
    augs = T.Compose([T.Normalize(mean=means, std=stdevs)])

    def transform(sample: Sample):
        x = sample.pack_to_3d(band_names=band_names)[0].astype("float32")
        image = torch.from_numpy(x).permute(2, 0, 1).squeeze(0)
        image = augs(image)
        return {"image": image, "label": torch.tensor(sample.label)}

    return transform


def get_datasets(
    dataset_name="m-forestnet",
    partition_name="0.02x_train",
    batch_size=32,
    normalization="mean_stdev",
    return_val=False,
    only_return_datasets=False,
):
    for task in geobench.task_iterator(benchmark_name="classification_v1.0"):
        if task.dataset_name != dataset_name:
            continue

        rgb_bands = ("red", "green", "blue")
        all_bands = [band.name for band in task.bands_info]
        selected_bands = rgb_bands

        train_dataset = task.get_dataset(
            split="train", band_names=selected_bands, partition_name=partition_name
        )
        if normalization == "mean_stdev":
            means, stdevs = train_dataset.normalization_stats()
        elif normalization == "min_max_99th":
            means = [0] * len(selected_bands)
            stdevs = []
            for band_name in train_dataset.band_names:
                stdevs.append(train_dataset.band_stats[band_name].percentile_99)
        elif normalization == "percentile_2_98":
            means = []
            stdevs = []
            for band_name in train_dataset.band_names:
                stats = train_dataset.band_stats[band_name]
                # Interpolate 2nd and 98th percentiles if not available
                p2 = getattr(stats, "percentile_2", None) or (
                    stats.percentile_1 + 0.25 * (stats.percentile_5 - stats.percentile_1)
                )
                p98 = getattr(stats, "percentile_98", None) or (
                    stats.percentile_95 + 0.75 * (stats.percentile_99 - stats.percentile_95)
                )
                means.append(p2)
                stdevs.append(p98 - p2)
        elif normalization == "min_max":
            means = []
            stdevs = []
            for band_name in train_dataset.band_names:
                min_val = train_dataset.band_stats[band_name].min
                max_val = train_dataset.band_stats[band_name].max
                means.append(min_val)
                stdevs.append(max_val - min_val)
        elif normalization == "255":
            means = [0] * len(selected_bands)
            stdevs = [255] * len(selected_bands)
        elif normalization == "none":
            means = [0] * len(selected_bands)
            stdevs = [1] * len(selected_bands)

        valid_partitions = train_dataset.list_partitions()
        transform = get_transform(means, stdevs, selected_bands)

        train_dataset = task.get_dataset(
            split="train",
            band_names=selected_bands,
            transform=transform,
            partition_name=partition_name,
        )
        valid_dataset = task.get_dataset(
            split="valid", band_names=selected_bands, transform=transform
        )
        test_dataset = task.get_dataset(
            split="test", band_names=selected_bands, transform=transform
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=stack_samples,
            num_workers=8,
        )
        val_dataloader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=stack_samples,
            num_workers=8,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=stack_samples,
            num_workers=8,
        )
        if only_return_datasets:
            if return_val:
                return train_dataset, valid_dataset, test_dataset
            else:
                return train_dataset, test_dataset

        if return_val:
            return train_dataset, train_dataloader, val_dataloader, test_dataloader
        else:
            return train_dataset, train_dataloader, test_dataloader
