import numpy as np
import torch
from tqdm import tqdm


def extract_features(model, dataloader, device, transforms=None, verbose=True):
    x_all = []
    y_all = []

    enumerator = enumerate(dataloader)
    if verbose:
        enumerator = enumerate(tqdm(dataloader, total=len(dataloader)))

    for i, batch in enumerator:
        images = batch["image"].to(device)
        labels = batch["label"].numpy()

        if transforms is not None:
            images = transforms(images)

        with torch.no_grad():
            with torch.inference_mode():
                features = model(images)
                if isinstance(features, torch.Tensor):
                    features = features.detach().cpu().numpy()
                else:
                    if "norm" in features:
                        features = features["norm"].detach().cpu().numpy()
                    elif "global_pool" in features:
                        features = features["global_pool"].detach().cpu().numpy()
                    elif "head.global_pool" in features:
                        features = features["head.global_pool"].detach().cpu().numpy().squeeze()
                    else:
                        raise ValueError(f"Unexpected features format: {features.keys()}")

                # handles the case where features are 1D (e.g., the ResNet model has batch x features)
                if len(features.shape) == 1:
                    features = features[np.newaxis, :]

                # handles the case where features are 3D (e.g., the DinoV2 model has batch x tokens x features)
                if len(features.shape) == 3:
                    features = np.mean(features, axis=1, keepdims=False)

        x_all.append(features)
        y_all.append(labels)

    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    return x_all, y_all


def extract_features_transformers(model, dataloader, device, processor=None, verbose=True):
    x_all = []
    y_all = []

    enumerator = enumerate(dataloader)
    if verbose:
        enumerator = enumerate(tqdm(dataloader, total=len(dataloader)))

    for i, batch in enumerator:
        images = batch["image"].to(device)
        labels = batch["label"].numpy()

        # images = processor(images=images, return_tensors="pt")

        images = processor(images)
        # images = {'pixel_values': images}
        with torch.no_grad(), torch.inference_mode():
            # features = model(**images)
            features = model(images)

        # last_hidden_states = features.last_hidden_state
        # cls_token = last_hidden_states[:, 0, :]

        # x_all.append(cls_token.cpu().numpy())
        x_all.append(features.cpu().numpy())
        y_all.append(labels)

    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    return x_all, y_all
