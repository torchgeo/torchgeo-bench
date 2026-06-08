"""Normalizing flow generative classifier for UQ via Bayes inversion."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class NormalizingFlowProbe:
    """Conditional MAF flow modelling p(z|y); inverted via Bayes to give posteriors.

    Args:
        prior: ``"empirical"`` uses class frequencies; ``"uniform"`` uses equal weights.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        n_transforms: Number of MAF affine transforms.
        epochs: Training epochs.
        batch_size: Mini-batch size.
    """

    def __init__(
        self,
        prior: Literal["empirical", "uniform"] = "empirical",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        n_transforms: int = 8,
        epochs: int = 100,
        batch_size: int = 512,
    ) -> None:
        self.prior = prior
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.n_transforms = int(n_transforms)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self._flow = None
        self._log_prior: torch.Tensor | None = None
        self._n_classes: int | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the conditional MAF on training embeddings.

        Args:
            X_train: Training embeddings with shape ``(N, D)``.
            y_train: Training labels with shape ``(N,)``.

        Raises:
            ModuleNotFoundError: If ``zuko`` is not installed.
        """
        try:
            import zuko.flows as zflows
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "zuko>=0.3 required; pip install 'torchgeo-bench[uq]'"
            ) from exc

        D = X_train.shape[1]
        classes, counts = np.unique(y_train, return_counts=True)
        K = int(classes.shape[0])
        self._n_classes = K

        if self.prior == "empirical":
            log_prior = torch.log(torch.tensor(counts / counts.sum(), dtype=torch.float32))
        else:
            log_prior = torch.full((K,), -np.log(K), dtype=torch.float32)
        self._log_prior = log_prior

        hidden = [max(D // 8, 8), max(D // 8, 8)]
        flow = zflows.MAF(features=D, context=K, transforms=self.n_transforms, hidden_features=hidden)
        flow.train()

        x_t = torch.from_numpy(X_train.astype(np.float32, copy=False))
        y_t = torch.from_numpy(y_train.astype(np.int64, copy=False))
        ctx_t = F.one_hot(y_t, num_classes=K).float()

        loader = DataLoader(
            TensorDataset(x_t, ctx_t), batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        optimizer = torch.optim.AdamW(flow.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        for _ in range(self.epochs):
            for z_batch, ctx_batch in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = -flow(ctx_batch).log_prob(z_batch).mean()
                loss.backward()
                optimizer.step()
            scheduler.step()

        flow.eval()
        self._flow = flow

    def _log_class_lls(self, X: np.ndarray) -> torch.Tensor:
        """Compute per-class log-likelihoods for each sample.

        Args:
            X: Input embeddings with shape ``(N, D)``.

        Returns:
            Tensor of shape ``(N, K)`` with log p(z|y=k).
        """
        if self._flow is None:
            raise RuntimeError("NormalizingFlowProbe has not been fit yet.")

        K = self._n_classes
        x_t = torch.from_numpy(X.astype(np.float32, copy=False))
        N, D = x_t.shape

        # Memory guard: chunk if N*K*D > 500M float32 elements
        chunk_size = max(1, 500_000_000 // (K * D))
        log_class_lls_chunks = []

        with torch.no_grad():
            for start in range(0, N, chunk_size):
                x_chunk = x_t[start : start + chunk_size]  # (B, D)
                B = x_chunk.shape[0]
                # Expand to (B*K, D) and build one-hot contexts (B*K, K)
                x_exp = x_chunk.unsqueeze(1).expand(B, K, D).reshape(B * K, D)
                ctx = torch.eye(K).unsqueeze(0).expand(B, K, K).reshape(B * K, K)
                lp = self._flow(ctx).log_prob(x_exp)  # (B*K,)
                log_class_lls_chunks.append(lp.reshape(B, K))

        return torch.cat(log_class_lls_chunks, dim=0)  # (N, K)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class posterior probabilities via Bayes inversion.

        Args:
            X: Input embeddings with shape ``(N, D)``.

        Returns:
            Probability matrix with shape ``(N, C)`` as ``np.ndarray``.
        """
        log_lls = self._log_class_lls(X)  # (N, K)
        log_joint = log_lls + self._log_prior.unsqueeze(0)  # (N, K)
        probs = torch.softmax(log_joint, dim=1)
        return probs.numpy()

    def predict_confidence(self, X: np.ndarray) -> np.ndarray:
        """Return marginal log-likelihood as per-sample confidence score.

        Args:
            X: Input embeddings with shape ``(N, D)``.

        Returns:
            Array of shape ``(N,)`` with log p(z) values.
        """
        log_lls = self._log_class_lls(X)  # (N, K)
        log_joint = log_lls + self._log_prior.unsqueeze(0)  # (N, K)
        log_marg = torch.logsumexp(log_joint, dim=1)  # (N,)
        return log_marg.numpy().astype(np.float32)
