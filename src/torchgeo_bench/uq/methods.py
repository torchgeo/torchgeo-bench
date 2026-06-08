"""Post-hoc uncertainty method implementations for linear probes."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torchgeo_bench.linear import LogisticRegression


class Uncalibrated:
    """Baseline method using the fitted probe probabilities as-is."""

    def __init__(self, probe: LogisticRegression) -> None:
        self._probe = probe

    def fit(self, *args, **kwargs) -> None:  # noqa: ARG002
        """No-op fit method for interface parity.

        Args:
            *args: Unused positional arguments.
            **kwargs: Unused keyword arguments.
        """

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return uncalibrated class probabilities from the fitted probe.

        Args:
            X: Input embeddings with shape ``(N, D)``.

        Returns:
            Probability matrix with shape ``(N, C)``.
        """
        X_t = torch.from_numpy(X.astype(np.float32, copy=False))
        return self._probe.predict_proba(X_t)


class TemperatureScaling:
    """Temperature scaling on a fixed fitted probe."""

    def __init__(self, probe: LogisticRegression) -> None:
        self._probe = probe
        self._log_T = nn.Parameter(torch.zeros(1))

    @property
    def log_temperature(self) -> nn.Parameter:
        """Return the learnable log-temperature parameter.

        Returns:
            Scalar log-temperature parameter.
        """
        return self._log_T

    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """Fit temperature by minimizing NLL on a calibration split.

        Args:
            X_cal: Calibration embeddings with shape ``(N, D)``.
            y_cal: Calibration labels with shape ``(N,)``.
        """
        logits_np = self._probe.decision_function(torch.from_numpy(X_cal.astype(np.float32)))
        logits = torch.from_numpy(logits_np.astype(np.float32))
        y_t = torch.from_numpy(y_cal.astype(np.int64))

        # Remap y_t through probe.classes_: the probe re-indexes training labels to [0, n_unique)
        # but cal labels are still original values. Mismatches cause IndexError in cross_entropy
        # when the cal set contains a class absent from training (e.g. class 9 in sen12ms).
        assert self._probe.classes_ is not None
        classes = torch.from_numpy(self._probe.classes_.astype(np.int64))
        remap = torch.full((int(classes.max()) + 1,), -1, dtype=torch.long)
        remap[classes] = torch.arange(len(classes), dtype=torch.long)
        y_t = remap[y_t]

        optimizer = torch.optim.LBFGS([self._log_T], lr=0.1, max_iter=200)

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            T = self._log_T.exp()
            loss = nn.functional.cross_entropy(logits / T, y_t)
            loss.backward()
            return loss

        optimizer.step(closure)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return temperature-scaled class probabilities.

        Args:
            X: Input embeddings with shape ``(N, D)``.

        Returns:
            Probability matrix with shape ``(N, C)``.
        """
        logits_np = self._probe.decision_function(torch.from_numpy(X.astype(np.float32, copy=False)))
        logits = torch.from_numpy(logits_np.astype(np.float32))
        probs = torch.softmax(logits / self._log_T.exp().detach(), dim=-1)
        return probs.detach().cpu().numpy()


class _EnsembleBase:
    """Shared prediction logic for linear probe ensembles."""

    _members: list[LogisticRegression]

    @property
    def classes_(self) -> np.ndarray | None:
        """Return classes seen during training, delegating to the first member."""
        if not self._members:
            return None
        return self._members[0].classes_

    def _member_probs(self, X: np.ndarray) -> np.ndarray:
        """Return stacked per-member probability arrays with shape ``(M, N, C)``."""
        if not self._members:
            raise RuntimeError(f"{self.__class__.__name__} has not been fit yet.")
        X_t = torch.from_numpy(X.astype(np.float32, copy=False))
        return np.stack([m.predict_proba(X_t) for m in self._members], axis=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return ensemble-averaged class probabilities.

        Args:
            X: Input embeddings with shape ``(N, D)``.

        Returns:
            Probability matrix with shape ``(N, C)``.

        Raises:
            RuntimeError: If ``fit`` has not been called.
        """
        return self._member_probs(X).mean(axis=0)

    def predict_confidence(self, X: np.ndarray) -> np.ndarray:
        """Return BALD-based confidence scores for selective prediction.

        Confidence is ``1 - normalised_BALD`` where BALD is the mutual
        information between predictions and model parameters (epistemic
        uncertainty).  Higher values mean the ensemble members agree and are
        collectively confident.

        Args:
            X: Input embeddings with shape ``(N, D)``.

        Returns:
            Confidence scores with shape ``(N,)`` in ``[0, 1]``.
        """
        member_probs = self._member_probs(X)  # (M, N, C)
        mean_probs = member_probs.mean(axis=0)  # (N, C)
        n_classes = mean_probs.shape[1]

        clipped_mean = np.clip(mean_probs, 1e-12, 1.0)
        H_mean = -(clipped_mean * np.log(clipped_mean)).sum(axis=1)  # (N,)

        clipped_members = np.clip(member_probs, 1e-12, 1.0)
        H_members = -(clipped_members * np.log(clipped_members)).sum(axis=2).mean(axis=0)  # (N,)

        bald = np.maximum(0.0, H_mean - H_members)  # (N,)
        max_bald = np.log(float(n_classes)) if n_classes > 1 else 1.0
        return 1.0 - bald / max_bald


class BootstrapEnsemble(_EnsembleBase):
    """Bootstrap ensemble of independently-fitted linear probes (lbfgs)."""

    def __init__(self, n: int = 5) -> None:
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        self.n = int(n)
        self._members: list[LogisticRegression] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        best_c: float,
        seed: int = 0,
    ) -> None:
        """Fit ensemble members on bootstrap resamples of the training data.

        Args:
            X_train: Training embeddings with shape ``(N, D)``.
            y_train: Training labels with shape ``(N,)``.
            best_c: Regularization strength for each member.
            seed: Base RNG seed for bootstrap sampling and member initialization.
        """
        self._members = []
        n = X_train.shape[0]
        for i in range(self.n):
            rng = np.random.default_rng(seed + i)
            idx = rng.integers(0, n, size=n)
            x_boot = torch.from_numpy(X_train[idx].astype(np.float32, copy=False))
            y_boot = torch.from_numpy(y_train[idx].astype(np.int64, copy=False))
            member = LogisticRegression(
                C=best_c,
                max_iter=4000,
                tol=1e-6,
                random_state=seed + i,
                solver="lbfgs",
            )
            member.fit(x_boot, y_boot)
            self._members.append(member)


class DeepEnsemble(_EnsembleBase):
    """Deep ensemble of linear probes trained with AdamW from random initializations.

    Unlike ``BootstrapEnsemble``, all members train on the full training set.
    Diversity comes from different random weight initializations and AdamW
    mini-batch stochasticity, following Lakshminarayanan et al. (2017).
    """

    def __init__(self, n: int = 5) -> None:
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        self.n = int(n)
        self._members: list[LogisticRegression] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        best_c: float,
        seed: int = 0,
    ) -> None:
        """Fit ensemble members on the full training set with different random initializations.

        Args:
            X_train: Training embeddings with shape ``(N, D)``.
            y_train: Training labels with shape ``(N,)``.
            best_c: Regularization strength for each member.
            seed: Base RNG seed for weight initialization and mini-batch ordering.
        """
        self._members = []
        x_t = torch.from_numpy(X_train.astype(np.float32, copy=False))
        y_t = torch.from_numpy(y_train.astype(np.int64, copy=False))
        for i in range(self.n):
            member = LogisticRegression(
                C=best_c,
                solver="adam",
                random_state=seed + i,
                random_init=True,
            )
            member.fit(x_t, y_t)
            self._members.append(member)


class LaplaceProbe:
    """Laplace approximation over all weights of a fitted linear probe."""

    def __init__(
        self,
        probe: LogisticRegression,
        batch_size: int = 512,
        pred_batch_size: int = 64,
    ) -> None:
        self._probe = probe
        self._batch_size = int(batch_size)
        self._pred_batch_size = int(pred_batch_size)
        self._la = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit a Kronecker-factored Laplace posterior over all probe weights.

        Args:
            X_train: Training embeddings with shape ``(N, D)``.
            y_train: Training labels with shape ``(N,)``.

        Raises:
            ModuleNotFoundError: If ``laplace-torch`` is not installed.
        """
        try:
            from laplace import Laplace
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "laplace-torch is required for LaplaceProbe. Install with `--extra uq`."
            ) from exc

        device = self._probe.module.weight.device
        x_t = torch.from_numpy(X_train.astype(np.float32, copy=False)).to(device)
        y_t = torch.from_numpy(y_train.astype(np.int64, copy=False)).to(device)
        loader = DataLoader(TensorDataset(x_t, y_t), batch_size=self._batch_size, shuffle=False)
        model = self._probe.module.to(device)
        model.eval()

        la = Laplace(
            model,
            likelihood="classification",
            subset_of_weights="all",
            hessian_structure="kron",
        )
        la.fit(loader)
        la.optimize_prior_precision(method="marglik")
        self._la = la

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return Laplace predictive class probabilities.

        Args:
            X: Input embeddings with shape ``(N, D)``.

        Returns:
            Probability matrix with shape ``(N, C)``.

        Raises:
            RuntimeError: If ``fit`` has not been called.
        """
        if self._la is None:
            raise RuntimeError("LaplaceProbe has not been fit yet.")
        device = next(self._la.model.parameters()).device
        chunks = []
        for start in range(0, len(X), self._pred_batch_size):
            x_chunk = torch.from_numpy(
                X[start : start + self._pred_batch_size].astype(np.float32, copy=False)
            ).to(device)
            p = self._la(x_chunk, pred_type="glm", link_approx="probit")
            if isinstance(p, tuple):
                p = p[0]
            chunks.append(p.detach().cpu())
        return torch.cat(chunks, dim=0).numpy()


class SVGPProbe:
    """Sparse variational GP probe for OOD diagnostics using GeoFM embeddings.

    Uses a Matérn-5/2 kernel with SoftmaxLikelihood and k-means inducing-point
    initialization. Intended as a diagnostic instrument, not a competing UQ method.
    """

    def __init__(
        self,
        n_inducing: int = 200,
        epochs: int = 100,
        lr: float = 0.01,
        batch_size: int = 512,
        n_mc_samples: int = 20,
    ) -> None:
        self.n_inducing = int(n_inducing)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.n_mc_samples = int(n_mc_samples)
        self._model = None
        self._likelihood = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit a sparse variational GP on training embeddings.

        Args:
            X_train: Training embeddings with shape ``(N, D)``.
            y_train: Training labels with shape ``(N,)``.

        Raises:
            ModuleNotFoundError: If ``gpytorch`` is not installed.
        """
        try:
            import gpytorch
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "gpytorch is required for SVGPProbe. Install with `pip install gpytorch`."
            ) from exc

        from sklearn.cluster import MiniBatchKMeans

        n_classes = int(np.unique(y_train).shape[0])
        n_inducing = min(self.n_inducing, len(X_train))

        kmeans = MiniBatchKMeans(n_clusters=n_inducing, random_state=0, n_init=3)
        kmeans.fit(X_train)
        inducing_pts = torch.from_numpy(kmeans.cluster_centers_.astype(np.float32))

        class _GP(gpytorch.models.ApproximateGP):
            def __init__(mdl, ind_pts: torch.Tensor, n_cls: int) -> None:
                vd = gpytorch.variational.CholeskyVariationalDistribution(
                    num_inducing_points=ind_pts.size(0),
                    batch_shape=torch.Size([n_cls]),
                )
                base_vs = gpytorch.variational.VariationalStrategy(
                    mdl, ind_pts, vd, learn_inducing_locations=True
                )
                # IndependentMultitaskVariationalStrategy turns batch_shape=[C]
                # into MultitaskMultivariateNormal(N, C) expected by SoftmaxLikelihood
                vs = gpytorch.variational.IndependentMultitaskVariationalStrategy(
                    base_vs, num_tasks=n_cls
                )
                super().__init__(vs)
                mdl.mean_module = gpytorch.means.ConstantMean(
                    batch_shape=torch.Size([n_cls])
                )
                mdl.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(nu=2.5, batch_shape=torch.Size([n_cls])),
                    batch_shape=torch.Size([n_cls]),
                )

            def forward(mdl, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
                return gpytorch.distributions.MultivariateNormal(
                    mdl.mean_module(x), mdl.covar_module(x)
                )

        model = _GP(inducing_pts, n_classes)
        likelihood = gpytorch.likelihoods.SoftmaxLikelihood(
            num_classes=n_classes, mixing_weights=False
        )

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(
            [*model.parameters(), *likelihood.parameters()],
            lr=self.lr,
        )
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(X_train))

        x_t = torch.from_numpy(X_train.astype(np.float32, copy=False))
        y_t = torch.from_numpy(y_train.astype(np.int64, copy=False))
        loader = DataLoader(TensorDataset(x_t, y_t), batch_size=self.batch_size, shuffle=True)

        for _ in range(self.epochs):
            for x_batch, y_batch in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = -mll(model(x_batch), y_batch)
                loss.backward()
                optimizer.step()

        self._model = model
        self._likelihood = likelihood

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities via MC sampling from the GP posterior.

        Args:
            X: Input embeddings with shape ``(N, D)``.

        Returns:
            Probability matrix with shape ``(N, C)``.

        Raises:
            RuntimeError: If ``fit`` has not been called.
        """
        if self._model is None:
            raise RuntimeError("SVGPProbe has not been fit yet.")

        self._model.eval()
        x_t = torch.from_numpy(X.astype(np.float32, copy=False))

        with torch.no_grad():
            # f_dist: MultitaskMultivariateNormal with event_shape=[N, C]
            f_dist = self._model(x_t)
            # rsample gives (S, N, C); softmax over C; mean over S
            f_samples = f_dist.rsample(torch.Size([self.n_mc_samples]))
            probs = torch.softmax(f_samples, dim=-1).mean(0)

        return probs.cpu().numpy()


@dataclass
class _SKLearnProbeWrapper:
    """Minimal sklearn-compatible wrapper around a fitted torchgeo probe."""

    probe: LogisticRegression

    @property
    def classes_(self) -> np.ndarray:
        """Return sorted class labels known by the fitted probe.

        Returns:
            Class label array with shape ``(C,)``.

        Raises:
            AttributeError: If the wrapped probe has not been fit.
        """
        if self.probe.classes_ is None:
            raise AttributeError("Probe must be fit before classes_ is available.")
        return self.probe.classes_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for MAPIE compatibility.

        Args:
            X: Input embeddings with shape ``(N, D)``.

        Returns:
            Probability matrix with shape ``(N, C)``.
        """
        return self.probe.predict_proba(torch.from_numpy(X.astype(np.float32, copy=False)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard class predictions for MAPIE compatibility.

        Args:
            X: Input embeddings with shape ``(N, D)``.

        Returns:
            Predicted class labels with shape ``(N,)``.
        """
        probs = self.predict_proba(X)
        return self.classes_[probs.argmax(axis=1)]


class ConformalPredictor:
    """Conformal predictor on top of a fitted probe."""

    def __init__(self, probe: LogisticRegression) -> None:
        self._probe = probe
        self._conformal = None
        self._fitted_alpha: float | None = None
        self._conformity_score: str = "raps"

    def _select_conformity_score(self, y_cal: np.ndarray) -> str:
        """Return a MAPIE-compatible conformity score for calibration labels.

        Args:
            y_cal: Calibration labels with shape ``(N,)``.

        Returns:
            ``"raps"`` for multiclass targets with enough calibration samples,
            ``"lac"`` otherwise (binary or too few cal samples for RAPS split).
        """
        n_classes = np.unique(y_cal).size
        if n_classes == 2:
            return "lac"
        # MAPIE's set_external_attributes resets size_raps to None, so sklearn uses
        # its default of 0.1 for StratifiedShuffleSplit regardless of RAPSConformityScore's
        # __init__ default of 0.2.  Guard against the actual effective split size.
        raps_internal_size = int(len(y_cal) * 0.1)
        if raps_internal_size < n_classes:
            return "lac"
        return "raps"

    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray, alpha: float = 0.1) -> None:
        """Fit conformal calibration in prefit mode on calibration embeddings.

        Args:
            X_cal: Calibration embeddings with shape ``(N, D)``.
            y_cal: Calibration labels with shape ``(N,)``.
            alpha: Miscoverage level for the fitted conformal object.

        Raises:
            ModuleNotFoundError: If MAPIE is not installed.
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        X_cal_np = X_cal.astype(np.float32, copy=False)
        y_cal_np = y_cal.astype(np.int64, copy=False)
        self._conformity_score = self._select_conformity_score(y_cal_np)
        wrapper = _SKLearnProbeWrapper(self._probe)

        try:
            from mapie.classification import SplitConformalClassifier
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "mapie is required for ConformalPredictor. Install with `--extra uq`."
            ) from exc

        conformal = SplitConformalClassifier(
            estimator=wrapper,
            confidence_level=1.0 - alpha,
            conformity_score=self._conformity_score,
            prefit=True,
        )
        conformal.conformalize(X_cal_np, y_cal_np)
        self._conformal = conformal
        self._fitted_alpha = float(alpha)

    def predict_sets(self, X_test: np.ndarray, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Predict point labels and conformal prediction sets.

        Args:
            X_test: Test embeddings with shape ``(N, D)``.
            alpha: Miscoverage level for prediction sets.

        Returns:
            Tuple ``(point_preds, pred_sets)`` where:
            - ``point_preds`` has shape ``(N,)``.
            - ``pred_sets`` has shape ``(N, C)`` with boolean membership.

        Raises:
            RuntimeError: If ``fit`` has not been called.
        """
        if self._conformal is None or self._fitted_alpha is None:
            raise RuntimeError("ConformalPredictor has not been fit yet.")
        if not np.isclose(float(alpha), self._fitted_alpha):
            raise ValueError(
                "ConformalPredictor was fit with alpha="
                f"{self._fitted_alpha:.6g}; refit to use alpha={float(alpha):.6g}."
            )
        X_test_np = X_test.astype(np.float32, copy=False)
        point_preds, pred_sets = self._conformal.predict_set(X_test_np)

        if pred_sets.ndim == 3:
            pred_sets = pred_sets[:, :, 0]
        return point_preds.astype(np.int64, copy=False), pred_sets.astype(bool, copy=False)

    def predict_confidence(self, X: np.ndarray) -> np.ndarray:
        """Return continuous confidence scores from the underlying probe.

        For LAC this equals ``1 - conformity_score(predicted_class)`` exactly.
        For RAPS it is the best available continuous proxy without reimplementing
        the full score function.  Use these scores for selective-classification
        metrics (AURC, E-AURC, selective accuracy) instead of the coarse
        ``1 / set_size`` signal.

        Args:
            X: Input embeddings with shape ``(N, D)``.

        Returns:
            Max-probability confidence scores with shape ``(N,)``.
        """
        X_t = torch.from_numpy(X.astype(np.float32, copy=False))
        return self._probe.predict_proba(X_t).max(axis=1)
