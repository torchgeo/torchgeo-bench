"""Smoke test: KNNClassifier(device='cuda') ↔ device='cpu' parity."""
import sys, os
sys.path.insert(0, "src")
import numpy as np
from torchgeo_bench.knn import KNNClassifier

rng = np.random.default_rng(0)
n_train, n_test, d, k = 2000, 500, 64, 5

X_train = rng.standard_normal((n_train, d)).astype(np.float32)
X_test  = rng.standard_normal((n_test,  d)).astype(np.float32)

# --- single label ---
n_classes = 10
y_train = rng.integers(0, n_classes, size=n_train).astype(np.int64)

cpu = KNNClassifier(n_neighbors=k, device="cpu").fit(X_train, y_train)
cu  = KNNClassifier(n_neighbors=k, device="cuda").fit(X_train, y_train)

p_cpu = cpu.predict(X_test)
p_cu  = cu.predict(X_test)
agree = float((p_cpu == p_cu).mean())
print(f"singlelabel  predict agreement = {agree:.4f}  (n_classes={n_classes})")

pp_cpu = cpu.predict_proba(X_test)
pp_cu  = cu.predict_proba(X_test)
print(f"singlelabel  predict_proba shapes = {pp_cpu.shape} {pp_cu.shape}")
print(f"singlelabel  predict_proba max abs diff = {np.abs(pp_cpu - pp_cu).max():.4g}")

# --- multilabel ---
Y_train = (rng.random((n_train, n_classes)) > 0.7).astype(np.int64)
cpu_ml = KNNClassifier(n_neighbors=k, device="cpu").fit(X_train, Y_train)
cu_ml  = KNNClassifier(n_neighbors=k, device="cuda").fit(X_train, Y_train)

p_cpu = cpu_ml.predict(X_test)
p_cu  = cu_ml.predict(X_test)
print(f"multilabel   predict shapes = {p_cpu.shape} {p_cu.shape}")
print(f"multilabel   predict agreement = {float((p_cpu == p_cu).mean()):.4f}")

pp_cpu = cpu_ml.predict_proba(X_test)
pp_cu  = cu_ml.predict_proba(X_test)
print(f"multilabel   predict_proba max abs diff = {np.abs(pp_cpu - pp_cu).max():.4g}")
print("OK")
