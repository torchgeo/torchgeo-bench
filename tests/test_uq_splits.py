import numpy as np

from torchgeo_bench.uq.splits import stratified_cal_split


def test_stratified_cal_split_shapes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 8)).astype(np.float32)
    y = np.repeat(np.arange(4), 50).astype(np.int64)

    X_cal, y_cal, X_rem, y_rem = stratified_cal_split(X, y, cal_size=40, seed=123)
    assert X_cal.shape == (40, 8)
    assert y_cal.shape == (40,)
    assert X_rem.shape == (160, 8)
    assert y_rem.shape == (160,)


def test_stratified_cal_split_no_overlap():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 6)).astype(np.float32)
    y = np.repeat(np.arange(4), 50).astype(np.int64)

    X_cal, _, X_rem, _ = stratified_cal_split(X, y, cal_size=40, seed=42)

    x_to_indices: dict[tuple[float, ...], list[int]] = {}
    for idx, row in enumerate(X):
        x_to_indices.setdefault(tuple(row.tolist()), []).append(idx)

    cal_indices: set[int] = set()
    rem_indices: set[int] = set()
    for row in X_cal:
        cal_indices.add(x_to_indices[tuple(row.tolist())].pop())
    for row in X_rem:
        rem_indices.add(x_to_indices[tuple(row.tolist())].pop())

    assert cal_indices & rem_indices == set()


def test_stratified_cal_split_reproducible():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 5)).astype(np.float32)
    y = np.repeat(np.arange(4), 50).astype(np.int64)

    split_a = stratified_cal_split(X, y, cal_size=40, seed=7)
    split_b = stratified_cal_split(X, y, cal_size=40, seed=7)
    split_c = stratified_cal_split(X, y, cal_size=40, seed=8)

    for arr_a, arr_b in zip(split_a, split_b, strict=True):
        assert np.array_equal(arr_a, arr_b)
    assert not np.array_equal(split_a[0], split_c[0])


def test_stratified_cal_split_class_distribution():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 4)).astype(np.float32)
    y = np.repeat(np.arange(4), 50).astype(np.int64)

    _, y_cal, _, _ = stratified_cal_split(X, y, cal_size=40, seed=99)
    unique = set(np.unique(y_cal).tolist())
    assert unique == {0, 1, 2, 3}
