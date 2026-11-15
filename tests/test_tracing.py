import numpy as np

from napari_filament_finder.tracing import trace_filaments


def _sample(curve, tangent, n, rng, pos_noise, ang_noise, gap_frac):
    t = np.sort(rng.uniform(0, 1, n))
    t = t[rng.uniform(size=n) > gap_frac]
    pos = curve(t) + rng.normal(0, pos_noise, (len(t), 3))
    d = tangent(t) + rng.normal(0, ang_noise, (len(t), 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d *= rng.choice([-1, 1], size=(len(t), 1))  # bidirectional
    return np.stack([pos, d], axis=1)


def _line(offset, length=200.0):
    off = np.array(offset, float)
    return (
        lambda t: off + np.outer(t, [length, 0, 0]),
        lambda t: np.tile([1.0, 0, 0], (len(t), 1)),
    )


def _circle(radius=50.0):
    return (
        lambda t: np.column_stack(
            [
                radius * np.cos(2 * np.pi * t),
                radius * np.sin(2 * np.pi * t),
                np.zeros_like(t),
            ]
        ),
        lambda t: np.column_stack(
            [-np.sin(2 * np.pi * t), np.cos(2 * np.pi * t), np.zeros_like(t)]
        ),
    )


def _purity(ids, truth):
    """Every true filament maps to exactly one found id, no mixing."""
    found = [set(ids[truth == k]) for k in np.unique(truth)]
    return all(len(f) == 1 and -1 not in f for f in found) and len(
        set.union(*found)
    ) == len(found)


def _build(specs, rng, **kw):
    parts, truth = [], []
    for k, (curve, tangent) in enumerate(specs):
        s = _sample(curve, tangent, 80, rng, **kw)
        parts.append(s)
        truth += [k] * len(s)
    return np.concatenate(parts), np.array(truth)


def test_parallel_lines_noisy_with_gaps():
    rng = np.random.default_rng(0)
    specs = [_line([0, 0, 0]), _line([0, 12, 0]), _line([0, 0, 12])]
    data, truth = _build(
        specs, rng, pos_noise=0.5, ang_noise=0.1, gap_frac=0.3
    )
    ids, paths, contours = trace_filaments(data, max_gap=8, max_bridge=40)
    assert _purity(ids, truth)
    assert len(paths) == len(contours) == 3


def test_paths_are_ordered_along_filament():
    rng = np.random.default_rng(1)
    data, _ = _build(
        [_line([0, 0, 0])], rng, pos_noise=0.3, ang_noise=0.05, gap_frac=0.2
    )
    _, paths, _ = trace_filaments(data, max_gap=8, max_bridge=40)
    assert len(paths) == 1
    x = data[paths[0], 0, 0]
    # allow small local swaps from position noise
    rank = np.argsort(np.argsort(x))
    assert abs(np.corrcoef(rank, np.arange(len(x)))[0, 1]) > 0.99
    assert np.mean(np.sign(np.diff(x)) == np.sign(x[-1] - x[0])) > 0.85


def test_curved_filament_is_single():
    rng = np.random.default_rng(2)
    data, truth = _build(
        [_circle()], rng, pos_noise=0.3, ang_noise=0.05, gap_frac=0.2
    )
    ids, paths, _ = trace_filaments(data, max_gap=8, max_bridge=40, max_angle=40)
    assert len(paths) == 1
    assert (ids >= 0).mean() > 0.95


def test_big_gap_is_bridged():
    rng = np.random.default_rng(3)
    data, _ = _build(
        [_line([0, 0, 0])], rng, pos_noise=0.2, ang_noise=0.03, gap_frac=0.0
    )
    keep = (data[:, 0, 0] < 80) | (data[:, 0, 0] > 120)
    ids, paths, _ = trace_filaments(data[keep], max_gap=8, max_bridge=60)
    assert len(paths) == 1


def test_crossing_filaments_stay_separate():
    rng = np.random.default_rng(4)
    a = _line([0, 0, 0])
    b = (
        lambda t: np.array([100.0, -100, 0]) + np.outer(t, [0, 200, 0]),
        lambda t: np.tile([0, 1.0, 0], (len(t), 1)),
    )
    data, truth = _build(
        [a, b], rng, pos_noise=0.3, ang_noise=0.03, gap_frac=0.0
    )
    ids, paths, _ = trace_filaments(data, max_gap=8, max_bridge=40)
    assert len(paths) == 2
    for p in paths:
        assert len(set(truth[p])) == 1 or np.mean(truth[p] == truth[p][0]) > 0.9


def test_outliers_and_min_points():
    rng = np.random.default_rng(5)
    data, truth = _build(
        [_line([0, 0, 0])], rng, pos_noise=0.2, ang_noise=0.03, gap_frac=0.0
    )
    far = np.array([[[500.0, 500, 500], [1, 0, 0]]])
    ids, paths, _ = trace_filaments(
        np.concatenate([data, far]), max_gap=8, max_bridge=40
    )
    assert ids[-1] == -1
    assert len(paths) == 1


def test_bad_shape_raises():
    import pytest

    with pytest.raises(ValueError):
        trace_filaments(np.zeros((5, 3)))
