import numpy as np

from napari_filament_finder import utils


def test_fit_filament_contour_line():
    t = np.linspace(0, 10, 8)
    coords = np.stack([t, 2 * t, np.zeros_like(t)], axis=1)
    rng = np.random.default_rng(0)
    contour, used = utils.fit_filament_contour(coords[rng.permutation(8)])
    assert used.all()
    assert contour.shape == (40, 3)
    assert np.allclose(contour[:, 2], 0, atol=1e-6)
    assert np.allclose(contour[:, 1], 2 * contour[:, 0], atol=1e-6)


def test_fit_filament_contours_skips_unassigned():
    coords = np.random.default_rng(1).random((6, 3))
    ids = np.array([0, 0, 0, 1, 1, -1])
    out, used = utils.fit_filament_contours(coords, ids)
    assert set(out) == {0, 1}


def _two_filaments_and_noise():
    t = np.linspace(0, 50, 6)
    f1 = np.stack([t, np.zeros(6), np.zeros(6)], axis=1)
    f2 = f1 + [0, 0, 200]
    noise = np.array([[500.0, 500, 500]])
    pos = np.concatenate([f1, f2, noise])
    vecs = np.tile([10.0, 0, 0], (len(pos), 1))
    return np.stack([pos, vecs], axis=1)


def test_widget_assigns_ids_and_creates_contours(make_napari_viewer):
    from napari_filament_finder import find_filaments

    viewer = make_napari_viewer()
    layer = viewer.add_vectors(_two_filaments_and_noise())
    w = find_filaments()
    viewer.window.add_dock_widget(w)
    w(viewer, layer, cone_angle=15, max_distance=15, tolerance=0)
    ids = layer.features['rlnFilamentID'].to_numpy()
    assert (ids[:6] == ids[0]).all() and (ids[6:12] == ids[6]).all()
    assert ids[0] != ids[6] and ids[0] >= 1 and ids[6] >= 1
    assert ids[12] == 0  # unassigned
    assert set(ids) == {0, 1, 2}  # filaments are numbered from 1
    assert (layer.edge_color[12] == 0).all()  # transparent
    assert layer.edge_color[0, 3] == 1
    # optionally show unassigned vectors in white
    w(viewer, layer, cone_angle=15, max_distance=15, tolerance=0, show_unassigned=True)
    assert (layer.edge_color[12] == 1).all()
    assert (layer.edge_color[0] != 1).any()  # filaments keep their color
    contours = viewer.layers[f'{layer.name} contours']
    assert len(contours.data) == 2
    assert set(contours.features['rlnFilamentID']) == {1, 2}
    # Re-running replaces the contours layer instead of duplicating it
    w(viewer, layer, cone_angle=15, max_distance=15, tolerance=0)
    assert sum(lyr.name.endswith('contours') for lyr in viewer.layers) == 1


def test_perpendicular_tolerance():
    # 3 px lateral jitter at 10 px spacing is ~17 degrees: outside a 15 deg cone
    pos = np.array([[0, 0, 0], [10, 3, 0], [20, 0, 0], [30, 3, 0]], float)
    v = np.stack([pos, np.tile([10.0, 0, 0], (4, 1))], axis=1)
    off = utils.assign_filament_ids(v, 15, 11, 2)
    on = utils.assign_filament_ids(v, 15, 11, 2, perpendicular_tolerance=3.5)
    assert (off == -1).all()
    assert (on == 0).all()


def test_max_deviation_ignores_outlier():
    t = np.linspace(0, 100, 11)
    coords = np.stack([t, np.zeros_like(t), np.zeros_like(t)], axis=1)
    coords[5, 1] = 20  # off-track point
    kept_all, used_all = utils.fit_filament_contour(coords)
    trimmed, used = utils.fit_filament_contour(coords, max_deviation=5)
    assert used_all.all()
    assert np.abs(kept_all[:, 1]).max() > 10
    assert np.abs(trimmed[:, 1]).max() < 1e-6
    assert not used[5] and used.sum() == 10
    # clean data is untouched
    clean = coords.copy()
    clean[5, 1] = 0
    contour, used = utils.fit_filament_contour(clean, max_deviation=5)
    assert used.all()
    assert np.allclose(contour, utils.fit_filament_contour(clean)[0])


def test_widget_assigns_minus_one_to_ignored_points(make_napari_viewer):
    from napari_filament_finder import find_filaments

    viewer = make_napari_viewer()
    data = _two_filaments_and_noise()
    data[3, 0, 1] = 8  # off-track point in the first filament
    layer = viewer.add_vectors(data)
    w = find_filaments()
    viewer.window.add_dock_widget(w)
    # the off-track vector (8 px off) is beyond the tolerance of 3 -> ID 0;
    # max_distance 25 still links its two neighbours (20 px apart)
    w(viewer, layer, cone_angle=15, max_distance=25, tolerance=3)
    ids = layer.features['rlnFilamentID'].to_numpy()
    assert ids[3] == 0
    assert (np.delete(ids[:6], 3) == ids[0]).all() and ids[0] >= 1


def test_column_choices_refresh(make_napari_viewer):
    from napari_filament_finder import find_closest_filament

    viewer = make_napari_viewer()
    w = find_closest_filament()
    viewer.window.add_dock_widget(w)
    layer = viewer.add_vectors(np.ones((3, 2, 3)))
    assert w.vectors_layer.value is layer
    assert 'rlnFilamentID' not in w.filament_id_column.choices
    layer.features = layer.features.assign(rlnFilamentID=[0, 0, 0])
    assert 'rlnFilamentID' in w.filament_id_column.choices
    assert w.filament_id_column.value == 'rlnFilamentID'


def _line_with_stray_point():
    """A straight filament, plus a point near it with a wrong direction."""
    t = np.arange(10) * 10.0
    pos = np.stack([t, np.zeros(10), np.zeros(10)], axis=1)
    vec = np.tile([10.0, 0, 0], (10, 1))
    # 2.5 px off the line (26 degrees seen from its neighbours) and with a
    # vector pointing sideways, so no cone at 15 degrees reaches it
    pos = np.vstack([pos, [45.0, 2.5, 0.0]])
    vec = np.vstack([vec, [0.0, 10.0, 0.0]])
    return np.stack([pos, vec], axis=1)


def test_point_with_neighbour_in_cone_joins_filament():
    # Regression: a point that only *has* neighbours (none has it in their
    # cone) used to be left unassigned although its neighbours were grouped.
    t = np.arange(6) * 10.0
    pos = np.stack([t, np.zeros(6), np.zeros(6)], axis=1)
    vec = np.tile([10.0, 0, 0], (6, 1))
    pos = np.vstack([pos, [20.0, 0.0, 400.0]])  # far away, pointing at filament
    vec = np.vstack([vec, [0.0, 0.0, 1.0]])
    v = np.stack([pos, vec], axis=1)
    ids = utils.assign_filament_ids(v, 5, 500, 2)
    assert (ids >= 0).all() and len(set(ids)) == 1


def test_refine_reassigns_points_on_contour():
    v = _line_with_stray_point()
    ids = utils.assign_filament_ids(v, 15, 11, 2)
    assert ids[-1] == -1 and (ids[:-1] == 0).all()
    # the stray vector points sideways, so only a loose angle accepts it
    refined, contours = utils.refine_filament_ids(
        v, ids, cone_angle=95, tolerance=3, min_point_count=2
    )
    assert (refined == 0).all()
    assert list(contours) == [0]
    # with a strict angle the sideways vector stays excluded
    strict, _ = utils.refine_filament_ids(
        v, ids, cone_angle=15, tolerance=3, min_point_count=2
    )
    assert strict[-1] == -1 and (strict[:-1] == 0).all()


def test_refine_moves_point_to_nearest_contour():
    t = np.arange(8) * 10.0
    a = np.stack([t, np.zeros(8), np.zeros(8)], axis=1)
    b = a + [0, 0, 100]
    pos = np.vstack([a, b])
    vec = np.tile([10.0, 0, 0], (16, 1))
    v = np.stack([pos, vec], axis=1)
    ids = np.array([0] * 8 + [1] * 8)
    ids[3] = 1  # wrong filament: lies clearly on contour 0
    refined, contours = utils.refine_filament_ids(
        v, ids, cone_angle=15, tolerance=3
    )
    assert refined[3] == refined[0]
    assert (refined[8:] == refined[8]).all() and refined[8] != refined[0]
    assert set(contours) == {0, 1}


def _arc(n=16, radius=40.0, sweep=np.pi * 1.5, z=0.0):
    a = np.linspace(0, sweep, n)
    return np.stack([radius * np.cos(a), radius * np.sin(a), np.full(n, z)], 1)


def test_ordering_follows_curved_filament():
    # A 270 degree arc: projecting on one axis would fold it back on itself
    arc = _arc()
    perm = np.random.default_rng(0).permutation(len(arc))
    order = utils.order_points_along_filament(arc[perm])
    sequence = perm[order]
    assert (np.diff(sequence) == 1).all() or (np.diff(sequence) == -1).all()


def test_points_on_curve_next_to_gap_are_kept():
    arc = _arc()
    arc = np.delete(arc, [7, 8], axis=0)  # a gap in the middle of the curve
    contour, used = utils.fit_filament_contour(arc, max_deviation=1.0)
    assert used.all()
    assert contour.shape[0] > 0


def test_end_points_of_curve_are_kept():
    arc = _arc(n=12)
    _, used = utils.fit_filament_contour(arc, max_deviation=1.0)
    assert used.all()


def test_off_track_point_is_still_ignored_on_curve():
    arc = _arc()
    arc[8] += [0, 0, 15]  # clearly off the arc
    _, used = utils.fit_filament_contour(arc, max_deviation=3.0)
    assert not used[8] and used.sum() == len(arc) - 1


def test_point_in_line_beyond_end_is_taken_back():
    # an end point lying on the straight continuation is not dropped
    t = np.arange(8) * 10.0
    pts = np.stack([t, np.zeros(8), np.zeros(8)], 1)
    pts[-1, 1] = 2.0  # slightly off, but within tolerance of the extension
    _, used = utils.fit_filament_contour(pts, max_deviation=3.0)
    assert used.all()


def test_widget_sets_distances_only_on_button_press(make_napari_viewer):
    from napari_filament_finder import find_filaments

    viewer = make_napari_viewer()
    w = find_filaments()
    viewer.window.add_dock_widget(w)
    assert w.join_tolerance.native.isHidden() and w.smoothing.native.isHidden()
    defaults = (w.max_distance.value, w.tolerance.value)
    layer = viewer.add_vectors(_two_filaments_and_noise())
    assert w.vectors_layer.value is layer
    # selecting a layer must not change the values by itself
    assert (w.max_distance.value, w.tolerance.value) == defaults
    w.set_from_data_button.native.click()
    spacing = utils.estimate_spacing(layer.data)
    assert abs(w.max_distance.value - 4 * spacing) < 0.01 * spacing + 0.1
    assert abs(w.tolerance.value - spacing / 3) < 0.01 * spacing + 0.1
    assert w.join_tolerance.value == 0  # 0 = same as tolerance
    w.advanced.value = True
    assert not w.join_tolerance.native.isHidden()
    assert not w.smoothing.native.isHidden()
    w.advanced.value = False
    assert w.join_tolerance.native.isHidden()


def test_join_tolerance_is_separate_from_keep_tolerance():
    t = np.arange(8) * 10.0
    pos = np.stack([t, np.zeros(8), np.zeros(8)], axis=1)
    pos = np.vstack([pos, [35.0, 5.0, 0.0]])  # 5 px off the filament line
    vec = np.vstack([np.tile([10.0, 0, 0], (8, 1)), [10.0, 0, 0]])
    v = np.stack([pos, vec], axis=1)
    ids = np.array([0] * 8 + [-1])
    strict, _ = utils.refine_filament_ids(v, ids, 15, tolerance=3)
    assert strict[-1] == -1  # beyond the 3 px tolerance
    loose, _ = utils.refine_filament_ids(
        v, ids, 15, tolerance=3, join_tolerance=6
    )
    assert loose[-1] == 0  # joins with the larger join tolerance
    assert (loose[:-1] == 0).all()  # keep tolerance unchanged
