import numpy as np

from napari_filament_finder import utils


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
    w(viewer, layer, max_gap=15, tolerance=0)
    ids = layer.features['rlnFilamentID'].to_numpy()
    assert (ids[:6] == ids[0]).all() and (ids[6:12] == ids[6]).all()
    assert ids[0] != ids[6] and ids[0] >= 1 and ids[6] >= 1
    assert ids[12] == 0  # unassigned
    assert set(ids) == {0, 1, 2}  # filaments are numbered from 1
    assert (layer.edge_color[12] == 0).all()  # transparent
    assert layer.edge_color[0, 3] == 1
    # optionally show unassigned vectors in white
    w(viewer, layer, max_gap=15, tolerance=0, show_unassigned=True)
    assert (layer.edge_color[12] == 1).all()
    assert (layer.edge_color[0] != 1).any()  # filaments keep their color
    contours = viewer.layers[f'{layer.name} contours']
    assert len(contours.data) == 2
    assert set(contours.features['rlnFilamentID']) == {1, 2}
    # both filaments run 50 units along x; the contour is a straight line
    assert np.allclose(contours.features['length'], 50, atol=1e-6)
    # Re-running replaces the contours layer instead of duplicating it
    w(viewer, layer, max_gap=15, tolerance=0)
    assert sum(lyr.name.endswith('contours') for lyr in viewer.layers) == 1


def test_widget_ignores_off_track_point(make_napari_viewer):
    from napari_filament_finder import find_filaments

    viewer = make_napari_viewer()
    data = _two_filaments_and_noise()
    data[3, 0, 1] = 40  # far off-track point in the first filament
    layer = viewer.add_vectors(data)
    w = find_filaments()
    viewer.window.add_dock_widget(w)
    w(viewer, layer, max_gap=25, tolerance=1, min_point_count=3)
    ids = layer.features['rlnFilamentID'].to_numpy()
    assert ids[3] == 0
    assert (np.delete(ids[:6], 3) == ids[0]).all() and ids[0] >= 1


def test_closest_filament_is_written_to_contours_and_vectors(make_napari_viewer):
    from napari_filament_finder import find_filaments

    viewer = make_napari_viewer()
    layer = viewer.add_vectors(_two_filaments_and_noise())
    find_filaments()(viewer, layer, max_gap=15, tolerance=0)
    contours = viewer.layers[f'{layer.name} contours']
    # filaments 1 and 2 are 200 apart; each is the other's neighbor
    features = contours.features
    ids = features['rlnFilamentID'].to_numpy()
    closest = features['rlnClosestFilamentID'].to_numpy()
    assert (closest == 3 - ids).all()
    assert np.allclose(features['rlnClosestFilamentDistance'], 200)
    # vectors get the distance only
    vfeatures = layer.features
    assert 'rlnClosestFilamentPointIndex' not in vfeatures.columns
    assert 'rlnClosestFilamentID' not in vfeatures.columns
    assert np.allclose(vfeatures['rlnClosestFilamentPointDistance'][:12], 200)
    # the unassigned noise vector at (500, 500, 500) is far from everything
    assert vfeatures['rlnClosestFilamentPointDistance'][12] > 200


def test_closest_other_filament_without_candidates():
    idx, dist = utils.closest_other_filament(
        np.zeros((2, 3)), np.array([1, 1]), np.zeros((3, 3)), np.array([1, 1, 1])
    )
    assert (idx == -1).all() and np.isnan(dist).all()


def test_widget_sets_distances_only_on_button_press(make_napari_viewer):
    from napari_filament_finder import find_filaments

    viewer = make_napari_viewer()
    w = find_filaments()
    viewer.window.add_dock_widget(w)
    assert w.max_bridge.native.isHidden() and w.smoothing.native.isHidden()
    defaults = (w.max_gap.value, w.tolerance.value)
    layer = viewer.add_vectors(_two_filaments_and_noise())
    assert w.vectors_layer.value is layer
    # selecting a layer must not change the values by itself
    assert (w.max_gap.value, w.tolerance.value) == defaults
    w.set_from_data_button.native.click()
    spacing = utils.estimate_spacing(layer.data)
    assert abs(w.max_gap.value - 4 * spacing) < 0.01 * spacing + 0.1
    assert abs(w.tolerance.value - spacing) < 0.01 * spacing + 0.1
    assert w.max_bridge.value == 0  # 0 = automatic
    w.advanced.value = True
    assert not w.max_bridge.native.isHidden()
    assert not w.smoothing.native.isHidden()
    w.advanced.value = False
    assert w.max_bridge.native.isHidden()


def test_contour_length_uses_scale():
    contour = np.array([[0.0, 0, 0], [3, 4, 0], [3, 4, 12]])
    assert utils.contour_length(contour) == 5 + 12
    assert utils.contour_length(contour, scale=[2, 2, 1]) == 10 + 12
