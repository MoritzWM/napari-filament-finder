from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from magicgui import magic_factory
from napari.utils.colormaps.vendored import cm

from . import utils

if TYPE_CHECKING:
    import napari


@magic_factory(
    cone_angle={'min': 0.1, 'max': 180},
    max_distance={'min': 0.1, 'max': 1e9},
    perpendicular_tolerance={'min': 0, 'max': 1e9},
    min_point_count={'min': 1, 'max': 1e6},
)
def find_filaments(
    vectors_layer: 'napari.layers.Vectors',  # pyright: ignore[reportAttributeAccessIssue]
    cone_angle: float = 15,
    max_distance: float = 100,
    perpendicular_tolerance: float = 0,
    min_point_count: int = 2,
):
    """Group vectors into filaments.

    A neighbour is accepted if it lies within the cone, or within
    `perpendicular_tolerance` of the vector's axis (0 = disabled).
    """
    ids = utils.assign_filament_ids(
        vectors_layer.data,
        cone_angle=cone_angle,
        max_distance=max_distance,
        min_point_count=min_point_count,
        perpendicular_tolerance=perpendicular_tolerance,
    )
    # Assign a new DataFrame (not in-place) so napari emits the features event
    # and the layer controls' feature comboboxes refresh.
    vectors_layer.features = vectors_layer.features.assign(rlnFilamentID=ids)
    vectors_layer.edge_color = cm.get_cmap('tab20')(ids % 20)


def find_closest_filament_init(widget):
    """Keep the filament ID column choices in sync with the vectors layer.

    Refreshes when the selected layer changes, when its features change
    (e.g. after running "Find filaments"), and when the widget is shown.
    """
    state = {'layer': None}

    def refresh(*_):
        layer = widget.vectors_layer.value
        features = None if layer is None else layer.features
        columns = [] if features is None else list(features.columns)
        widget.filament_id_column.choices = columns
        if not columns:
            return
        current = widget.filament_id_column.value
        if current in columns:
            return
        index = utils.find_word_index('filament', columns)
        widget.filament_id_column.value = columns[index or 0]

    def on_layer_changed(layer):
        old = state['layer']
        if old is not None:
            old.events.features.disconnect(refresh)
        state['layer'] = layer
        if layer is not None:
            layer.events.features.connect(refresh)
        refresh()

    def on_choices_reset(*_):
        # The layer combobox is populated (and repopulated whenever layers are
        # added/removed) without necessarily emitting `changed`.
        layer = widget.vectors_layer.value
        if layer is not state['layer']:
            on_layer_changed(layer)
        else:
            refresh()

    widget.vectors_layer.changed.connect(on_layer_changed)
    original_reset = widget.vectors_layer.reset_choices

    def reset_choices(*args):
        original_reset(*args)
        on_choices_reset()

    widget.vectors_layer.reset_choices = reset_choices
    widget.called.connect(on_choices_reset)
    widget.native_parent_changed.connect(on_choices_reset)

    import napari

    viewer = napari.current_viewer()
    if viewer is not None:
        # Connected after magicgui's own layer-list handlers, so the layer
        # combobox is already up to date when we refresh.
        for event in (viewer.layers.events.inserted, viewer.layers.events.removed):
            event.connect(on_choices_reset)
    widget.native.showEvent = _chain_show_event(widget.native, refresh)
    on_layer_changed(widget.vectors_layer.value)


def _chain_show_event(native, callback):
    original = native.showEvent

    def show_event(event):
        callback()
        original(event)

    return show_event


@magic_factory(
    widget_init=find_closest_filament_init,
    filament_id_column={'choices': [], 'widget_type': 'ComboBox'},
)
def find_closest_filament(
    vectors_layer: 'napari.layers.Vectors',  # pyright: ignore[reportAttributeAccessIssue]
    filament_id_column: str,
):
    features = vectors_layer.features
    if features is None:
        return
    if filament_id_column not in features.columns:
        return
    filament_ids = features[filament_id_column].to_numpy()
    indices, distances = utils.assign_closest_neighbor(
        vectors_or_coords=vectors_layer.data, filament_ids=filament_ids
    )
    vectors_layer.features = features.assign(
        rlnClosestFilamentID=filament_ids[indices],
        rlnClosestFilamentPointIndex=indices,
        rlnClosestFilamentPointDistance=distances,
    )


def fit_filament_contours_init(widget):
    # Reuse the column-selection logic of find_closest_filament
    find_closest_filament_init(widget)


@magic_factory(
    widget_init=fit_filament_contours_init,
    filament_id_column={'choices': [], 'widget_type': 'ComboBox'},
    smoothing={'min': 0, 'max': 1e9},
    max_deviation={'min': 0, 'max': 1e9},
    samples_per_point={'min': 1, 'max': 1000},
    edge_width={'min': 0.1, 'max': 1e6},
)
def fit_filament_contours(
    viewer: 'napari.Viewer',  # pyright: ignore[reportAttributeAccessIssue]
    vectors_layer: 'napari.layers.Vectors',  # pyright: ignore[reportAttributeAccessIssue]
    filament_id_column: str,
    smoothing: float = 0,
    max_deviation: float = 0,
    samples_per_point: int = 5,
    edge_width: float = 2,
):
    """Fit a spline contour (Shapes layer) through each identified filament.

    Points further than `max_deviation` from the curve through the other points are
    ignored for the fit (0 = use all points).
    """
    features = vectors_layer.features
    if features is None or filament_id_column not in features.columns:
        return
    data = vectors_layer.data
    coords = data[:, 0, :] if data.ndim == 3 else data
    filament_ids = features[filament_id_column].to_numpy()
    contours = utils.fit_filament_contours(
        coords,
        filament_ids,
        smoothing=smoothing,
        max_deviation=max_deviation,
        samples_per_point=samples_per_point,
    )
    # Contours need at least two vertices to form a path
    contours = {k: v for k, v in contours.items() if len(v) >= 2}
    if not contours:
        return
    ids = list(contours)
    name = f'{vectors_layer.name} contours'
    kwargs = {
        'shape_type': 'path',
        'edge_width': edge_width,
        'edge_color': cm.get_cmap('tab20')(np.array(ids) % 20),
        'face_color': 'transparent',
        'features': pd.DataFrame({'filament_id': ids}),
        'scale': vectors_layer.scale,
        'translate': vectors_layer.translate,
        'name': name,
    }
    if name in viewer.layers:
        viewer.layers.remove(name)
    viewer.add_shapes([contours[i] for i in ids], **kwargs)
