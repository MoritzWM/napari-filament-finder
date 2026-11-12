from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from magicgui import magic_factory
from magicgui.widgets import PushButton
from napari.utils.colormaps.vendored import cm
from napari.utils.notifications import show_warning

from . import utils

if TYPE_CHECKING:
    import napari

_CONTOUR_SAMPLES_PER_POINT = 10
_CONTOUR_EDGE_WIDTH = 2


def _filament_colors(ids: np.ndarray, show_unassigned: bool = False) -> np.ndarray:
    """
    tab20 colors for filament IDs >= 1. Unassigned vectors (ID 0) are white if
    `show_unassigned`, otherwise transparent.
    """
    colors = np.asarray(cm.get_cmap('tab20')((ids - 1) % 20), dtype=float)
    colors[ids <= 0] = 1.0 if show_unassigned else 0.0
    return colors


def find_filaments_init(widget):
    """Add a button that derives `max_distance` and `tolerance` from the
    layer's point spacing; fold away the advanced settings until requested."""

    def set_from_data():
        layer = widget.vectors_layer.value
        if layer is None or len(layer.data) < 2:
            show_warning('Select a vectors layer with at least two vectors.')
            return
        spacing = utils.estimate_spacing(layer.data)
        if spacing <= 0:
            show_warning('Could not estimate the spacing of the vectors.')
            return
        # Four times the mean spacing bridges a few missing points
        widget.max_distance.value = float(f'{4 * spacing:.3g}')
        # A third of the spacing: above typical picking scatter, yet far
        # below the distance between neighbouring filaments
        widget.tolerance.value = float(f'{spacing / 3:.3g}')

    def on_advanced_changed(show):
        widget.join_tolerance.visible = show
        widget.smoothing.visible = show

    button = PushButton(text='Estimate from data')
    button.changed.connect(set_from_data)
    widget.insert(widget.index('tolerance') + 1, button)
    widget.set_from_data_button = button

    widget.advanced.changed.connect(on_advanced_changed)
    on_advanced_changed(widget.advanced.value)


@magic_factory(
    widget_init=find_filaments_init,
    cone_angle={'min': 0.1, 'max': 180},
    max_distance={'min': 0.1, 'max': 1e9},
    tolerance={'min': 0, 'max': 1e9},
    min_point_count={'min': 1, 'max': 1e6},
    join_tolerance={'min': 0, 'max': 1e9},
    smoothing={'min': 0, 'max': 1e9},
)
def find_filaments(
    viewer: 'napari.Viewer',  # pyright: ignore[reportAttributeAccessIssue]
    vectors_layer: 'napari.layers.Vectors',  # pyright: ignore[reportAttributeAccessIssue]
    cone_angle: float = 15,
    max_distance: float = 100,
    tolerance: float = 1,
    min_point_count: int = 2,
    show_unassigned: bool = False,
    advanced: bool = False,
    join_tolerance: float = 0,
    smoothing: float = 1,
):
    """Group vectors into filaments and fit a contour through each one.

    Every vector gets a filament ID (`rlnFilamentID`), numbered from 1;
    vectors that are not part of any filament get ID 0 and a transparent
    color (or white, see `show_unassigned`). A Shapes layer with one spline
    contour per filament is created.

    cone_angle: a neighbour must lie within this angle of the vector's axis.
    max_distance: how far to look for neighbours. The button "Set distances
        from data" sets it to four times the mean spacing of the vectors.
    tolerance: maximum distance of a vector from its filament, in the layer's
        units (the button sets it to a third of the mean spacing). A neighbour within this distance of the vector's axis is
        accepted regardless of the angle, and vectors further than this from
        the fitted contour are ignored (ID 0). 0 disables both, and the
        contour-based reassignment.
    min_point_count: smallest number of vectors forming a filament.
    show_unassigned: show unassigned vectors (ID 0) in white instead of
        hiding them (transparent).
    join_tolerance: (advanced) distance from a contour within which a vector
        is assigned to it, also if the first grouping missed it. 0 = same as
        tolerance.
    smoothing: (advanced) contour smoothing; 0 passes through every point.
    """
    data = vectors_layer.data
    ids = utils.assign_filament_ids(
        data,
        cone_angle=cone_angle,
        max_distance=max_distance,
        min_point_count=min_point_count,
        perpendicular_tolerance=tolerance,
    )
    # Fit contours; with a tolerance, also reassign every vector to the
    # contour it lies on, and ignore (ID 0) vectors that are too far off track.
    ids, contours = utils.refine_filament_ids(
        data,
        ids,
        cone_angle=cone_angle,
        tolerance=tolerance,
        min_point_count=min_point_count,
        smoothing=smoothing,
        samples_per_point=_CONTOUR_SAMPLES_PER_POINT,
        join_tolerance=join_tolerance if join_tolerance > 0 else tolerance,
    )
    # Internally unassigned is -1 and filaments are 0-based; in the output ID 0
    # means unassigned and filaments are numbered from 1.
    ids = ids + 1
    contours = {fid + 1: contour for fid, contour in contours.items()}
    # Assign a new DataFrame (not in-place) so napari emits the features event
    # and the layer controls' feature comboboxes refresh.
    vectors_layer.features = vectors_layer.features.assign(rlnFilamentID=ids)
    vectors_layer.edge_color = _filament_colors(ids, show_unassigned)

    # Contours need at least two vertices to form a path
    contours = {k: v for k, v in contours.items() if len(v) >= 2}
    name = f'{vectors_layer.name} contours'
    # Remove the previous result so a stale contours layer never lingers
    if name in viewer.layers:
        viewer.layers.remove(name)
    if not contours:
        return
    contour_ids = list(contours)
    viewer.add_shapes(
        [contours[i] for i in contour_ids],
        shape_type='path',
        edge_width=_CONTOUR_EDGE_WIDTH,
        edge_color=_filament_colors(np.array(contour_ids)),
        face_color='transparent',
        features=pd.DataFrame({'rlnFilamentID': contour_ids}),
        scale=vectors_layer.scale,
        translate=vectors_layer.translate,
        name=name,
    )


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
