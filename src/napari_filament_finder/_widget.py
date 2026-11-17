from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from magicgui import magic_factory
from magicgui.widgets import PushButton
from napari.utils.colormaps.vendored import cm
from napari.utils.notifications import show_warning

from . import utils
from .tracing import trace_filaments

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
    """Add a button that derives `max_gap` and `tolerance` from the
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
        widget.max_gap.value = float(f'{4 * spacing:.3g}')
        # The spacing: above typical picking scatter, yet below the
        # distance between neighbouring filaments
        widget.tolerance.value = float(f'{spacing:.3g}')

    def on_advanced_changed(show):
        widget.max_bridge.visible = show
        widget.smoothing.visible = show

    button = PushButton(text='Estimate from data')
    button.changed.connect(set_from_data)
    widget.insert(widget.index('tolerance') + 1, button)
    widget.set_from_data_button = button

    widget.advanced.changed.connect(on_advanced_changed)
    on_advanced_changed(widget.advanced.value)


@magic_factory(
    widget_init=find_filaments_init,
    max_angle={'min': 0.1, 'max': 90},
    max_gap={'min': 0.1, 'max': 1e9},
    tolerance={'min': 0, 'max': 1e9},
    min_point_count={'min': 1, 'max': 1e6},
    max_bridge={'min': 0, 'max': 1e9},
    smoothing={'min': 0, 'max': 1e9},
)
def find_filaments(
    viewer: 'napari.Viewer',  # pyright: ignore[reportAttributeAccessIssue]
    vectors_layer: 'napari.layers.Vectors',  # pyright: ignore[reportAttributeAccessIssue]
    max_angle: float = 30,
    max_gap: float = 100,
    tolerance: float = 1,
    min_point_count: int = 3,
    show_unassigned: bool = False,
    advanced: bool = False,
    max_bridge: float = 0,
    smoothing: float = 1,
):
    """Trace filaments through the vectors and fit a contour to each.

    Every vector gets a filament ID (`rlnFilamentID`), numbered from 1;
    vectors that are not part of any filament get ID 0 and a transparent
    color (or white, see `show_unassigned`). A Shapes layer with one spline
    contour per filament is created; its features table holds the filament ID
    and the contour `length` (in world units, i.e. including the layer scale).
    The closest other filament is found as well: a vector with a different ID
    (unassigned vectors count as different from every filament). Each contour
    gets one neighbor, the closest such vector from any point of the contour:
    its filament ID (`rlnClosestFilamentID`) and the distance
    (`rlnClosestFilamentDistance`). Each vector gets the distance to its
    closest such vector (`rlnClosestFilamentPointDistance`). Distances are in
    world units.
    Filaments never branch. Vector directions are treated as bidirectional.

    max_angle: largest angular deviation (degrees) between a vector and the
        line to its neighbour, between neighbouring vectors, and at gaps.
    max_gap: largest distance between neighbouring vectors of a filament. The
        button "Estimate from data" sets it to four times the mean spacing.
    tolerance: position scatter (layer units) that is forgiven when judging
        angles; the button sets it to the mean spacing.
    min_point_count: smallest number of vectors forming a filament.
    show_unassigned: show unassigned vectors (ID 0) in white instead of
        hiding them (transparent).
    max_bridge: (advanced) largest gap bridged between filament pieces.
        0 = four times max_gap.
    smoothing: (advanced) contour smoothing; 0 passes through every point.
    """
    ids, _, contour_list = trace_filaments(
        vectors_layer.data,
        max_gap=max_gap,
        max_angle=max_angle,
        max_bridge=max_bridge or None,
        tolerance=tolerance,
        min_points=min_point_count,
        smoothing=smoothing,
        samples_per_point=_CONTOUR_SAMPLES_PER_POINT,
    )
    contours = dict(enumerate(contour_list))
    # Internally unassigned is -1 and filaments are 0-based; in the output ID 0
    # means unassigned and filaments are numbered from 1.
    ids = ids + 1
    contours = {fid + 1: contour for fid, contour in contours.items()}
    # Assign a new DataFrame (not in-place) so napari emits the features event
    # and the layer controls' feature comboboxes refresh.
    vectors_layer.features = vectors_layer.features.assign(rlnFilamentID=ids)
    vectors_layer.edge_color = _filament_colors(ids, show_unassigned)

    # Closest other filament (a vector with a different ID, unassigned
    # included), in world units. Contours share the vectors' coordinates.
    vector_coords = vectors_layer.data[:, 0, :] * vectors_layer.scale
    _, vector_distances = utils.closest_other_filament(
        points=vector_coords,
        point_ids=ids,
        vector_coords=vector_coords,
        vector_ids=ids,
    )
    vectors_layer.features = vectors_layer.features.assign(
        rlnClosestFilamentPointDistance=vector_distances
    )

    # Contours need at least two vertices to form a path
    contours = {k: v for k, v in contours.items() if len(v) >= 2}
    name = f'{vectors_layer.name} contours'
    # Remove the previous result so a stale contours layer never lingers
    if name in viewer.layers:
        viewer.layers.remove(name)
    if not contours:
        return
    contour_ids = list(contours)
    # One neighbor per contour: the closest foreign vector over all its points
    closest_ids = []
    closest_distances = []
    for i in contour_ids:
        points = contours[i] * vectors_layer.scale
        indices, distances = utils.closest_other_filament(
            points=points,
            point_ids=np.full(len(points), i),
            vector_coords=vector_coords,
            vector_ids=ids,
        )
        if np.isnan(distances).all():
            closest_ids.append(-1)
            closest_distances.append(np.nan)
            continue
        best = np.nanargmin(distances)
        closest_ids.append(int(ids[indices[best]]))
        closest_distances.append(float(distances[best]))
    viewer.add_shapes(
        [contours[i] for i in contour_ids],
        shape_type='path',
        edge_width=_CONTOUR_EDGE_WIDTH,
        edge_color=_filament_colors(np.array(contour_ids)),
        face_color='transparent',
        features=pd.DataFrame(
            {
                'rlnFilamentID': contour_ids,
                'length': [
                    utils.contour_length(contours[i], vectors_layer.scale)
                    for i in contour_ids
                ],
                'rlnClosestFilamentID': closest_ids,
                'rlnClosestFilamentDistance': closest_distances,
            }
        ),
        scale=vectors_layer.scale,
        translate=vectors_layer.translate,
        name=name,
    )
