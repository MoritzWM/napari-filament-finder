from typing import TYPE_CHECKING

from magicgui import magic_factory
from napari.utils.colormaps.vendored import cm

from . import utils

if TYPE_CHECKING:
    import napari


@magic_factory(
    cone_angle={'min': 0.1, 'max': 180},
    max_distance={'min': 0.1, 'max': 1e9},
    min_point_count={'min': 1, 'max': 1e6},
)
def find_filaments(
    vectors_layer: 'napari.layers.Vectors',  # pyright: ignore[reportAttributeAccessIssue]
    cone_angle: float=15,
    max_distance: float=100,
    min_point_count: int=2,
):
    ids = utils.assign_filament_ids(
        vectors_layer.data,
        cone_angle=cone_angle,
        max_distance=max_distance,
        min_point_count=min_point_count,
    )
    vectors_layer.features['filament_id'] = ids
    vectors_layer.edge_color = cm.get_cmap('tab20')(ids % 20)


def _on_threshold_widget_init(widget):
    @widget.img_layer.changed.connect
    def _on_layer_changed(image_layer):
        if image_layer is None:
            return
        widget.threshold.min = image_layer.data.min()
        widget.threshold.max = image_layer.data.max()

    _on_layer_changed(widget.img_layer.value)
    if widget.img_layer.value is not None:
        widget()


def find_closest_filament_init(widget):
    @widget.vectors_layer.changed.connect
    def _on_vectors_layer_changed(vectors_layer):
        if vectors_layer is None:
            widget.filament_id_column.choices = []
            return
        features = vectors_layer.features
        if features is None:
            widget.filament_id_column.choices = []
            return
        columns = list(features.columns)
        column_index = utils.find_word_index("filament", columns)
        widget.filament_id_column.choices = columns
        widget.filament_id_column.value = columns[column_index] if column_index is not None else columns[0]
    _on_vectors_layer_changed(widget.vectors_layer.value)



@magic_factory(widget_init=find_closest_filament_init, filament_id_column={"choices": [], "widget_type": "ComboBox"})
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
    vectors_layer.features['closest_filament_id'] = filament_ids[indices]
    vectors_layer.features['closest_filament_point_index'] = indices
    vectors_layer.features['closest_filament_point_distance'] = distances
