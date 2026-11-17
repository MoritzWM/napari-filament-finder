import numpy as np
import numpy.ma as ma
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform


def estimate_spacing(vector_data: np.ndarray) -> float:
    """Mean distance from each point to its nearest neighbour."""
    points = vector_data[:, 0, :] if vector_data.ndim == 3 else vector_data
    if len(points) < 2:
        return 0.0
    return float(cKDTree(points).query(points, k=2)[0][:, 1].mean())


def closest_other_filament(
    points: np.ndarray,
    point_ids: np.ndarray,
    vector_coords: np.ndarray,
    vector_ids: np.ndarray,
):
    """
    For each point, find the closest vector that belongs to a *different*
    filament (ID 0, i.e. unassigned, counts as different from every filament).

    Returns `(indices, distances)`: the index into `vector_coords` and the
    distance. Points without any candidate get index -1 and distance NaN.
    """
    points = np.asarray(points, dtype=float)
    point_ids = np.asarray(point_ids)
    vector_coords = np.asarray(vector_coords, dtype=float)
    vector_ids = np.asarray(vector_ids)
    if len(point_ids) != len(points):
        raise ValueError('point_ids must have one entry per point')
    if len(vector_ids) != len(vector_coords):
        raise ValueError('vector_ids must have one entry per vector')
    indices = np.full(len(points), -1, dtype=int)
    distances = np.full(len(points), np.nan, dtype=float)
    for filament_id in np.unique(point_ids):
        candidates = np.where(vector_ids != filament_id)[0]
        if len(candidates) == 0:
            continue
        selected = point_ids == filament_id
        dist, nearest = cKDTree(vector_coords[candidates]).query(points[selected])
        indices[selected] = candidates[nearest]
        distances[selected] = dist
    return indices, distances


def contour_length(contour: np.ndarray, scale=None) -> float:
    """
    Length of a contour polyline. With `scale` (per-axis, e.g. a layer's
    scale) the length is measured in world units rather than data units.
    """
    contour = np.asarray(contour, dtype=float)
    if scale is not None:
        contour = contour * np.asarray(scale, dtype=float)
    return float(np.linalg.norm(np.diff(contour, axis=0), axis=1).sum())
