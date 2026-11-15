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


def assign_closest_neighbor(
    vectors_or_coords: np.ndarray,
    filament_ids: np.ndarray,
):
    if vectors_or_coords.ndim == 3 and vectors_or_coords.shape[1:] == (2, 3):
        coords = vectors_or_coords[:, 0, :]
    elif vectors_or_coords.ndim == 2 and vectors_or_coords.shape[1] == 3:
        coords = vectors_or_coords[:, :]
    else:
        raise ValueError(
            f'Vectors/coords: invalid shape, must be (N, 2, 3) or (N, 3) but is {vectors_or_coords.shape}'
        )
    if len(filament_ids) != len(coords):
        raise ValueError(
            f'Filament IDs must be of the same length as coords ({len(filament_ids)}) != {len(coords)}'
        )
    closest_neighbor_indices = np.zeros(len(coords), dtype=int)
    closest_neighbor_distances = np.zeros(len(coords), dtype=float)

    distance_matrix = squareform(pdist(coords))
    for filament_id in np.unique(filament_ids):
        filament_idxs = np.where(filament_ids == filament_id)[0]
        distance_matrix_mask = np.zeros_like(distance_matrix)
        distance_matrix_mask[filament_ids == filament_id, :] = 1
        distance_matrix_mask[:, filament_ids != filament_id] = 1
        distance_matrix_masked = ma.masked_array(
            distance_matrix, distance_matrix_mask
        )
        min_distances = distance_matrix_masked.min(axis=0)
        min_distance_idx = distance_matrix_masked.argmin(axis=0)[
            np.nonzero(~min_distances.mask)
        ]
        closest_neighbor_indices[filament_idxs] = min_distance_idx
        closest_neighbor_distances[filament_idxs] = min_distances.compressed()

    return closest_neighbor_indices, closest_neighbor_distances


def find_word_index(word: str, strings: list[str]) -> int | None:
    """Find the index of the first string containing the word (case-insensitive)."""
    word_lower = word.lower()
    for i, s in enumerate(strings):
        if word_lower in s.lower():
            return i
    return None


def contour_length(contour: np.ndarray, scale=None) -> float:
    """
    Length of a contour polyline. With `scale` (per-axis, e.g. a layer's
    scale) the length is measured in world units rather than data units.
    """
    contour = np.asarray(contour, dtype=float)
    if scale is not None:
        contour = contour * np.asarray(scale, dtype=float)
    return float(np.linalg.norm(np.diff(contour, axis=0), axis=1).sum())
