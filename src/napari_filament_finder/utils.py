import numpy as np
import numpy.ma as ma
from scipy.spatial.distance import pdist, squareform


def find_points_in_cone(
    vector_data: np.ndarray, cone_angle: float, max_distance: float
):
    """
    Find all points that fall within a cone projected from each point.

    Parameters:
    - xyz: (N, 3) array of XYZ coordinates
    - vecs: (N, 3) array of direction vectors (should be normalized)
    - cone_angle: half-angle of the cone in degrees
    - max_distance: maximum distance along the cone axis

    Returns:
    - cone_points: (N,) array where each element is a list of indices within the cone
    """
    if vector_data.ndim != 3 or vector_data.shape[1:] != (2, 3):
        raise ValueError('vector_data needs to be of shape (N, 2, 3)')
    cone_angle_rad = np.deg2rad(cone_angle)
    cone_points = [[] for _ in range(len(vector_data[:, 0, :]))]

    for i, (point, direction) in enumerate(
        zip(vector_data[:, 0, :], vector_data[:, 1, :], strict=True)
    ):
        # Normalize direction vector
        direction_norm_pos = direction / np.linalg.norm(direction)
        direction_norm_neg = -direction_norm_pos

        # Vector from this point to all other points
        diff = vector_data[:, 0, :] - point
        distances = np.linalg.norm(diff, axis=1)

        # Angle between direction and each point
        dot_products_pos = np.dot(diff, direction_norm_pos)
        dot_products_neg = np.dot(diff, direction_norm_neg)
        cos_angles = dot_products_pos / (
            distances + 1e-10
        )  # avoid division by zero
        cos_angles_neg = dot_products_neg / (
            distances + 1e-10
        )  # avoid division by zero

        # Find points within cone: angle < cone_angle AND distance < max_distance AND dot product > 0
        in_cone_pos = (
            (np.arccos(np.clip(cos_angles, -1, 1)) < cone_angle_rad)
            & (distances < max_distance)
            & (dot_products_pos > 0)
        )
        in_cone_neg = (
            (np.arccos(np.clip(cos_angles_neg, -1, 1)) < cone_angle_rad)
            & (distances < max_distance)
            & (dot_products_neg > 0)
        )
        cone_points[i] = np.where(in_cone_pos | in_cone_neg)[0].tolist()

    return cone_points


def assign_filament_ids(
    vector_data: np.ndarray,
    cone_angle: float,
    max_distance: float,
    min_point_count: int,
):
    if vector_data.ndim != 3 or vector_data.shape[1:] != (2, 3):
        raise ValueError('vector_data needs to be of shape (N, 2, 3)')
    cone_points = find_points_in_cone(
        vector_data, cone_angle=cone_angle, max_distance=max_distance
    )

    filaments: list[set[int]] = []
    for i, points in enumerate(cone_points):
        if len(points) == 0:
            continue
        # Check if this point (i) or it's neighbors (points) are already part of an existing filament
        for filament in filaments:
            if i in filament or any(p in filament for p in points):
                filament.update(points)
                break
        else:
            filaments.append({i} | set(points))

    # De-duplicate and merge
    for i, filament in enumerate(filaments):
        for other_filament in filaments[i + 1 :]:
            if filament.intersection(other_filament):
                filament.update(other_filament)
                other_filament.clear()

    # Remove filaments with less than min_point_count points
    for i in reversed(range(len(filaments))):
        if len(filaments[i]) < min_point_count:
            filaments.pop(i)

    filament_ids = np.full(len(vector_data), -1, dtype=int)

    for i in range(len(filament_ids)):
        for j, filament in enumerate(filaments):
            if i in filament:
                filament_ids[i] = j
                break
    return filament_ids


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
