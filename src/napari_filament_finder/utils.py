import numpy as np
import numpy.ma as ma
from scipy.interpolate import splev, splprep
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform


def find_points_in_cone(
    vector_data: np.ndarray,
    cone_angle: float,
    max_distance: float,
    perpendicular_tolerance: float = 0.0,
):
    """
    Find all points that fall within a cone projected from each point.

    Parameters:
    - xyz: (N, 3) array of XYZ coordinates
    - vecs: (N, 3) array of direction vectors (should be normalized)
    - cone_angle: half-angle of the cone in degrees
    - max_distance: maximum distance along the cone axis
    - perpendicular_tolerance: a point is also accepted if it lies within this
      distance of the axis line (in front of or behind the point), even when
      outside the cone angle. 0 disables this. Helps at short range, where a
      small lateral offset corresponds to a large angle.

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
        # Distance of each point from the axis line through `point`
        perpendicular = np.sqrt(
            np.maximum(distances**2 - dot_products_pos**2, 0.0)
        )
        near_axis = perpendicular <= perpendicular_tolerance

        in_cone_pos = (
            (
                (np.arccos(np.clip(cos_angles, -1, 1)) < cone_angle_rad)
                | near_axis
            )
            & (distances < max_distance)
            & (dot_products_pos > 0)
        )
        in_cone_neg = (
            (
                (np.arccos(np.clip(cos_angles_neg, -1, 1)) < cone_angle_rad)
                | near_axis
            )
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
    perpendicular_tolerance: float = 0.0,
):
    if vector_data.ndim != 3 or vector_data.shape[1:] != (2, 3):
        raise ValueError('vector_data needs to be of shape (N, 2, 3)')
    cone_points = find_points_in_cone(
        vector_data,
        cone_angle=cone_angle,
        max_distance=max_distance,
        perpendicular_tolerance=perpendicular_tolerance,
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


def order_points_along_filament(coords: np.ndarray) -> np.ndarray:
    """Return indices that order (N, 3) points along their principal axis."""
    centered = coords - coords.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return np.argsort(centered @ vt[0])


def _ordered_unique(coords: np.ndarray) -> np.ndarray:
    """Order points along the filament and drop consecutive duplicates."""
    ordered = coords[order_points_along_filament(coords)]
    keep = np.r_[True, np.any(np.diff(ordered, axis=0) != 0, axis=1)]
    return ordered[keep]


def _fit_spline(ordered: np.ndarray, smoothing: float, n_samples: int):
    n = len(ordered)
    tck, _ = splprep(ordered.T, k=min(3, n - 1), s=smoothing * n)
    return np.stack(splev(np.linspace(0, 1, n_samples), tck), axis=1)


def on_track_mask(
    ordered: np.ndarray, max_deviation: float, min_points: int = 4
) -> np.ndarray:
    """
    Flag points to use in the fit: iteratively exclude the point that lies
    furthest from the curve through all *other* points (leave-one-out), as
    long as that distance exceeds `max_deviation`. Excluded points are only
    ignored for fitting; the input is never modified. The two end points of
    the filament are always kept.

    Returns a boolean mask over `ordered` (True = use for fitting).
    """
    keep = np.ones(len(ordered), dtype=bool)
    while keep.sum() > min_points:
        idx = np.flatnonzero(keep)
        pts = ordered[idx]
        # Endpoints are never judged: the curve through the other points does
        # not extend to them, so their deviation is undefined.
        deviation = np.zeros(len(pts))
        for j in range(1, len(pts) - 1):
            others = np.delete(pts, j, axis=0)
            curve = _fit_spline(others, 0.0, len(others) * 20)
            deviation[j] = cKDTree(curve).query(pts[j])[0]
        worst = int(np.argmax(deviation))
        if deviation[worst] <= max_deviation:
            break
        keep[idx[worst]] = False
    return keep


def fit_filament_contour(
    coords: np.ndarray,
    smoothing: float = 0.0,
    samples_per_point: int = 5,
    max_deviation: float = 0.0,
) -> np.ndarray:
    """
    Fit a smooth 3D contour (polyline of spline samples) through the points
    of one filament.

    Parameters:
    - coords: (N, 3) array of points belonging to a single filament
    - smoothing: spline smoothing, in units of squared distance per point
      (0 = interpolate through all points)
    - samples_per_point: number of contour vertices generated per input point
    - max_deviation: points further than this from the curve through the
      other points are ignored for the fit (0 = use all points)

    Returns:
    - (M, 3) array of contour vertices, ordered along the filament
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError('coords needs to be of shape (N, 3)')
    ordered = _ordered_unique(coords)
    if max_deviation > 0 and len(ordered) > 4:
        ordered = ordered[on_track_mask(ordered, max_deviation)]
    n = len(ordered)
    if n < 2:
        return ordered
    return _fit_spline(ordered, smoothing, max(2, n * samples_per_point))


def fit_filament_contours(
    coords: np.ndarray,
    filament_ids: np.ndarray,
    smoothing: float = 0.0,
    samples_per_point: int = 5,
    max_deviation: float = 0.0,
) -> dict[int, np.ndarray]:
    """Fit a contour per filament ID; IDs < 0 (unassigned) are skipped."""
    if len(filament_ids) != len(coords):
        raise ValueError('filament_ids must have the same length as coords')
    return {
        int(fid): fit_filament_contour(
            coords[filament_ids == fid],
            smoothing,
            samples_per_point,
            max_deviation,
        )
        for fid in np.unique(filament_ids)
        if fid >= 0
    }
