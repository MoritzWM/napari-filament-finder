import numpy as np
import numpy.ma as ma
from scipy.interpolate import splev, splprep
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import (
    connected_components,
    dijkstra,
    minimum_spanning_tree,
)
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform


def _cone_pairs(
    vector_data: np.ndarray,
    cone_angle: float,
    max_distance: float,
    perpendicular_tolerance: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized neighbour search. Returns index arrays (i, j), sorted by i,
    meaning point j lies in the (double-sided) cone of point i.
    """
    if vector_data.ndim != 3 or vector_data.shape[1:] != (2, 3):
        raise ValueError('vector_data needs to be of shape (N, 2, 3)')
    points = vector_data[:, 0, :]
    directions = vector_data[:, 1, :]
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.where(norms == 0, 1, norms)

    tree = cKDTree(points)
    pairs = tree.sparse_distance_matrix(
        tree, max_distance, output_type='ndarray'
    )
    i, j, distance = pairs['i'], pairs['j'], pairs['v']
    keep = (distance < max_distance) & (i != j)
    i, j, distance = i[keep], j[keep], distance[keep]

    along = np.einsum('ij,ij->i', points[j] - points[i], directions[i])
    # abs(): the cone is double-sided (forward and backward along the vector)
    cos_angle = np.abs(along) / (distance + 1e-10)
    angle_ok = np.arccos(np.clip(cos_angle, -1, 1)) < np.deg2rad(cone_angle)
    if perpendicular_tolerance > 0:
        perpendicular = np.sqrt(np.maximum(distance**2 - along**2, 0.0))
        angle_ok |= perpendicular <= perpendicular_tolerance
    keep = angle_ok & (along != 0)
    i, j = i[keep], j[keep]
    order = np.argsort(i, kind='stable')
    return i[order], j[order]


def find_points_in_cone(
    vector_data: np.ndarray,
    cone_angle: float,
    max_distance: float,
    perpendicular_tolerance: float = 0.0,
):
    """
    Find all points that fall within a cone projected from each point.

    Parameters:
    - vector_data: (N, 2, 3) array of (position, direction) pairs
    - cone_angle: half-angle of the cone in degrees
    - max_distance: maximum distance along the cone axis
    - perpendicular_tolerance: a point is also accepted if it lies within this
      distance of the axis line (in front of or behind the point), even when
      outside the cone angle. 0 disables this. Helps at short range, where a
      small lateral offset corresponds to a large angle.

    Returns:
    - cone_points: list with, for each point, the indices within its cone
    """
    i, j = _cone_pairs(
        vector_data, cone_angle, max_distance, perpendicular_tolerance
    )
    counts = np.bincount(i, minlength=len(vector_data))
    return [
        chunk.tolist() for chunk in np.split(j, np.cumsum(counts)[:-1])
    ]


def estimate_spacing(vector_data: np.ndarray) -> float:
    """Mean distance from each point to its nearest neighbour."""
    points = vector_data[:, 0, :] if vector_data.ndim == 3 else vector_data
    if len(points) < 2:
        return 0.0
    return float(cKDTree(points).query(points, k=2)[0][:, 1].mean())


def _relabel(ids: np.ndarray) -> np.ndarray:
    """Relabel IDs >= 0 to 0..K-1 in order of first appearance; keep -1."""
    out = np.full(len(ids), -1, dtype=int)
    valid = ids >= 0
    _, first, inverse = np.unique(
        ids[valid], return_index=True, return_inverse=True
    )
    rank = np.empty(len(first), dtype=int)
    rank[np.argsort(first)] = np.arange(len(first))
    out[valid] = rank[inverse]
    return out


def assign_filament_ids(
    vector_data: np.ndarray,
    cone_angle: float,
    max_distance: float,
    min_point_count: int,
    perpendicular_tolerance: float = 0.0,
):
    """
    Group points into filaments: connected components of the graph that links
    each point to the points in its cone. Points in no (large enough)
    component get ID -1.
    """
    i, j = _cone_pairs(
        vector_data, cone_angle, max_distance, perpendicular_tolerance
    )
    n = len(vector_data)
    graph = coo_matrix((np.ones(len(i), dtype=bool), (i, j)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)

    sizes = np.bincount(labels)
    linked = np.zeros(n, dtype=bool)  # points with at least one neighbour
    linked[i] = True
    linked[j] = True
    valid = linked & (sizes[labels] >= min_point_count)
    return _relabel(np.where(valid, labels, -1))


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
    """
    Return indices that order (N, 3) points along the filament they form.

    The order follows the chain of nearest neighbours (the longest path of the
    minimum spanning tree), so it works for curved filaments, unlike a
    projection onto a single axis. Points off that path are slotted in next to
    their nearest path point.
    """
    n = len(coords)
    if n < 3:
        return np.argsort(coords @ np.ones(3)) if n else np.array([], int)
    dist = squareform(pdist(coords))
    dist[(dist == 0) & ~np.eye(n, dtype=bool)] = 1e-12  # keep duplicate edges
    tree = minimum_spanning_tree(csr_matrix(dist))
    tree = tree + tree.T
    first = int(np.argmax(dijkstra(tree, indices=0, unweighted=False)))
    from_first, pred = dijkstra(tree, indices=first, return_predecessors=True)
    last = int(np.argmax(from_first))
    path = [last]
    while path[-1] != first:
        path.append(int(pred[path[-1]]))
    path = np.array(path[::-1])

    key = np.zeros(n)
    key[path] = np.arange(len(path))
    off = np.setdiff1d(np.arange(n), path)
    if len(off):
        path_pts = coords[path]
        _, nearest = cKDTree(path_pts).query(coords[off])
        ahead = path_pts[np.minimum(nearest + 1, len(path) - 1)]
        behind = path_pts[np.maximum(nearest - 1, 0)]
        tangent = ahead - behind
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
        along = np.einsum('ij,ij->i', coords[off] - path_pts[nearest], tangent)
        step = np.median(np.linalg.norm(np.diff(path_pts, axis=0), axis=1))
        key[off] = nearest + np.clip(along / max(step, 1e-12), -0.49, 0.49)
    return np.argsort(key, kind='stable')


def _ordered_unique(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Order points along the filament and drop consecutive duplicates.

    Returns the ordered points and their indices into `coords`.
    """
    order = order_points_along_filament(coords)
    ordered = coords[order]
    keep = np.r_[True, np.any(np.diff(ordered, axis=0) != 0, axis=1)]
    return ordered[keep], order[keep]


def _fit_spline(
    ordered: np.ndarray, smoothing: float, n_samples: int, degree: int = 3
):
    n = len(ordered)
    tck, _ = splprep(ordered.T, k=min(degree, n - 1), s=smoothing * n)
    return np.stack(splev(np.linspace(0, 1, n_samples), tck), axis=1)


def _typical_spacing(ordered: np.ndarray) -> float:
    """Median distance between consecutive points along the filament."""
    if len(ordered) < 2:
        return 0.0
    return float(np.median(np.linalg.norm(np.diff(ordered, axis=0), axis=1)))


def _extend_contour(
    contour: np.ndarray, length: float, span: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extend a contour straight past both ends by `length`, along the end
    tangents. Returns the extra samples and the unit tangent of each.
    """
    if length <= 0 or len(contour) < 2:
        return np.empty((0, 3)), np.empty((0, 3))
    span = max(1, min(span, len(contour) - 1))
    extra, tangents = [], []
    for end, other in ((contour[-1], contour[-1 - span]), (contour[0], contour[span])):
        direction = end - other
        norm = np.linalg.norm(direction)
        if norm == 0:
            continue
        direction = direction / norm
        steps = max(2, int(np.ceil(length / max(norm / span, 1e-12))))
        t = np.linspace(0, length, steps + 1)[1:, None]
        extra.append(end + t * direction)
        tangents.append(np.tile(direction, (steps, 1)))
    if not extra:
        return np.empty((0, 3)), np.empty((0, 3))
    return np.concatenate(extra), np.concatenate(tangents)


def _distance_to_contour(
    points: np.ndarray, contour: np.ndarray, extension: float = 0.0
) -> np.ndarray:
    """Distance from each point to the contour (plus straight end extensions)."""
    extra, _ = _extend_contour(contour, extension, span=10)
    samples = np.concatenate([contour, extra]) if len(extra) else contour
    return cKDTree(samples).query(points)[0]


def on_track_mask(
    ordered: np.ndarray, max_deviation: float, min_points: int = 4
) -> np.ndarray:
    """
    Flag points that do not fit the filament (candidates for being ignored).

    Single pass, deliberately generous: a good point next to an outlier can be
    flagged too. `fit_filament_contour` therefore fits the contour on the
    unflagged points and takes back every candidate that lies within
    `max_deviation` of it.

    - Interior points are compared to a cubic through up to two other points
      on each side along the chain. The window is symmetric, so an outlier in
      it pulls the curve at the point less than it deviates itself.
    - End points cannot be compared that way. Extrapolating a line misjudges
      curved filaments, so an end point is flagged by how much further it is
      from its nearest other point than the usual point spacing.

    Returns a boolean mask over `ordered` (True = not flagged); at least
    `min_points` points stay unflagged.
    """
    n = len(ordered)
    deviation = np.zeros(n)
    for j in range(1, n - 1):
        reach = min(2, j, n - 1 - j)
        window = np.concatenate(
            [ordered[j - reach : j], ordered[j + 1 : j + 1 + reach]]
        )
        curve = _fit_spline(window, 0.0, 60)
        deviation[j] = cKDTree(curve).query(ordered[j])[0]
    spacing = _typical_spacing(ordered)
    tree = cKDTree(ordered)
    for end in (0, n - 1):
        gap = tree.query(ordered[end], k=2)[0][1]
        deviation[end] = max(0.0, gap - spacing)
    # Flag the worst offenders first, but always leave `min_points` unflagged
    # (in a short filament every point can look off-track near an outlier)
    keep = np.ones(n, dtype=bool)
    for j in np.argsort(-deviation):
        if deviation[j] <= max_deviation or keep.sum() <= min_points:
            break
        keep[j] = False
    return keep


def fit_filament_contour(
    coords: np.ndarray,
    smoothing: float = 0.0,
    samples_per_point: int = 5,
    max_deviation: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a smooth 3D contour (polyline of spline samples) through the points
    of one filament.

    Parameters:
    - coords: (N, 3) array of points belonging to a single filament
    - smoothing: spline smoothing, in units of squared distance per point
      (0 = interpolate through all points)
    - samples_per_point: number of contour vertices generated per input point
    - max_deviation: points that do not fit the filament are ignored for the
      fit (0 = use all points). Candidates are found by comparing each point
      to the curve through the others; the contour is then fitted on the
      remaining points, and every candidate within `max_deviation` of that
      contour (or of its straight extension past the ends) is taken back.

    Returns:
    - (M, 3) array of contour vertices, ordered along the filament
    - (N,) boolean mask over `coords`; False for points ignored in the fit
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError('coords needs to be of shape (N, 3)')
    ordered, index = _ordered_unique(coords)
    used = np.ones(len(coords), dtype=bool)
    if max_deviation > 0 and len(ordered) > 4:
        on_track = on_track_mask(ordered, max_deviation)
        if not on_track.all():
            inliers = ordered[on_track]
            contour = _fit_spline(
                inliers, 0.0, max(2, len(inliers) * 10)
            )
            spacing = _typical_spacing(inliers)
            # Second chance: judge candidates against the final contour
            dist = _distance_to_contour(
                ordered[~on_track], contour, extension=2 * spacing
            )
            on_track[~on_track] = dist <= max_deviation
        used[index[~on_track]] = False
        ordered = ordered[on_track]
    n = len(ordered)
    if n < 2:
        return ordered, used
    contour = _fit_spline(ordered, smoothing, max(2, n * samples_per_point))
    return contour, used


def fit_filament_contours(
    coords: np.ndarray,
    filament_ids: np.ndarray,
    smoothing: float = 0.0,
    samples_per_point: int = 5,
    max_deviation: float = 0.0,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """
    Fit a contour per filament ID; IDs < 0 (unassigned) are skipped.

    Returns the contours by filament ID and an (N,) boolean mask over
    `coords` that is False for points ignored during fitting.
    """
    if len(filament_ids) != len(coords):
        raise ValueError('filament_ids must have the same length as coords')
    contours = {}
    used = np.ones(len(coords), dtype=bool)
    for fid in np.unique(filament_ids):
        if fid < 0:
            continue
        member = filament_ids == fid
        contours[int(fid)], used[member] = fit_filament_contour(
            coords[member], smoothing, samples_per_point, max_deviation
        )
    return contours, used


def _contour_tangents(contour: np.ndarray, span: int) -> np.ndarray:
    """
    Unit tangents of a contour, estimated as the chord over +-`span` samples.
    A wide span keeps the wiggles of a contour through noisy points from
    dominating the direction.
    """
    n = len(contour)
    ahead = contour[np.minimum(np.arange(n) + span, n - 1)]
    behind = contour[np.maximum(np.arange(n) - span, 0)]
    tangent = ahead - behind
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    return tangent / np.where(norm == 0, 1, norm)


def _assign_to_contours(
    vector_data: np.ndarray,
    contours: dict[int, np.ndarray],
    current_ids: np.ndarray,
    cone_angle: float,
    tolerance: float,
    samples_per_point: int = 10,
    n_candidates: int = 8,
) -> np.ndarray:
    """
    Assign each point to the nearest contour that is within `tolerance` and
    whose tangent is within `cone_angle` of the point's vector (either
    direction). Points matching no contour get -1.

    Assignments are sticky: a point that is within `tolerance` of the contour
    of its current filament keeps it without the angle test, since the
    tangent of a short or noisy contour is unreliable. The angle test decides
    only whether a point joins a *different* filament or an unassigned point
    joins one.

    Each contour is extended straight past its ends by two point spacings, so
    that points continuing the filament across a small gap can join it.
    """
    ids = np.full(len(vector_data), -1, dtype=int)
    if not contours:
        return ids
    tangent_span = 3 * samples_per_point
    all_samples, all_tangents, all_owner = [], [], []
    for fid, contour in contours.items():
        tangent = _contour_tangents(contour, tangent_span)
        length = np.linalg.norm(np.diff(contour, axis=0), axis=1).sum()
        spacing = length / max(len(contour) - 1, 1) * samples_per_point
        extra, extra_tangent = _extend_contour(
            contour, 2 * spacing, tangent_span
        )
        all_samples += [contour, extra]
        all_tangents += [tangent, extra_tangent]
        all_owner.append(np.full(len(contour) + len(extra), fid))
    samples = np.concatenate(all_samples)
    tangents = np.concatenate(all_tangents)
    owner = np.concatenate(all_owner)
    points = vector_data[:, 0, :]
    norms = np.linalg.norm(vector_data[:, 1, :], axis=1, keepdims=True)
    directions = vector_data[:, 1, :] / np.where(norms == 0, 1, norms)

    k = min(n_candidates, len(samples))
    dist, idx = cKDTree(samples).query(
        points, k=k, distance_upper_bound=tolerance
    )
    dist, idx = dist.reshape(len(points), k), idx.reshape(len(points), k)
    found = np.isfinite(dist)
    idx = np.where(found, idx, 0)  # placeholder for missing neighbours
    cos = np.abs(np.einsum('nkd,nd->nk', tangents[idx], directions))
    ok = found & (cos >= np.cos(np.deg2rad(cone_angle)))
    stays = found & (owner[idx] == current_ids[:, None])
    ok = np.where(stays.any(axis=1, keepdims=True), stays, ok)
    has_match = ok.any(axis=1)
    first = ok.argmax(axis=1)
    chosen = idx[np.arange(len(points)), first]
    ids[has_match] = owner[chosen][has_match]
    return ids


def refine_filament_ids(
    vector_data: np.ndarray,
    filament_ids: np.ndarray,
    cone_angle: float,
    tolerance: float,
    min_point_count: int = 2,
    smoothing: float = 0.0,
    samples_per_point: int = 5,
    max_iterations: int = 5,
    join_tolerance: float | None = None,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """
    Refine a filament assignment using the fitted contours.

    Repeats until stable: fit a contour per filament, then reassign every
    point (including unassigned ones) to the nearest contour within
    `join_tolerance` whose tangent agrees with the point's vector. Points that are
    near no contour get -1, as do filaments left with fewer than
    `min_point_count` points.

    Two distances are used: `tolerance` decides which points are kept for
    fitting a contour (points further off are ignored), `join_tolerance`
    (default: `tolerance`) how close a point must be to a contour to join it.
    If both are 0 only the contours are fitted and nothing is reassigned.

    Returns the final IDs and the contours (keyed by those IDs). Points that
    were ignored in the final fit are returned as -1.
    """
    coords = vector_data[:, 0, :]
    ids = _relabel(np.asarray(filament_ids))
    if join_tolerance is None:
        join_tolerance = tolerance
    if tolerance > 0 or join_tolerance > 0:
        for _ in range(max_iterations):
            contours, used = fit_filament_contours(
                coords,
                ids,
                smoothing=smoothing,
                samples_per_point=max(samples_per_point, 10),
                max_deviation=tolerance,
            )
            new_ids = _assign_to_contours(
                vector_data,
                contours,
                ids,
                cone_angle,
                join_tolerance,
                max(samples_per_point, 10),
            )
            sizes = np.bincount(new_ids[new_ids >= 0], minlength=1)
            too_small = np.isin(new_ids, np.flatnonzero(sizes < min_point_count))
            new_ids[too_small] = -1
            new_ids = _relabel(new_ids)
            converged = np.array_equal(new_ids, ids)
            ids = new_ids
            if converged:
                break
    contours, used = fit_filament_contours(
        coords,
        ids,
        smoothing=smoothing,
        samples_per_point=samples_per_point,
        max_deviation=tolerance,
    )
    ids = np.where(used, ids, -1)
    if tolerance > 0 or join_tolerance > 0:
        # Last chance for points left out: join the contour they lie on
        rejoined = _assign_to_contours(
            vector_data,
            contours,
            ids,
            cone_angle,
            join_tolerance,
            samples_per_point,
        )
        ids = np.where(ids < 0, rejoined, ids)
    return ids, contours
