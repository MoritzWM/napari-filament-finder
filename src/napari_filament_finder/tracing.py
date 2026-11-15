"""Trace non-branching filaments through noisy oriented points.

Input is an (n, 2, 3) array: [:, 0] positions, [:, 1] bidirectional
direction vectors (the sign of a direction is meaningless).

Method: score candidate links by distance and angular consistency, link
greedily with degree <= 2 and no cycles (-> ordered path fragments), then
repeatedly bridge fragment ends across gaps using their end tangents.
"""

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree


def _normalize(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, 1e-12)


def _angle(u, v):
    """Double-sided angle (radians, 0..pi/2) between vectors."""
    cos = np.abs(np.einsum('ij,ij->i', u, v))
    return np.arccos(np.clip(cos, 0, 1))


def _one_sided_angle(u, v):
    cos = np.einsum('ij,ij->i', u, v)
    return np.arccos(np.clip(cos, -1, 1))


def _effective(angle, r, tol):
    """Angle after forgiving a perpendicular offset of up to `tol`."""
    perp = r * np.sin(np.minimum(angle, np.pi / 2))
    eff = np.arcsin(np.clip(np.maximum(perp - tol, 0) / np.maximum(r, 1e-12), 0, 1))
    return np.where(angle > np.pi / 2, angle, eff)


class _Links:
    """Degree-limited, acyclic link set (union-find)."""

    def __init__(self, n):
        self.nbr = np.full((n, 2), -1, dtype=np.int64)
        self.parent = np.arange(n)

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def try_link(self, a, b):
        if a == b or self.nbr[a, 1] != -1 or self.nbr[b, 1] != -1:
            return False
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[ra] = rb
        self.nbr[a, 0 if self.nbr[a, 0] == -1 else 1] = b
        self.nbr[b, 0 if self.nbr[b, 0] == -1 else 1] = a
        return True

    def link_greedy(self, i, j, cost):
        """Accept edges in order of increasing cost; return #accepted."""
        count = 0
        for k in np.argsort(cost, kind='stable'):
            count += self.try_link(int(i[k]), int(j[k]))
        return count

    def paths(self):
        """Ordered index arrays, one per connected chain."""
        n = len(self.nbr)
        visited = np.zeros(n, dtype=bool)
        out = []
        degree = (self.nbr >= 0).sum(axis=1)
        for start in np.flatnonzero(degree <= 1):
            if visited[start]:
                continue
            path = [int(start)]
            visited[start] = True
            prev, cur = -1, int(start)
            while True:
                nxt = next(
                    (int(x) for x in self.nbr[cur] if x >= 0 and x != prev),
                    -1,
                )
                if nxt < 0 or visited[nxt]:
                    break
                path.append(nxt)
                visited[nxt] = True
                prev, cur = cur, nxt
            out.append(np.array(path))
        return out


def _median_spacing(positions):
    """Median nearest-neighbour distance (robust to isolated outliers)."""
    dist, _ = cKDTree(positions).query(positions, k=2)
    return float(np.median(dist[:, 1]))


def _edge_costs(pos, dirs, max_gap, max_angle, sigma_r, sigma_a, tol):
    pairs = cKDTree(pos).query_pairs(max_gap, output_type='ndarray')
    if len(pairs) == 0:
        return pairs[:, 0], pairs[:, 1], np.empty(0)
    i, j = pairs[:, 0], pairs[:, 1]
    off = pos[j] - pos[i]
    r = np.linalg.norm(off, axis=1)
    off = off / np.maximum(r, 1e-12)[:, None]
    a_i = _effective(_angle(off, dirs[i]), r, tol)
    a_j = _effective(_angle(off, dirs[j]), r, tol)
    b = _angle(dirs[i], dirs[j])
    keep = (a_i < max_angle) & (a_j < max_angle) & (b < max_angle)
    i, j = i[keep], j[keep]
    cost = (
        (r[keep] / sigma_r) ** 2
        + (a_i[keep] ** 2 + a_j[keep] ** 2 + b[keep] ** 2) / sigma_a**2
    )
    return i, j, cost


def _end_tangents(path, pos, dirs, k=5):
    """Outward tangents at (start, end) of a path."""
    if len(path) == 1:
        d = dirs[path[0]]
        return -d, d
    tans = []
    for ordered in (path[::-1], path):  # outward = away from the interior
        end = ordered[-1]
        inner = ordered[-1 - min(k, len(ordered) - 1) : -1]
        t = pos[end] - pos[inner].mean(axis=0)
        t /= max(np.linalg.norm(t), 1e-12)
        d = dirs[end]
        t = t + np.sign(np.dot(d, t) or 1.0) * d
        tans.append(t / max(np.linalg.norm(t), 1e-12))
    # tans[0] belongs to path[0] (reversed order), tans[1] to path[-1]
    return tans[0], tans[1]


def _bridge_once(
    links, pos, dirs, max_bridge, max_angle, sigma_r, sigma_a, tol
):
    paths = links.paths()
    ends, tangents = [], []
    for p in paths:
        t0, t1 = _end_tangents(p, pos, dirs)
        ends += [p[0], p[-1]]
        tangents += [t0, t1]
    ends = np.array(ends)
    tangents = np.array(tangents)
    if len(ends) < 2:
        return 0
    pairs = cKDTree(pos[ends]).query_pairs(max_bridge, output_type='ndarray')
    if len(pairs) == 0:
        return 0
    a, b = pairs[:, 0], pairs[:, 1]
    off = pos[ends[b]] - pos[ends[a]]
    r = np.linalg.norm(off, axis=1)
    off = off / np.maximum(r, 1e-12)[:, None]
    ang_a = _effective(_one_sided_angle(off, tangents[a]), r, tol)
    ang_b = _effective(_one_sided_angle(-off, tangents[b]), r, tol)
    keep = (ang_a < max_angle) & (ang_b < max_angle)
    a, b = a[keep], b[keep]
    cost = (r[keep] / sigma_r) ** 2 + (ang_a[keep] ** 2 + ang_b[keep] ** 2) / (
        sigma_a**2
    )
    return links.link_greedy(ends[a], ends[b], cost)


def fit_contour(points, smoothing=0.0, samples_per_point=5):
    """Fit a smoothing spline through ordered points -> (M, 3) samples."""
    points = np.asarray(points, dtype=float)
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9]
    points = points[keep]
    m = len(points)
    if m < 2:
        return points
    k = min(3, m - 1)
    tck, _ = splprep(points.T, k=k, s=smoothing * m)
    u = np.linspace(0, 1, max(m * samples_per_point, 2))
    return np.column_stack(splev(u, tck))


def trace_filaments(
    data,
    max_gap=None,
    max_angle=30.0,
    max_bridge=None,
    tolerance=None,
    min_points=3,
    max_deviation=0.0,
    smoothing=0.0,
    samples_per_point=5,
):
    """Trace filaments through oriented points.

    Parameters:
    - data: (n, 2, 3) array; [:, 0] positions, [:, 1] bidirectional
      direction vectors.
    - max_gap: largest distance between directly linked points.
      Default: 4 * median nearest-neighbour spacing.
    - max_angle: degrees; largest angular deviation (point direction vs
      link, direction vs direction, tangent vs bridge).
    - max_bridge: largest gap bridged between fragment ends.
      Default: 4 * max_gap.
    - tolerance: perpendicular offset forgiven when judging angles
      (position noise). Default: the median spacing.
    - min_points: smaller filaments are discarded (ids -1).
    - max_deviation: if > 0, points farther than this from the fitted
      contour are unassigned.
    - smoothing: spline smoothing per point (0 = interpolate).
    - samples_per_point: contour samples per point.

    Returns:
    - ids: (n,) int, filament id (0..K-1) or -1 if unassigned.
    - paths: list of K index arrays, ordered along each filament.
    - contours: list of K (M, 3) arrays (spline samples).
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 3 or data.shape[1:] != (2, 3):
        raise ValueError('data must have shape (n, 2, 3)')
    n = len(data)
    pos, dirs = data[:, 0], _normalize(data[:, 1])
    if n < 2:
        return np.full(n, -1), [], []

    ang = np.deg2rad(max_angle)
    sigma_a = ang / 2
    spacing = _median_spacing(pos)
    max_gap = 4 * spacing if max_gap is None else max_gap
    max_bridge = 4 * max_gap if max_bridge is None else max_bridge
    sigma_r = spacing
    tol = spacing if tolerance is None else tolerance

    links = _Links(n)
    i, j, cost = _edge_costs(pos, dirs, max_gap, ang, sigma_r, sigma_a, tol)
    links.link_greedy(i, j, cost)
    while _bridge_once(
        links, pos, dirs, max_bridge, ang, max_bridge / 2, sigma_a, tol
    ):
        pass

    ids = np.full(n, -1, dtype=np.int64)
    paths, contours = [], []
    for path in links.paths():
        if len(path) < min_points:
            continue
        contour = fit_contour(pos[path], smoothing, samples_per_point)
        if max_deviation > 0 and len(contour) > 1:
            dist, _ = cKDTree(contour).query(pos[path])
            path = path[dist <= max_deviation]
            if len(path) < min_points:
                continue
            contour = fit_contour(pos[path], smoothing, samples_per_point)
        ids[path] = len(paths)
        paths.append(path)
        contours.append(contour)
    _reattach(ids, pos, dirs, contours, paths, max_angle=ang, tol=3 * tol)
    return ids, paths, contours


def _reattach(ids, pos, dirs, contours, paths, max_angle, tol):
    """Assign leftover points lying on (and aligned with) a contour.

    Updates `ids` in place; the point is inserted into the path at the
    position of its nearest contour sample.
    """
    free = np.flatnonzero(ids < 0)
    if len(free) == 0 or not contours:
        return
    samples = np.concatenate(contours)
    owner = np.concatenate([np.full(len(c), k) for k, c in enumerate(contours)])
    tang = np.concatenate([_tangents(c) for c in contours])
    dist, idx = cKDTree(samples).query(pos[free])
    cos = np.abs(np.einsum('ij,ij->i', dirs[free], tang[idx]))
    ok = (dist <= tol) & (np.arccos(np.clip(cos, 0, 1)) < max_angle)
    for pt, k in zip(free[ok], owner[idx[ok]], strict=True):
        ids[pt] = k
    for k, path in enumerate(paths):
        members = np.flatnonzero(ids == k)
        if len(members) == len(path):
            continue
        # order all members by arc position of nearest contour sample
        c = contours[k]
        _, near = cKDTree(c).query(pos[members])
        paths[k] = members[np.argsort(near, kind='stable')]


def _tangents(contour):
    if len(contour) < 2:
        return np.tile([1.0, 0, 0], (len(contour), 1))
    return _normalize(np.gradient(contour, axis=0))
