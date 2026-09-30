"""Original-grid connectivity, exact component distance trees, and proximity groups."""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


def physical_grid(mask, spacing):
    spacing = np.asarray(spacing, dtype=float)
    if len(spacing) != mask.ndim or np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('Physical post-processing requires positive finite spacing for every spatial axis.')
    if mask.ndim == 3 and mask.shape[0] == 1 and spacing[0] == 999:
        return mask[0], spacing[1:]
    if mask.ndim not in (2, 3):
        raise ValueError('Adaptive post-processing requires a 2D or 3D physical grid.')
    return mask, spacing


def effective_connectivity(ndim, connectivity):
    if ndim == 2 and connectivity in (4, 6, 8, 18, 26):
        return 4 if connectivity in (4, 6) else 8
    if ndim == 3 and connectivity in (6, 18, 26):
        return connectivity
    raise ValueError(f'Connectivity {connectivity} is invalid for a {ndim}D grid; '
                     'saved 8-connectivity policies require genuine 2D data. Use full 3D connectivity (26) or refit.')


def component_volumes(mask, spacing, connectivity=26):
    original_shape = mask.shape
    mask, spacing = physical_grid(mask, spacing)
    connectivity = effective_connectivity(mask.ndim, connectivity)
    rank = {4: 1, 8: 2, 6: 1, 18: 2, 26: 3}[connectivity]
    components, count = ndi.label(mask, ndi.generate_binary_structure(mask.ndim, rank))
    volumes = np.bincount(components.ravel(), minlength=count + 1)[1:] * np.prod(spacing)
    if count <= np.iinfo(np.uint16).max:
        components = components.astype(np.uint16)
    return components.reshape(original_shape), volumes


def boundary_points(components, spacing):
    components, spacing = physical_grid(components, spacing)
    points = []
    for i, bbox in enumerate(ndi.find_objects(components), start=1):
        local = components[bbox] == i
        boundary = local & ~ndi.binary_erosion(local)
        offset = np.asarray([s.start for s in bbox])
        points.append((np.argwhere(boundary) + offset) * spacing)
    return points


def distance_tree(points):
    """Exact Prim MST, O(k) edge storage; box bounds avoid most pair queries.

    Boundary points retain all possible closest points. Build trees lazily and query
    the larger component's tree with the smaller point set. Prim and edge ties use
    deterministic component identifiers; no dense k-by-k distance matrix is needed.
    """
    n = len(points)
    if n < 2:
        return np.empty((0, 3), dtype=float)
    low = np.stack([p.min(0) for p in points])
    high = np.stack([p.max(0) for p in points])
    trees = {}
    selected = np.zeros(n, dtype=bool)
    best = np.full(n, np.inf)
    source = np.full(n, -1, dtype=int)
    current = max(range(n), key=lambda i: (len(points[i]), -i))
    edges = []
    for step in range(n):
        selected[current] = True
        if step:
            edges.append((min(current, source[current]) + 1, max(current, source[current]) + 1, best[current]))
        candidates = np.flatnonzero(~selected)
        lower = np.linalg.norm(np.maximum(0, np.maximum(low[current] - high[candidates],
                                                       low[candidates] - high[current])), axis=1)
        for j in candidates[lower <= best[candidates]]:
            large, small = (current, j) if len(points[current]) >= len(points[j]) else (j, current)
            if large not in trees:
                trees[large] = cKDTree(points[large])
            distance = float(trees[large].query(points[small])[0].min())
            pair = (min(current, j), max(current, j))
            old_pair = (min(source[j], j), max(source[j], j))
            if distance < best[j] or (distance == best[j] and pair < old_pair):
                best[j], source[j] = distance, current
        if len(candidates):
            current = min(candidates, key=lambda j: (best[j], min(source[j], j), max(source[j], j)))
    return np.asarray(sorted(edges, key=lambda e: (e[2], e[0], e[1])), dtype=float)


def component_measurements(mask, spacing, connectivity=26, with_tree=True):
    components, volumes = component_volumes(mask, spacing, connectivity)
    edges = distance_tree(boundary_points(components, spacing)) if with_tree else np.empty((0, 3))
    return components, volumes, edges


def group_membership(volumes, edges, distance):
    parent = np.arange(len(volumes) + 1)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if distance is not None:
        for a, b, gap in edges:
            if gap <= distance:
                a, b = find(int(a)), find(int(b))
                parent[max(a, b)] = min(a, b)
    roots = np.asarray([find(i) for i in range(1, len(volumes) + 1)], dtype=int)
    _, inverse = np.unique(roots, return_inverse=True)
    mapping = np.concatenate(([0], inverse + 1))
    sums = np.bincount(inverse, weights=volumes) if len(volumes) else np.empty(0)
    return mapping, sums


def grouped_components(mask, spacing, connectivity, distance, cache=None):
    compute = lambda: component_volumes(mask, spacing, connectivity)
    components, volumes = (compute() if cache is None else
                           cache.value('components', mask, spacing, connectivity, compute))
    if distance is None:
        compute_groups = lambda: group_membership(volumes, [], None)
    else:
        compute_points = lambda: tuple(boundary_points(components, spacing))
        points = (compute_points() if cache is None else
                  cache.value('boundary_points', mask, spacing, connectivity, compute_points))
        compute_groups = lambda: proximity_membership(volumes, points, distance)
    mapping, sums = (compute_groups() if cache is None else
                     cache.value('membership', mask, spacing, (connectivity, distance), compute_groups))
    return mapping.astype(components.dtype, copy=False)[components], sums


def proximity_membership(volumes, points, distance):
    """Exact threshold-graph groups without constructing a complete prediction MST.

    Bounding-sphere buckets produce a conservative spatial candidate set. Boundary KD queries
    confirm edges within the threshold, and union/find discards edges already joining one group.
    Radius buckets prevent one large object from making all tiny-object queries quadratic.
    """
    count = len(points)
    if count < 2:
        return group_membership(volumes, [], None)
    low = np.stack([p.min(0) for p in points])
    high = np.stack([p.max(0) for p in points])
    centers = (low + high) / 2
    radii = np.linalg.norm(high - low, axis=1) / 2
    # Zero-radius singletons get their own bucket, including when spacing is submillimeter.
    bucket_levels = np.full(count, -10000, dtype=int)
    positive = radii > 0
    bucket_levels[positive] = np.floor(np.log2(radii[positive])).astype(int)
    buckets = []
    for level in np.unique(bucket_levels):
        ids = np.flatnonzero(bucket_levels == level)
        buckets.append((ids, cKDTree(centers[ids]), float(radii[ids].max())))
    parent, trees, edges = np.arange(count), {}, []

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    bound = float(distance) + 8 * np.finfo(float).eps * max(1., abs(distance))
    for i in sorted(range(count), key=lambda j: (-len(points[j]), j)):
        candidates = []
        for ids, tree, maximum in buckets:
            radius = distance + radii[i] + maximum
            radius += 8 * np.finfo(float).eps * max(1., abs(radius))
            candidates.extend(ids[tree.query_ball_point(centers[i], radius)].tolist())
        for j in sorted(candidates):
            a, b = root(i), root(j)
            if a == b:
                continue
            lower = np.linalg.norm(np.maximum(0, np.maximum(low[i] - high[j], low[j] - high[i])))
            if lower > bound:
                continue
            large, small = (i, j) if len(points[i]) >= len(points[j]) else (j, i)
            if large not in trees:
                trees[large] = cKDTree(points[large])
            gap = float(trees[large].query(points[small], distance_upper_bound=bound)[0].min())
            if gap <= distance:
                parent[max(a, b)] = min(a, b)
                edges.append((i + 1, j + 1, gap))
                if len(edges) == count - 1:
                    return group_membership(volumes, edges, distance)
    return group_membership(volumes, edges, distance)
