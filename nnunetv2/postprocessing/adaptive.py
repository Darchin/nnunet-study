"""Training-fitted morphology and component priors. All geometry is on the original grid.

The search is bounded coordinate search on in-sample training predictions, not a
global optimizer. Validation annotations are never inputs to the fitter.
"""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from numbers import Integral
from pathlib import Path
import warnings
import weakref

import numpy as np
from scipy import ndimage as ndi

from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.postprocessing.components import (
    physical_grid, effective_connectivity, component_volumes, component_measurements,
    distance_tree, group_membership, grouped_components,
)
from nnunetv2.postprocessing.configuration import (
    PERCENTILE_CATALOGUE, QUANTILE_CONVENTIONS, percentile_table,
)

VERSION = 2
EPS = 1e-8
OPERATIONS = ("grouping_distance", "closing_radius", "hole_volume", "min_volume", "max_count")


def label_spec(dataset_json, label_manager=None):
    lm = label_manager or LabelManager(dataset_json['labels'], dataset_json.get('regions_class_order'))
    regions = lm.foreground_regions if lm.has_regions else lm.foreground_labels
    sets = [sorted(set([int(r)] if np.isscalar(r) else map(int, r))) for r in regions]
    return {"labels": dataset_json['labels'], "regions_class_order": dataset_json.get('regions_class_order'),
            "regions": sets, "output_labels": [int(v) for v in
                                                (lm.regions_class_order if lm.has_regions else lm.foreground_labels)],
            "has_regions": lm.has_regions}


def validate_spec(spec):
    sets = list(map(set, spec['regions']))
    if not sets:
        raise ValueError('Adaptive post-processing requires at least one foreground label/region.')
    for i, a in enumerate(sets):
        if not a or 0 in a:
            raise ValueError('Adaptive post-processing requires nonempty foreground regions excluding background.')
        for j, b in enumerate(sets[:i]):
            if a == b or (a & b and not (a < b or b < a)):
                raise ValueError('Adaptive post-processing supports only distinct nested or disjoint regions; '
                                 f'regions {j} and {i} cross or are duplicates. Revise dataset label definitions.')
        # The exported scalar label must express this region's membership in every other region.
        expected = {j for j, b in enumerate(sets) if a <= b}
        actual = {j for j, b in enumerate(sets) if spec['output_labels'][i] in b}
        if expected != actual:
            raise ValueError('regions_class_order cannot encode this nested-region hierarchy. '
                             'Give each region an output label belonging to it and all ancestors only.')
    if spec['has_regions'] and any(a < b and i < j for i, a in enumerate(sets) for j, b in enumerate(sets)):
        raise ValueError('Adaptive post-processing requires regions_class_order in parent-before-child dataset order.')


def relations(spec):
    sets = list(map(set, spec['regions']))
    parents = [[j for j, b in enumerate(sets) if a < b] for a in sets]
    disjoint = [[j for j, b in enumerate(sets) if not a & b] for a in sets]
    order = sorted(range(len(sets)), key=lambda i: (-len(sets[i]), i))
    return parents, disjoint, order


def masks_from_segmentation(segmentation, spec):
    return np.stack([np.isin(segmentation, r) for r in spec['regions']])


def masks_from_probabilities(probabilities, spec):
    if spec['has_regions']:
        return np.asarray(probabilities) > .5
    return masks_from_segmentation(np.asarray(probabilities).argmax(0), spec)


def masks_to_segmentation(masks, spec):
    result = np.zeros(masks[0].shape, dtype=np.uint16)
    for mask, output_label in zip(masks, spec['output_labels']):
        result[mask] = output_label
    return result


def cavity_components(mask, spacing):
    original_shape = mask.shape
    mask, spacing = physical_grid(mask, spacing)
    # Background connectivity stays full independently of foreground connectivity.
    background, count = ndi.label(~mask, ndi.generate_binary_structure(mask.ndim, mask.ndim))
    exterior = set()
    for axis in range(mask.ndim):
        exterior.update(np.unique(np.take(background, 0, axis)).tolist())
        exterior.update(np.unique(np.take(background, -1, axis)).tolist())
    ids = np.asarray([i for i in range(1, count + 1) if i not in exterior], dtype=int)
    sizes = np.bincount(background.ravel(), minlength=count + 1) * np.prod(spacing)
    return background.reshape(original_shape), ids, sizes[ids]


def close_mask(mask, spacing, radius):
    original_shape = mask.shape
    mask, spacing = physical_grid(mask, spacing)
    if radius is None or radius <= 0 or not mask.any():
        return mask.copy().reshape(original_shape)
    spacing = np.asarray(spacing, dtype=float)
    if radius < spacing.min():
        return mask.copy().reshape(original_shape)
    # Closing is local to interacting dilations. Expanded component boxes conservatively include
    # touching discrete footprints; separate groups can be closed independently with identical results.
    components, volumes = component_volumes(mask, spacing)
    boxes = ndi.find_objects(components)
    extent = np.ceil(radius / spacing).astype(int) + 1
    low = np.asarray([[s.start for s in box] for box in boxes])
    high = np.asarray([[s.stop for s in box] for box in boxes])
    parent = np.arange(len(boxes))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(boxes)):
        matches = np.flatnonzero(np.all((low[i] - extent <= high[i + 1:] + extent) &
                                        (low[i + 1:] - extent <= high[i] + extent), axis=1)) + i + 1
        for j in matches:
            a, b = root(i), root(j)
            parent[max(a, b)] = min(a, b)
    groups = {}
    for i in range(len(boxes)):
        groups.setdefault(root(i), []).append(i)
    result = mask.copy()
    for members in groups.values():
        if len(members) == 1 and volumes[members[0]] == np.prod(spacing):
            continue  # A singleton is unchanged by closing with any finite spherical footprint.
        start = np.maximum(0, low[members].min(0) - extent)
        stop = np.minimum(mask.shape, high[members].max(0) + extent)
        crop = tuple(slice(int(a), int(b)) for a, b in zip(start, stop))
        membership = np.zeros(len(boxes) + 1, dtype=bool)
        membership[np.asarray(members) + 1] = True
        local = membership[components[crop]]
        result[crop] |= _close_dense(local, spacing, radius)
    return result.reshape(original_shape)


def _close_dense(mask, spacing, radius):
    extent = np.ceil(radius / spacing).astype(int)
    padding = [(int(e) + 1, int(e) + 1) for e in extent]
    padded = np.pad(mask, padding)
    # These are dilation/erosion by the discrete spacing-aware spherical footprint.
    # Distance transforms avoid the radius-dependent cost of a huge explicit kernel.
    dilated = ndi.distance_transform_edt(~padded, sampling=spacing) <= radius
    closed = ndi.distance_transform_edt(dilated, sampling=spacing) > radius
    crop = tuple(slice(p, -p) for p, _ in padding)
    return mask | closed[crop]


class MorphologyCache:
    """Content-addressed, bounded cache shared across repeated coordinate trials."""
    def __init__(self, max_bytes=128 * 1024 ** 2):
        self.max_bytes, self.nbytes = max_bytes, 0
        self.entries = OrderedDict()
        self.hashes = {}

    def get(self, operation, mask, spacing, parameter, compute):
        saved = self.hashes.get(id(mask)) if not mask.flags.writeable else None
        if saved is not None and saved[0]() is mask:
            digest = saved[1]
        else:
            digest = hashlib.blake2b(np.ascontiguousarray(mask).view(np.uint8), digest_size=16).digest()
            if not mask.flags.writeable:
                identifier = id(mask)
                owner = weakref.ref(self)

                def discard(_, key=identifier, owner=owner):
                    cache = owner()
                    if cache is not None:
                        cache.hashes.pop(key, None)

                self.hashes[identifier] = (weakref.ref(mask, discard), digest)
        key = (operation, mask.shape, tuple(spacing), parameter, digest)
        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key]
        result = compute()
        arrays = result if isinstance(result, tuple) else (result,)
        size = sum(a.nbytes for a in arrays if isinstance(a, np.ndarray))
        if size <= self.max_bytes:
            while self.entries and self.nbytes + size > self.max_bytes:
                _, (_, old_size) = self.entries.popitem(last=False)
                self.nbytes -= old_size
            self.entries[key] = (result, size)
            for array in arrays:
                if isinstance(array, np.ndarray):
                    array.flags.writeable = False
            self.nbytes += size
        return result, size

    def value(self, operation, mask, spacing, parameter, compute):
        return self.get(operation, mask, spacing, parameter, compute)[0]


def repair_hierarchy(masks, spec, direction):
    parents, _, order = relations(spec)
    if direction == 'expand':
        for i in reversed(order):
            for parent in parents[i]:
                masks[parent] |= masks[i]
    elif direction == 'restrict':
        for i in order:
            for parent in parents[i]:
                masks[i] &= masks[parent]
    else:
        raise ValueError(f'Unknown hierarchy direction: {direction}')
    return masks


def repair_masks(masks, spacing, spec, settings, direction, cache=None, stop_before_fill=None):
    masks = repair_hierarchy(np.array(masks, dtype=bool, copy=True), spec, direction)
    parents, disjoint, _ = relations(spec)
    cache = cache or MorphologyCache()
    for i in range(len(masks)):  # dataset order breaks ties between competing additions
        old = masks[i]
        radius = settings[i]['closing_radius']
        if radius is None and (settings[i]['hole_volume'] is None or stop_before_fill == i):
            if stop_before_fill == i:
                return masks
            continue
        closed = cache.value('close', old, spacing, radius, lambda: close_mask(old, spacing, radius))
        allowed = np.ones(old.shape, dtype=bool)
        for j in disjoint[i]:
            allowed &= ~masks[j]
        if direction == 'restrict':
            for j in parents[i]:
                allowed &= masks[j]
        masks[i] = old | (closed & allowed)
        repair_hierarchy(masks, spec, direction)
        if stop_before_fill == i:
            return masks
        limit = settings[i]['hole_volume']
        if limit is not None:
            mask = masks[i]
            cavities, ids, volumes = cache.value('cavities', mask, spacing, None,
                                                lambda: cavity_components(mask, spacing))
            masks[i] |= np.isin(cavities, ids[volumes <= limit]) & allowed
        repair_hierarchy(masks, spec, direction)
    return masks


def apply_policy(masks, spacing, spec, policy, cache=None):
    if policy['direction'] == 'identity':
        return masks_to_segmentation(masks, spec)
    cache = cache or MorphologyCache()
    settings = policy['settings']
    morphology = (policy['direction'], tuple(map(tuple, spec['regions'])),
                  tuple((s['closing_radius'], s['hole_volume']) for s in settings))
    repaired = cache.value('repair_pipeline', masks, spacing, morphology,
                           lambda: repair_masks(masks, spacing, spec, settings, policy['direction'], cache))
    # Filtering replaces arrays; cached morphology remains immutable and shared across threshold trials.
    masks = list(repaired)
    parents, _, order = relations(spec)
    for i in order:
        mask = masks[i]
        if settings[i]['min_volume'] is None and settings[i]['max_count'] is None:
            continue
        if (settings[i]['max_count'] is None and settings[i]['min_volume'] is not None
            and settings[i]['min_volume'] <= np.prod(physical_grid(mask, spacing)[1])):
            continue  # No component or group can be smaller than one physical foreground voxel.
        connectivity = policy['connectivity']['foreground']
        components, volumes = grouped_components(mask, spacing, connectivity,
                                                 settings[i]['grouping_distance'], cache)
        ids = np.arange(1, len(volumes) + 1)
        minimum = settings[i]['min_volume']
        if minimum is not None:
            ids = ids[volumes >= minimum]
        maximum = settings[i]['max_count']
        if maximum is not None:
            ids = ids[np.lexsort((ids, -volumes[ids - 1]))[:maximum]]
        keep = np.zeros(len(volumes) + 1, dtype=bool)
        keep[ids] = True
        masks[i] = keep[components]
        for child in order:
            if i in parents[child]:
                masks[child] = masks[child] & masks[i]
    return masks_to_segmentation(masks, spec)


def fragmentation_gaps(predicted, reference, spacing, connectivity=26, grouping_distance=None, cache=None):
    predicted_ids, volumes = component_volumes(predicted, spacing, connectivity)
    reference_ids, _ = grouped_components(reference, spacing, connectivity, grouping_distance, cache)
    predicted_ids, grid_spacing = physical_grid(predicted_ids, spacing)
    reference_ids, _ = physical_grid(reference_ids, spacing)
    objects = ndi.find_objects(predicted_ids)
    groups = {}
    for i in range(1, len(volumes) + 1):
        bbox = objects[i - 1]
        local = predicted_ids[bbox] == i
        matches = np.unique(reference_ids[bbox][local])
        matches = matches[matches > 0]
        if len(matches) == 1:
            boundary = local & ~ndi.binary_erosion(local)
            offset = np.asarray([s.start for s in bbox])
            points = (np.argwhere(boundary) + offset) * grid_spacing
            groups.setdefault(int(matches[0]), []).append(points)
    gaps = []
    for fragments in groups.values():
        if len(fragments) < 2:
            continue
        gaps.extend(distance_tree(fragments)[:, 2].tolist())
    return gaps


def candidates(values, percentiles, discrete=False, scale=1.):
    if not len(values):
        return [None]
    method = 'inverted_cdf' if discrete else 'linear'
    values = np.percentile(values, percentiles, method=method) * scale
    return [None] + sorted(set(int(v) if discrete else float(v) for v in values))


def component_metadata(spec):
    return {**spec, "foreground_connectivities": [26],
            "2d_connectivities": [8], "background_connectivity": "full", "grid": "original",
            "units": "mm^ndim", "distance_units": "mm", "distance": "shortest boundary voxel centers",
            "percentile_catalogue": PERCENTILE_CATALOGUE, "quantiles": QUANTILE_CONVENTIONS,
            "annotation_digest": "sha256 of C-order little-endian int32 labels"}


def extract_case_components(segmentation, spacing, spec):
    masks = masks_from_segmentation(segmentation, spec)
    grid, _ = physical_grid(segmentation, spacing)
    variants = {}
    for connectivity in ([8] if grid.ndim == 2 else [26]):
        regions = []
        for mask in masks:
            _, volumes, edges = component_measurements(mask, spacing, connectivity)
            edges_list = [[int(a), int(b), float(d)] for a, b, d in edges]
            regions.append({'component_ids': list(range(1, len(volumes) + 1)), 'volumes': volumes.tolist(),
                            'edges': edges_list, 'volume_percentiles': percentile_table([volumes]),
                            'distance_percentiles': percentile_table([edges[:, 2]])})
        variants[str(connectivity)] = regions
    from nnunetv2.postprocessing.runtime import annotation_digest
    return {'spacing': list(map(float, spacing)), 'shape': list(segmentation.shape),
            'connectivities': variants, 'annotation_digest': annotation_digest(segmentation)}


def case_regions(case, connectivity):
    grid_ndim = len(case['shape'])
    if grid_ndim == 3 and case['shape'][0] == 1 and case['spacing'][0] == 999:
        grid_ndim = 2
    return case['connectivities'][str(effective_connectivity(grid_ndim, connectivity))]


def fingerprint_summaries(cases, region_count):
    summaries = {}
    for connectivity in sorted({v for case in cases.values() for v in case['connectivities']}):
        eligible = [case['connectivities'][connectivity] for _, case in sorted(cases.items())
                    if connectivity in case['connectivities']]
        summaries[connectivity] = {}
        for aggregation in ('instance', 'case'):
            summaries[connectivity][aggregation] = [
                {'volume_percentiles': percentile_table([c[i]['volumes'] for c in eligible], aggregation),
                 'distance_percentiles': percentile_table([[e[2] for e in c[i]['edges']] for c in eligible], aggregation),
                 'count_percentiles': percentile_table([[len(c[i]['volumes'])] for c in eligible
                                                        if c[i]['volumes']], aggregation, True)}
                for i in range(region_count)]
    return summaries


def component_case_is_valid(case, region_count, connectivity=26, require_digest=True):
    try:
        regions = case_regions(case, connectivity)
        physical_grid(np.empty((1,) * len(case['shape']), dtype=bool), case['spacing'])
        valid = (len(regions) == region_count and len(case['shape']) == len(case['spacing'])
                 and all(isinstance(s, int) and s > 0 for s in case['shape']))
        if require_digest:
            digest = case.get('annotation_digest')
            valid &= isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest)
        for region in regions:
            volumes, edges = region['volumes'], region['edges']
            valid &= region['component_ids'] == list(range(1, len(volumes) + 1))
            valid &= all(np.isfinite(v) and v > 0 for v in volumes)
            valid &= len(edges) == max(0, len(volumes) - 1)
            valid &= all(len(e) == 3 and 1 <= e[0] < e[1] <= len(volumes)
                         and int(e[0]) == e[0] and int(e[1]) == e[1]
                         and np.isfinite(e[2]) and e[2] > 0 for e in edges)
            if valid and volumes:
                valid &= len(group_membership(np.asarray(volumes), edges, np.inf)[1]) == 1
        return bool(valid)
    except (KeyError, TypeError, ValueError, IndexError):
        return False


def validate_fingerprint(fingerprint, spec, training_ids=None, connectivity=26):
    validate_spec(spec)
    if 'ignore' in spec['labels']:
        raise ValueError('Adaptive fitting does not support ignore regions or partially annotated targets.')
    statistics = fingerprint.get('component_statistics', {})
    if statistics.get('metadata') != component_metadata(spec) or not statistics.get('cases'):
        raise ValueError('Missing or incompatible component statistics. Refresh the fingerprint with '
                         'PercentileFingerprintExtractor (preprocessing need not be repeated).')
    missing = set(training_ids or ()) - statistics['cases'].keys()
    if missing:
        raise ValueError(f'Component fingerprint lacks training cases {sorted(missing)}; refresh the fingerprint.')
    for identifier in training_ids or ():
        case = statistics['cases'][identifier]
        if not component_case_is_valid(case, len(spec['regions']), connectivity):
            raise ValueError(f'Invalid component statistics for {identifier}; refresh the fingerprint.')
    return statistics


def policy_rank(policy):
    settings = policy['settings']
    return (sum(v is not None for s in settings for v in s.values()),
            tuple(s['grouping_distance'] or 0 for s in settings),
            tuple(s['closing_radius'] or 0 for s in settings),
            tuple(s['hole_volume'] or 0 for s in settings),
            tuple(s['min_volume'] or 0 for s in settings),
            tuple(-s['max_count'] if s['max_count'] is not None else -float('inf') for s in settings))


def mean_dice(segmentation, reference_masks, spec):
    predicted = masks_from_segmentation(segmentation, spec)
    intersection = (predicted & reference_masks).sum(axis=tuple(range(1, predicted.ndim)))
    denominator = predicted.sum(axis=tuple(range(1, predicted.ndim))) + reference_masks.sum(
        axis=tuple(range(1, predicted.ndim)))
    return float(np.divide(2 * intersection, denominator, out=np.ones(len(predicted)), where=denominator > 0).mean())


def fit_policy(case_ids, load_case, fingerprint, spec, metadata=None, progress=None, configuration=None, **kwargs):
    from nnunetv2.postprocessing.fitting import fit_policy as fit
    return fit(case_ids, load_case, fingerprint, spec, metadata, progress, configuration, **kwargs)


def load_policy(path_or_policy, spec, enable_tta=None):
    if isinstance(path_or_policy, (str, Path)):
        with open(path_or_policy, encoding='utf-8') as f:
            policy = json.load(f)
    else:
        policy = deepcopy(path_or_policy)
    validate_spec(spec)
    if policy.get('version') != VERSION or policy.get('spec') != spec:
        raise ValueError('Saved adaptive policy version or label definitions do not match this model. '
                         'Refit post-processing with the current fingerprint and algorithm.')
    connectivity = policy.get('connectivity', {}).get('foreground')
    if isinstance(connectivity, bool) or not isinstance(connectivity, Integral) or connectivity not in (8, 26):
        raise ValueError('Saved policy must use full foreground connectivity (8/26). Refit post-processing.')
    if policy.get('connectivity', {}).get('background') != 'full':
        raise ValueError('Saved policy must use full cavity background connectivity.')
    if policy.get('direction') not in ('identity', 'expand', 'restrict'):
        raise ValueError('Invalid adaptive hierarchy policy.')
    if len(policy.get('settings', [])) != len(spec['regions']):
        raise ValueError('Saved policy must contain settings for every region.')
    for settings in policy['settings']:
        if set(settings) != set(OPERATIONS):
            raise ValueError('Saved policy has missing or unknown operation settings.')
        for name, value in settings.items():
            if value is not None:
                if (not isinstance(value, (int, float)) or isinstance(value, bool) or
                    not np.isfinite(value) or value < 0 or (name == 'max_count' and int(value) != value)):
                    raise ValueError(f'Invalid saved {name}: {value}')
                if name == 'max_count':
                    settings[name] = int(value)
    fitted_tta = policy.get('metadata', {}).get('enable_tta')
    if enable_tta is not None and fitted_tta is not None and enable_tta != fitted_tta:
        warnings.warn('Inference TTA differs from adaptive post-processing fitting TTA.', stacklevel=2)
    return policy
