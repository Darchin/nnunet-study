"""Picklable case tasks. Only small measurements and metrics cross process boundaries."""
import numpy as np

from nnunetv2.postprocessing.adaptive import (
    MorphologyCache, apply_policy, repair_masks, cavity_components, case_regions,
)
from nnunetv2.postprocessing.components import (
    component_volumes, physical_grid, group_membership, distance_tree,
)


def prepare_gaps(identifier, load_case, spec, records, grouping_grids):
    masks, reference, spacing = load_case(identifier)
    result = []
    for region, grouping_grid in enumerate(grouping_grids):
        predicted, volumes = component_volumes(masks[region], spacing)
        target, _ = component_volumes(np.isin(reference, spec['regions'][region]), spacing)
        predicted, grid_spacing = physical_grid(predicted, spacing)
        target, _ = physical_grid(target, spacing)
        from scipy import ndimage as ndi
        overlaps, points = {}, {}
        for component, bbox in enumerate(ndi.find_objects(predicted), 1):
            local = predicted[bbox] == component
            matched = np.unique(target[bbox][local])
            matched = matched[matched > 0]
            if len(matched):
                overlaps[component] = matched
                boundary = local & ~ndi.binary_erosion(local)
                points[component] = (np.argwhere(boundary) + [s.start for s in bbox]) * grid_spacing
        record, contexts, tree_cache = records[region], [], {}
        for grouping in grouping_grid:
            mapping, _ = group_membership(np.asarray(record['volumes']), record['edges'], grouping)
            fragments = {}
            for component, matched in overlaps.items():
                groups = np.unique(mapping[matched])
                if len(groups) == 1:
                    fragments.setdefault(int(groups[0]), []).append(component)
            gaps = []
            for members in fragments.values():
                if len(members) > 1:
                    key = tuple(members)
                    if key not in tree_cache:
                        tree_cache[key] = distance_tree([points[i] for i in members])[:, 2].tolist()
                    gaps.extend(tree_cache[key])
            contexts.append(gaps)
        result.append(contexts)
    return result


def measure_holes(identifier, load_case, spec, policies, region):
    masks, _, spacing = load_case(identifier)
    cache, results = MorphologyCache(), []
    for policy in policies:
        repaired = repair_masks(masks, spacing, spec, policy['settings'], policy['direction'], cache,
                                stop_before_fill=region)
        volumes = cache.value('cavities', repaired[region], spacing, None,
                             lambda: cavity_components(repaired[region], spacing))[2]
        results.append(volumes.tolist())
    return results


METRICS = ('Dice', 'IoU', 'FP', 'TP', 'FN', 'TN', 'n_pred', 'n_ref')


def metric_arrays(segmentation, reference_indices, labels, membership):
    indices = np.searchsorted(labels, segmentation).astype(np.int32, copy=False)
    encoded = reference_indices.reshape(-1) * len(labels) + indices.reshape(-1)
    confusion = np.bincount(encoded, minlength=len(labels) ** 2).reshape(len(labels), len(labels))
    counts = []
    for included in membership:
        tp = int(confusion[np.ix_(included, included)].sum())
        n_ref = int(confusion[included].sum())
        n_pred = int(confusion[:, included].sum())
        fp, fn = n_pred - tp, n_ref - tp
        tn = int(confusion.sum()) - tp - fp - fn
        denominator = n_ref + n_pred
        counts.append([2 * tp / denominator if denominator else 1.,
                       tp / (tp + fp + fn) if tp + fp + fn else 1., fp, tp, fn, tn, n_pred, n_ref])
    return counts


def evaluate_case(identifier, load_case, spec, policies):
    masks, reference, spacing = load_case(identifier)
    masks.flags.writeable = False
    labels = np.asarray(sorted({0, *[v for region in spec['regions'] for v in region]}))
    indices = np.searchsorted(labels, reference).astype(np.int32)
    membership = [np.isin(labels, region) for region in spec['regions']]
    cache = MorphologyCache()
    results = []
    for policy in policies:
        segmentation = apply_policy(masks, spacing, spec, policy, cache)
        results.append(metric_arrays(segmentation, indices, labels, membership))
    return results


def summarize_case_metrics(case_ids, per_case, spec, prediction_files=None, reference_files=None):
    from nnunetv2.utilities.label_handling.label_handling import LabelManager
    from nnunetv2.evaluation.evaluate_predictions import label_or_region_to_key
    values = np.asarray(per_case, dtype=float)
    means = values.mean(axis=0)
    labels = LabelManager(spec['labels'], spec.get('regions_class_order'))
    keys = [label_or_region_to_key(r) for r in
            (labels.foreground_regions if spec['has_regions'] else labels.foreground_labels)]
    result = {'mean': {key: dict(zip(METRICS, means[i].tolist())) for i, key in enumerate(keys)},
              'foreground_mean': dict(zip(METRICS, means.mean(axis=0).tolist())), 'metric_per_case': []}
    for index, identifier in enumerate(case_ids):
        metrics = {key: {name: float(value) if name in ('Dice', 'IoU') else int(value)
                         for name, value in zip(METRICS, values[index, i])} for i, key in enumerate(keys)}
        result['metric_per_case'].append({'case': identifier, 'metrics': metrics,
            'prediction_file': prediction_files[index] if prediction_files else identifier,
            'reference_file': reference_files[index] if reference_files else identifier})
    return result
