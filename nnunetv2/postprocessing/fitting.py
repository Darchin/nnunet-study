"""Fold-isolated candidate generation and bounded region coordinate searches."""
from copy import deepcopy
from itertools import product

import numpy as np

from nnunetv2.postprocessing.adaptive import (
    VERSION, EPS, OPERATIONS, MorphologyCache, validate_fingerprint, case_regions, relations,
    masks_from_segmentation, mean_dice, apply_policy, repair_masks, cavity_components,
    fragmentation_gaps, group_membership, policy_rank,
)
from nnunetv2.postprocessing.configuration import (
    OPERATION_SELECTORS, QUANTILE_CONVENTIONS, resolve_configuration, resolve_percentiles,
    distribution_candidates, percentile_table,
)


def fit_policy(case_ids, load_case, fingerprint, spec, metadata=None, progress=None, configuration=None):
    """Stream only explicitly supplied training cases; global summaries are never read."""
    case_ids = sorted(case_ids)
    if not case_ids:
        raise ValueError('Adaptive fitting requires at least one training case.')
    configuration = resolve_configuration(configuration)
    connectivity, aggregation = configuration['connectivity'], configuration['aggregation']
    selectors = {op: resolve_percentiles(configuration[key]) for op, key in OPERATION_SELECTORS.items()}
    statistics = validate_fingerprint(fingerprint, spec, case_ids, connectivity)
    count, cache = len(spec['regions']), MorphologyCache()
    report = {'configuration': configuration, 'quantile_conventions': QUANTILE_CONVENTIONS,
              'fold_distributions': [], 'fixed_candidate_grids': [], 'trials': [], 'runs': [],
              'warnings': []}
    contexts, grouping_grids = {}, []

    def grid(values, operation, discrete=False, scale=1.):
        return distribution_candidates(values, selectors[operation], aggregation, discrete, scale)

    def describe(values, discrete=False):
        return {'per_case': {k: list(map(int if discrete else float, v)) for k, v in zip(case_ids, values)},
                'percentiles': percentile_table(values, aggregation, discrete)}

    for region in range(count):
        records = [case_regions(statistics['cases'][k], connectivity)[region] for k in case_ids]
        distances = [[e[2] for e in r['edges']] for r in records]
        grouping_grid = grid(distances, 'grouping_distance')
        grouping_grids.append(grouping_grid)
        description = {'grouping_distances': describe(distances), 'grouping_grid': grouping_grid, 'contexts': []}
        for grouping in grouping_grid:
            sizes = [group_membership(np.asarray(r['volumes']), r['edges'], grouping)[1].tolist() for r in records]
            counts = [[len(v)] if v else [] for v in sizes]
            gaps = []
            for k in case_ids:
                if selectors['closing_radius']:
                    masks, reference, spacing = load_case(k)
                    gaps.append(fragmentation_gaps(masks[region], np.isin(reference, spec['regions'][region]),
                                                   spacing, connectivity, grouping, cache))
                else:
                    gaps.append([])
            grids = {'grouping_distance': grouping_grid, 'min_volume': grid(sizes, 'min_volume'),
                     'max_count': grid(counts, 'max_count', True),
                     'closing_radius': grid(gaps, 'closing_radius', scale=.5)}
            contexts[region, grouping] = grids
            no_effect = [v for v in grids['min_volume'][1:] if all(v <= s for vs in sizes for s in vs)]
            duplicates = {op: len(selectors[op]) - (len(grids[op]) - 1)
                          for op in ('min_volume', 'max_count', 'closing_radius')
                          if selectors[op] and any({'min_volume': sizes, 'max_count': counts,
                                                   'closing_radius': gaps}[op])}
            context = {'grouping_distance': grouping, 'group_volumes': describe(sizes),
                       'positive_case_counts': describe(counts, True), 'fragmentation_gaps': describe(gaps),
                       'grids': grids, 'duplicate_candidate_counts': duplicates,
                       'size_candidates_without_reference_removals': no_effect}
            description['contexts'].append(context)
            if no_effect:
                report['warnings'].append(f'Region {region}, grouping {grouping}: size candidates {no_effect} '
                                          'cannot remove any observed annotation group.')
        report['fold_distributions'].append(description)
        report['fixed_candidate_grids'].append(deepcopy(contexts[region, None]))

    base = {'version': VERSION, 'spec': spec, 'direction': 'identity',
            'settings': [{op: None for op in OPERATIONS} for _ in range(count)],
            'training_identifiers': case_ids, 'metadata': metadata or {}, 'configuration': configuration,
            'percentile_grids': selectors, 'quantile_conventions': QUANTILE_CONVENTIONS,
            'search': f"bounded region coordinate search ({configuration['hyperparameter_search']})",
            'connectivity': {'foreground': connectivity, 'background': 'full'}}
    score_cache, hole_cache = {}, {}

    def policy_key(policy):
        return policy['direction'], tuple(tuple(s[op] for op in OPERATIONS) for s in policy['settings'])

    def score(policy):
        key = policy_key(policy)
        if key not in score_cache:
            scores = []
            for k in case_ids:
                masks, reference, spacing = load_case(k)
                segmentation = apply_policy(masks, spacing, spec, policy, cache)
                scores.append(mean_dice(segmentation, masks_from_segmentation(reference, spec), spec))
            score_cache[key] = float(np.mean(scores))
        return score_cache[key]

    def hole_candidates(policy, region):
        if not selectors['hole_volume']:
            return [None]
        key = (region, policy_key(policy))
        if key not in hole_cache:
            holes = []
            for k in case_ids:
                masks, _, spacing = load_case(k)
                repaired = repair_masks(masks, spacing, spec, policy['settings'], policy['direction'], cache,
                                        stop_before_fill=region)
                volumes = cache.value('cavities', repaired[region], spacing, None,
                                     lambda: cavity_components(repaired[region], spacing))[2]
                holes.append(volumes.tolist())
            hole_cache[key] = grid(holes, 'hole_volume')
            report.setdefault('filling_contexts', []).append({'region': region, 'direction': policy['direction'],
                'settings': deepcopy(policy['settings']), 'cavities': describe(holes), 'grid': hole_cache[key],
                'duplicate_candidate_count': len(selectors['hole_volume']) - len(hole_cache[key]) + 1
                                            if any(holes) else 0})
        return hole_cache[key]

    def better(trial, trial_score, incumbent, incumbent_score):
        return (trial_score > incumbent_score + EPS or
                (abs(trial_score - incumbent_score) <= EPS and policy_rank(trial) < policy_rank(incumbent)))

    def evaluate(trial, region, sweep, operation, grid_values):
        value = score(trial)
        report['trials'].append({'direction': trial['direction'], 'sweep': sweep + 1, 'region': region,
                                'operation': operation, 'settings': deepcopy(trial['settings'][region]),
                                'grid': grid_values, 'score': value})
        return value

    baseline_score = score(base)
    report['baseline_score'] = baseline_score
    winner, winner_score = deepcopy(base), baseline_score
    directions = {'auto': ('expand', 'restrict'), 'child': ('expand',), 'parent': ('restrict',)}[
        configuration['hierarchy_repair']]
    _, _, order = relations(spec)
    for direction in directions:
        current = deepcopy(base)
        current['direction'] = direction
        current_score = score(current)
        limit_reached = False
        for sweep in range(5):
            improved = False
            for region in order:
                if progress:
                    progress(f'Fitting post-processing: {direction}, sweep {sweep + 1}, region {region + 1}/{count}')
                local_best, local_score = deepcopy(current), current_score
                for grouping in grouping_grids[region]:
                    grids = contexts[region, grouping]
                    branch = deepcopy(current)
                    branch['settings'][region]['grouping_distance'] = grouping
                    if configuration['hyperparameter_search'] == 'greedy':
                        branch_score = score(branch)
                        for operation in OPERATIONS[1:]:
                            values = hole_candidates(branch, region) if operation == 'hole_volume' else grids[operation]
                            # Regenerated grids can exclude a previously fitted numeric threshold.
                            op_best, op_score = None, -float('inf')
                            for value in values:
                                trial = deepcopy(branch)
                                trial['settings'][region][operation] = value
                                trial_score = evaluate(trial, region, sweep, operation, values)
                                if op_best is None or better(trial, trial_score, op_best, op_score):
                                    op_best, op_score = trial, trial_score
                            branch, branch_score = op_best, op_score
                        if better(branch, branch_score, local_best, local_score):
                            local_best, local_score = branch, branch_score
                    else:
                        for radius in grids['closing_radius']:
                            branch['settings'][region]['closing_radius'] = radius
                            holes = hole_candidates(branch, region)
                            for hole, minimum, maximum in product(holes, grids['min_volume'], grids['max_count']):
                                trial = deepcopy(branch)
                                trial['settings'][region].update(hole_volume=hole, min_volume=minimum, max_count=maximum)
                                trial_score = evaluate(trial, region, sweep, 'joint',
                                    dict(grids, hole_volume=holes))
                                if better(trial, trial_score, local_best, local_score):
                                    local_best, local_score = trial, trial_score
                if local_score > current_score + EPS:
                    current, current_score = local_best, local_score
                    improved = True
            if not improved:
                break
            limit_reached = sweep == 4
        report['runs'].append({'direction': direction, 'score': current_score, 'sweeps': sweep + 1,
                               'sweep_limit_reached': limit_reached})
        if limit_reached and progress:
            progress('Post-processing coordinate search reached the five-sweep limit.')
        if (current_score > winner_score + EPS or
            (winner['direction'] != 'identity' and better(current, current_score, winner, winner_score))):
            winner, winner_score = deepcopy(current), current_score
    winner['objective'] = {'name': 'mean case/region Dice', 'raw': baseline_score, 'fitted': winner_score}
    winner['candidate_grids'] = []
    final = deepcopy(winner)
    if final['direction'] == 'identity':
        final['direction'] = directions[0]
    for region in range(count):
        grouping = winner['settings'][region]['grouping_distance']
        winner['candidate_grids'].append(dict(contexts[region, grouping], hole_volume=hole_candidates(final, region)))
    winner['search_runs'] = report['runs']
    winner['fold_distributions'] = report['fold_distributions']
    winner['warnings'] = report['warnings']
    if progress:
        for warning in dict.fromkeys(report['warnings']):
            progress(warning)
    return winner, report
