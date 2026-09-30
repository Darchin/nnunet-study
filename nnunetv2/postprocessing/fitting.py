"""Fold-isolated candidate generation and bounded region coordinate searches."""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import tempfile
import traceback

import numpy as np

from nnunetv2.postprocessing.adaptive import (
    VERSION, EPS, OPERATIONS, validate_fingerprint, case_regions, relations, group_membership, policy_rank,
)
from nnunetv2.postprocessing.configuration import (
    OPERATION_SELECTORS, QUANTILE_CONVENTIONS, resolve_configuration, resolve_percentiles,
    distribution_candidates, percentile_table,
)
from nnunetv2.postprocessing.runtime import (
    CachedCaseLoader, CaseExecutor, Progress, atomic_array, atomic_json, store_masks,
)
from nnunetv2.postprocessing.workers import prepare_gaps, measure_holes, evaluate_case, summarize_case_metrics


def fit_policy(case_ids, load_case, fingerprint, spec, metadata=None, progress=None, configuration=None,
               executor=None, num_processes=None, report_path=None):
    """Materialize arbitrary Python loaders once so spawn workers need only case paths."""
    progress = progress if isinstance(progress, Progress) else Progress(progress)
    with tempfile.TemporaryDirectory(prefix='nnunet-postprocessing-') as temporary:
        if not isinstance(load_case, CachedCaseLoader):
            with progress.phase('Preparing fitting case files'):
                for identifier in sorted(case_ids):
                    masks, reference, spacing = load_case(identifier)
                    directory = Path(temporary) / identifier
                    store_masks(directory, masks, spacing)
                    atomic_array(directory / 'reference.npy', reference)
            load_case = CachedCaseLoader(temporary)
        try:
            if executor is not None:
                return _fit_policy(case_ids, load_case, fingerprint, spec, metadata, progress, configuration,
                                   executor, report_path)
            with CaseExecutor(num_processes=num_processes, progress=progress) as executor:
                return _fit_policy(case_ids, load_case, fingerprint, spec, metadata, progress, configuration,
                                   executor, report_path)
        except BaseException as error:
            # Release arrays retained by failed task tracebacks before Windows removes owned temporary files.
            traceback.clear_frames(error.__traceback__)
            raise


def _fit_policy(case_ids, load_case, fingerprint, spec, metadata, progress, configuration, executor, report_path):
    """Stream only explicitly supplied training cases; global summaries are never read."""
    case_ids = sorted(case_ids)
    if not case_ids:
        raise ValueError('Adaptive fitting requires at least one training case.')
    configuration = resolve_configuration(configuration)
    connectivity, aggregation = 26, configuration['aggregation']
    selectors = {op: resolve_percentiles(configuration[key]) for op, key in OPERATION_SELECTORS.items()}
    statistics = validate_fingerprint(fingerprint, spec, case_ids, connectivity)
    count = len(spec['regions'])
    report = {'configuration': configuration, 'quantile_conventions': QUANTILE_CONVENTIONS,
              'label_definitions': spec,
              'training_identifiers': case_ids, 'metadata': metadata or {},
              'fold_distributions': [], 'fixed_candidate_grids': [], 'trials': [], 'runs': [],
              'warnings': []}
    contexts, grouping_grids = {}, []
    journal = Path(report_path).with_suffix('.jsonl') if report_path else None
    if journal:
        journal.parent.mkdir(parents=True, exist_ok=True)
        journal.write_text('', encoding='utf-8')

    def save_partial(status='fitting'):
        if report_path:
            atomic_json(report_path, dict(report, status=status))

    def grid(values, operation, discrete=False, scale=1.):
        return distribution_candidates(values, selectors[operation], aggregation, discrete, scale)

    def describe(values, discrete=False):
        return {'per_case': {k: list(map(int if discrete else float, v)) for k, v in zip(case_ids, values)},
                'percentiles': percentile_table(values, aggregation, discrete)}

    save_partial('preparing candidates')

    records_by_case = [case_regions(statistics['cases'][k], connectivity) for k in case_ids]
    for region in range(count):
        distances = [[e[2] for e in r[region]['edges']] for r in records_by_case]
        grouping_grids.append(grid(distances, 'grouping_distance'))
    if selectors['closing_radius']:
        gaps_by_case = executor.map(prepare_gaps,
            [(k, load_case, spec, record, grouping_grids) for k, record in zip(case_ids, records_by_case)],
            'Preparing fragmentation candidates')
    else:
        gaps_by_case = [[[[] for _ in g] for g in grouping_grids] for _ in case_ids]

    for region in range(count):
        records = [case_regions(statistics['cases'][k], connectivity)[region] for k in case_ids]
        distances = [[e[2] for e in r['edges']] for r in records]
        grouping_grid = grouping_grids[region]
        description = {'grouping_distances': describe(distances), 'grouping_grid': grouping_grid, 'contexts': []}
        for grouping_index, grouping in enumerate(grouping_grid):
            sizes = [group_membership(np.asarray(r['volumes']), r['edges'], grouping)[1].tolist() for r in records]
            counts = [[len(v)] if v else [] for v in sizes]
            gaps = [case[region][grouping_index] for case in gaps_by_case]
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
    save_partial('candidate grids prepared')

    base = {'version': VERSION, 'spec': spec, 'direction': 'identity',
            'settings': [{op: None for op in OPERATIONS} for _ in range(count)],
            'training_identifiers': case_ids, 'metadata': metadata or {}, 'configuration': configuration,
            'percentile_grids': selectors, 'quantile_conventions': QUANTILE_CONVENTIONS,
            'search': f"bounded region coordinate search ({configuration['hyperparameter_search']})",
            'connectivity': {'foreground': connectivity, 'background': 'full'}}
    score_cache, hole_cache, metric_cache, trial_keys = {}, {}, {}, set()
    voxel_floor = min(float(np.prod(case['spacing'][1:] if len(case['shape']) == 3 and
        case['shape'][0] == 1 and case['spacing'][0] == 999 else case['spacing']))
        for case in (statistics['cases'][k] for k in case_ids))

    def policy_key(policy):
        return policy['direction'], tuple(tuple(s[op] for op in OPERATIONS) for s in policy['settings'])

    def score_key(policy):
        settings = []
        for original in policy['settings']:
            s = dict(original)
            if s['min_volume'] is not None and s['min_volume'] <= voxel_floor:
                s['min_volume'] = None
            # Grouping has no voxel effect until a size/count rule consumes it.
            if s['min_volume'] is None and s['max_count'] is None:
                s['grouping_distance'] = None
            settings.append(tuple(s[op] for op in OPERATIONS))
        return policy['direction'], tuple(settings)

    def score_many(policies, description='Scoring candidate batch', retain_cases=False):
        unique = {score_key(p): p for p in policies if retain_cases or score_key(p) not in score_cache}
        if unique:
            trials = list(unique.values())
            totals = np.zeros((len(trials), count, 8), dtype=float)
            retained = [[] for _ in trials] if retain_cases else None

            def consume(index, result):
                values = np.asarray(result, dtype=float)
                totals[:] += values
                if retained is not None:
                    for j in range(len(trials)):
                        retained[j].append(values[j].tolist())

            executor.map(evaluate_case, [(k, load_case, spec, trials) for k in case_ids],
                         f'{description} ({len(trials)} policies)', consume)
            for index, key in enumerate(unique):
                means = totals[index] / len(case_ids)
                score_cache[key] = float(means[:, 0].mean())
                metric_cache[key] = means[:, 0].tolist()
                if retained is not None:
                    report.setdefault('case_summaries', {})['raw' if trials[index]['direction'] == 'identity'
                        else 'selected'] = summarize_case_metrics(case_ids, retained[index], spec)
        return [score_cache[score_key(p)] for p in policies]

    def score(policy):
        return score_many([policy])[0]

    def hole_key(policy, region):
        # Only morphology through this region affects its pre-fill mask; grouping/filtering do not.
        return (region, policy['direction'], tuple((s['closing_radius'], s['hole_volume'] if i < region else None)
                    for i, s in enumerate(policy['settings'][:region + 1])))

    def prepare_holes(policies, region):
        if not selectors['hole_volume']:
            return
        unique = {hole_key(p, region): p for p in policies if hole_key(p, region) not in hole_cache}
        if not unique:
            return
        trials = list(unique.values())
        results = executor.map(measure_holes, [(k, load_case, spec, trials, region) for k in case_ids],
                               f'Measuring cavities, region {region + 1}')
        for index, (key, policy) in enumerate(unique.items()):
            holes = [case[index] for case in results]
            hole_cache[key] = grid(holes, 'hole_volume')
            report.setdefault('filling_contexts', []).append({'region': region, 'direction': policy['direction'],
                'settings': deepcopy(policy['settings']), 'cavities': describe(holes), 'grid': hole_cache[key],
                'duplicate_candidate_count': len(selectors['hole_volume']) - len(hole_cache[key]) + 1
                                            if any(holes) else 0})

    def hole_candidates(policy, region):
        if not selectors['hole_volume']:
            return [None]
        prepare_holes([policy], region)
        return hole_cache[hole_key(policy, region)]

    def better(trial, trial_score, incumbent, incumbent_score):
        return (trial_score > incumbent_score + EPS or
                (abs(trial_score - incumbent_score) <= EPS and policy_rank(trial) < policy_rank(incumbent)))

    def evaluate(trial, region, sweep, operation, grid_values):
        value = score(trial)
        key = policy_key(trial)
        if key in trial_keys:
            return value
        trial_keys.add(key)
        entry = {'direction': trial['direction'], 'sweep': sweep + 1, 'region': region,
                                'operation': operation, 'settings': deepcopy(trial['settings'][region]),
                                'pipeline_settings': deepcopy(trial['settings']), 'grid': grid_values,
                                'score': value, 'region_dice': metric_cache[score_key(trial)]}
        report['trials'].append(entry)
        if journal:
            with journal.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(entry) + '\n')
        return value

    baseline_score = score_many([base], 'Scoring raw training predictions', retain_cases=True)[0]
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
                local_best, local_score = deepcopy(current), current_score
                branches = []
                for grouping in grouping_grids[region]:
                    branch = deepcopy(current)
                    branch['settings'][region]['grouping_distance'] = grouping
                    branches.append(branch)
                if configuration['hyperparameter_search'] == 'greedy':
                    for operation in OPERATIONS[1:]:
                        if operation == 'hole_volume':
                            prepare_holes(branches, region)
                        alternatives = []
                        for index, branch in enumerate(branches):
                            grouping = branch['settings'][region]['grouping_distance']
                            values = (hole_candidates(branch, region) if operation == 'hole_volume'
                                      else contexts[region, grouping][operation])
                            for value in values:
                                trial = deepcopy(branch)
                                trial['settings'][region][operation] = value
                                alternatives.append((index, trial, values))
                        score_many([trial for _, trial, _ in alternatives],
                                   f'{direction}, sweep {sweep + 1}, region {region + 1}: {operation}, percentiles {selectors[operation]}')
                        best = {}
                        for index, trial, values in alternatives:
                            value = evaluate(trial, region, sweep, operation, values)
                            if index not in best or better(trial, value, *best[index]):
                                best[index] = trial, value
                        branches = [best[index][0] for index in range(len(branches))]
                    for branch in branches:
                        value = score(branch)
                        if better(branch, value, local_best, local_score):
                            local_best, local_score = branch, value
                else:
                    upstream = []
                    for branch in branches:
                        grouping = branch['settings'][region]['grouping_distance']
                        for radius in contexts[region, grouping]['closing_radius']:
                            trial = deepcopy(branch)
                            trial['settings'][region]['closing_radius'] = radius
                            upstream.append(trial)
                    prepare_holes(upstream, region)
                    alternatives = []
                    for branch in upstream:
                        grouping = branch['settings'][region]['grouping_distance']
                        grids = contexts[region, grouping]
                        holes = hole_candidates(branch, region)
                        for hole, minimum, maximum in product(holes, grids['min_volume'], grids['max_count']):
                            trial = deepcopy(branch)
                            trial['settings'][region].update(hole_volume=hole, min_volume=minimum, max_count=maximum)
                            alternatives.append((trial, dict(grids, hole_volume=holes)))
                    score_many([trial for trial, _ in alternatives],
                               f'{direction}, sweep {sweep + 1}, region {region + 1}: joint search')
                    for trial, grids in alternatives:
                        value = evaluate(trial, region, sweep, 'joint', grids)
                        if better(trial, value, local_best, local_score):
                            local_best, local_score = trial, value
                if local_score > current_score + EPS:
                    current, current_score = local_best, local_score
                    improved = True
                save_partial()
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
    # Raw fold distributions belong in the permanent report, not every inference worker's policy payload.
    if report_path:
        winner['fitting_report'] = 'training/postprocessing_search.json'
    winner['warnings'] = report['warnings']
    score_many([winner], 'Evaluating selected training policy', retain_cases=True)
    if winner['direction'] == 'identity':
        report['case_summaries']['selected'] = deepcopy(report['case_summaries']['raw'])
    report['selected_policy'] = {'direction': winner['direction'], 'settings': winner['settings'],
                                 'score': winner_score}
    report['execution'] = dict(executor.statistics)
    report['status'] = 'complete'
    save_partial('complete')
    return winner, report
