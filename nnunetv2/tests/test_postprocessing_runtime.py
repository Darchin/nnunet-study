"""Parallel fitting, persistent cache integrity, and reporting acceptance."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nnunetv2.postprocessing.adaptive import (
    OPERATIONS, apply_policy, component_metadata, extract_case_components, fit_policy,
    fragmentation_gaps, label_spec, masks_from_segmentation, mean_dice,
)
from nnunetv2.postprocessing.runtime import (
    CachedCaseLoader, CaseExecutor, Progress, annotation_digest, atomic_array, atomic_json, file_digest, store_masks,
)
from nnunetv2.postprocessing.validation import inspect_prediction, prepare_reference, coordinated_prediction
from nnunetv2.postprocessing.workers import evaluate_case, prepare_gaps


DATASET = {'labels': {'background': 0, 'parent': [1, 2], 'child': 2, 'other': 3},
           'regions_class_order': [1, 2, 3]}


def example_cases(tmp_path):
    spec = label_spec(DATASET)
    references, cases = {}, {}
    for index in range(3):
        reference = np.zeros((18, 20), np.uint8)
        reference[3:9, 3:9] = 1
        reference[4:6, 4:6] = 2
        reference[12:15, 12:15] = 3
        if index == 2:
            reference[:] = 0
        masks = masks_from_segmentation(reference, spec)
        masks[0, 16, 18] = True
        masks[1, 2, 2] = True  # Inconsistent hierarchy tests the complete final-label-map score.
        identifier = str(index)
        store_masks(tmp_path / identifier, masks, [2., 1.])
        atomic_array(tmp_path / identifier / 'reference.npy', reference)
        references[identifier] = reference
        cases[identifier] = extract_case_components(reference, [2., 1.], spec)
    return spec, references, {'component_statistics': {'metadata': component_metadata(spec), 'cases': cases}}


@pytest.mark.parametrize('mode', ['greedy', 'exhaustive'])
def test_serial_and_spawn_fitting_are_identical(tmp_path, mode):
    spec, references, fingerprint = example_cases(tmp_path)
    kwargs = dict(configuration={'hyperparameter_search': mode}, progress=Progress(lambda _: None, False))
    first, serial = fit_policy(sorted(references), CachedCaseLoader(str(tmp_path)), fingerprint, spec,
                              num_processes=1, **kwargs)
    second, parallel = fit_policy(sorted(references), CachedCaseLoader(str(tmp_path)), fingerprint, spec,
                                 num_processes=2, **kwargs)
    assert first == second
    assert serial['trials'] == parallel['trials']
    assert serial['case_summaries'] == parallel['case_summaries']
    assert serial['execution']['processes'] == 1 and parallel['execution']['processes'] == 2
    assert parallel['execution']['case_tasks'] > 0


def test_batch_loads_once_and_metrics_match_final_map(tmp_path):
    spec, references, _ = example_cases(tmp_path)
    source = CachedCaseLoader(str(tmp_path))
    calls = []

    def load(identifier):
        calls.append(identifier)
        return source(identifier)

    settings = [{operation: None for operation in OPERATIONS} for _ in spec['regions']]
    policies = [{'direction': direction, 'settings': deepcopy(settings),
                 'connectivity': {'foreground': 26, 'background': 'full'}}
                for direction in ('identity', 'expand', 'restrict')]
    values = evaluate_case('0', load, spec, policies)
    assert calls == ['0']
    masks, reference, spacing = source('0')
    for policy, metrics in zip(policies, values):
        expected = mean_dice(apply_policy(masks, spacing, spec, policy), masks_from_segmentation(reference, spec), spec)
        assert np.mean(np.asarray(metrics)[:, 0]) == pytest.approx(expected, abs=1e-15)


def test_fragmentation_preparation_reuses_fingerprint_tree(tmp_path, monkeypatch):
    spec = label_spec({'labels': {'background': 0, 'lesion': 1}})
    reference = np.zeros((20, 20), np.uint8)
    reference[2:8, 2:8] = reference[2:8, 10:16] = 1
    prediction = reference[None].astype(bool)
    prediction[:, 4:6] = False
    record = extract_case_components(reference, [1, 1], spec)['connectivities']['8']
    loader = lambda _: (prediction, reference, [1, 1])
    grids = [[None, 3]]
    expected = [[fragmentation_gaps(prediction[0], reference > 0, [1, 1], 26, distance)
                 for distance in grids[0]]]
    result = prepare_gaps('case', loader, spec, record, grids)
    assert result == expected


def cache_prediction(tmp_path):
    spec = label_spec({'labels': {'background': 0, 'lesion': 1}})
    identity = {'model_digest': 'model', 'enable_tta': False, 'labels': spec}
    store_masks(tmp_path / 'cache', np.ones((1, 5, 5), bool), [1, 1])
    atomic_array(tmp_path / 'prediction.npy', np.ones((5, 5), np.uint8))
    atomic_json(tmp_path / 'cache' / 'prediction.json', {'identity': identity,
        'mask_digest': file_digest(tmp_path / 'cache' / 'masks.npy'),
        'geometry_digest': file_digest(tmp_path / 'cache' / 'geometry.json'),
        'prediction_digest': file_digest(tmp_path / 'prediction.npy')})
    return identity


@pytest.mark.parametrize('change', ['model', 'tta', 'missing', 'mask_corruption', 'geometry', 'prediction'])
def test_verified_cache_reuse_and_invalidation(tmp_path, change):
    identity = cache_prediction(tmp_path)
    assert inspect_prediction(tmp_path / 'cache', tmp_path / 'prediction.npy', identity)
    if change in ('model', 'tta'):
        identity['model_digest' if change == 'model' else 'enable_tta'] = 'different'
    elif change == 'missing':
        (tmp_path / 'cache' / 'prediction.json').unlink()
    elif change == 'mask_corruption':
        atomic_array(tmp_path / 'cache' / 'masks.npy', np.zeros(4, np.uint8))
    elif change == 'geometry':
        atomic_json(tmp_path / 'cache' / 'geometry.json', {'shape': [1, 25, 1], 'spacing': [1, 1]})
    else:
        atomic_array(tmp_path / 'prediction.npy', np.zeros((5, 5), np.uint8))
    assert not inspect_prediction(tmp_path / 'cache', tmp_path / 'prediction.npy', identity)


class Reader:
    reads = 0

    def read_seg(self, path):
        type(self).reads += 1
        return np.load(path)[None], {'spacing': [1., 1.]}


def test_reference_checks_digest_and_reuses_mmap_without_trees(tmp_path, monkeypatch):
    spec = label_spec({'labels': {'background': 0, 'lesion': 1}})
    reference = np.eye(5, dtype=np.uint8)
    atomic_array(tmp_path / 'gt.npy', reference)
    store_masks(tmp_path / 'cache', reference[None].astype(bool), [1, 1])
    record = extract_case_components(reference, [1, 1], spec)
    Reader.reads = 0
    prepare_reference('case', tmp_path / 'cache', str(tmp_path / 'gt.npy'), Reader, record)
    prepare_reference('case', tmp_path / 'cache', str(tmp_path / 'gt.npy'), Reader, record)
    assert Reader.reads == 1
    atomic_array(tmp_path / 'cache' / 'reference.npy', np.zeros_like(reference))
    prepare_reference('case', tmp_path / 'cache', str(tmp_path / 'gt.npy'), Reader, record)
    assert Reader.reads == 2  # Corrupt cached reference arrays are regenerated, including on Windows.
    assert annotation_digest(reference.astype(np.float32)) == record['annotation_digest']
    atomic_array(tmp_path / 'gt.npy', np.zeros_like(reference))
    with pytest.raises(ValueError, match='Stale component'):
        prepare_reference('case', tmp_path / 'cache', str(tmp_path / 'gt.npy'), Reader, record)


def test_journal_summary_and_visible_phase_events(tmp_path):
    spec, references, fingerprint = example_cases(tmp_path / 'cache')
    messages = []
    path = tmp_path / 'training' / 'postprocessing_search.json'
    policy, report = fit_policy(sorted(references), CachedCaseLoader(str(tmp_path / 'cache')), fingerprint, spec,
        num_processes=1, progress=Progress(messages.append, False), report_path=path)
    assert json.loads(path.read_text())['status'] == 'complete'
    journal = [json.loads(line) for line in path.with_suffix('.jsonl').read_text().splitlines()]
    assert journal == report['trials']
    assert 'region_dice' in journal[0] and 'pipeline_settings' in journal[0]
    assert len(report['case_summaries']['raw']['metric_per_case']) == 3
    assert 'Preparing fragmentation candidates' in messages
    assert not any(': starting' in message or ': complete (' in message for message in messages)
    assert not any('cases complete;' in message for message in messages)
    assert 'fold_distributions' not in policy  # Keep inference/export payloads small.


def test_worker_failure_is_propagated():
    with CaseExecutor(num_processes=2, progress=Progress(lambda _: None, False)) as executor:
        with pytest.raises(ValueError):
            executor.map(int, [('not an integer',)], 'Failing task')


def test_dead_worker_detection():
    pool = SimpleNamespace(_pool=[SimpleNamespace(is_alive=lambda: False)])
    with CaseExecutor(pool, 1, Progress(lambda _: None, False)) as executor:
        with pytest.raises(RuntimeError, match='worker died'):
            executor.map(int, [('1',)], 'Dead worker')


def test_ddp_prediction_heartbeats_cover_uneven_cached_assignments(monkeypatch):
    from nnunetv2.postprocessing import validation
    monkeypatch.setattr(validation.dist, 'get_world_size', lambda: 2)

    def gather(statuses, own):
        statuses[:] = [own, {'done': True}]

    monkeypatch.setattr(validation.dist, 'all_gather_object', gather)
    trainer = SimpleNamespace(is_ddp=True, device=SimpleNamespace(type='cpu'))
    coordinated_prediction(lambda stop: None, trainer, .001)

    def fail(stop):
        raise ValueError('failed export')

    with pytest.raises(RuntimeError, match='failed export'):
        coordinated_prediction(fail, trainer, .001)


def test_process_environment_override(monkeypatch):
    import nnunetv2.configuration as configuration
    monkeypatch.setenv('nnUNet_n_post_proc', '3')
    importlib.reload(configuration)
    assert configuration.default_num_postprocessing_processes == 3
    monkeypatch.delenv('nnUNet_n_post_proc')
    importlib.reload(configuration)
    assert configuration.default_num_postprocessing_processes == 8


def test_fingerprint_always_extracts_all_statistics_with_plain_progress(tmp_path, monkeypatch):
    from nnunetv2.experiment_planning.dataset_fingerprint import percentile_fingerprint_extractor as module
    from nnunetv2.postprocessing import runtime
    reference = np.eye(5, dtype=np.uint8)
    label = tmp_path / 'label.npy'
    atomic_array(label, reference)
    dataset_json = {'labels': {'background': 0, 'lesion': 1}, 'file_ending': '.npy'}
    spec = label_spec(dataset_json)
    case = extract_case_components(reference, [1, 1], spec)
    case['connectivities']['4'] = deepcopy(case['connectivities']['8'])
    case.pop('annotation_digest')
    stat = label.stat()
    case['source'] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    old_metadata = dict(component_metadata(spec), version=2, spec=spec,
                        foreground_connectivities=[6, 18, 26], **{'2d_connectivities': [4, 8]})
    old_metadata.pop('annotation_digest')
    existing = {'spacings': [[1, 1]], 'component_statistics': {'metadata': old_metadata, 'cases': {'case': case}}}
    base_calls = []
    def base_run(self, overwrite_existing=False):
        base_calls.append(overwrite_existing)
        return deepcopy(existing)
    monkeypatch.setattr(module.DatasetFingerprintExtractor, 'run', base_run)
    monkeypatch.setattr(module, 'determine_reader_writer_from_dataset_json', lambda *args: Reader)
    monkeypatch.setattr(module, 'nnUNet_preprocessed', str(tmp_path))

    scans = []
    extract = module.extract_case_components
    def scan(*args):
        scans.append(1)
        return extract(*args)
    monkeypatch.setattr(module, 'extract_case_components', scan)
    updates = []
    real_bar = runtime.tqdm

    class Bar(real_bar):
        def update(self, n=1):
            updates.append(n)
            return super().update(n)

    monkeypatch.setattr(runtime, 'tqdm', Bar)
    extractor = module.PercentileFingerprintExtractor.__new__(module.PercentileFingerprintExtractor)
    extractor.dataset_json, extractor.dataset_name, extractor.num_processes = dataset_json, 'Dataset001_Test', 1
    extractor.show_progress_bar = True
    extractor.dataset = {'case': {'images': ['unused.npy'], 'label': str(label)}}
    (tmp_path / extractor.dataset_name).mkdir()
    refreshed = extractor.run()
    statistics = refreshed['component_statistics']
    assert set(statistics['cases']['case']['connectivities']) == {'8'}
    assert statistics['cases']['case']['annotation_digest'] == annotation_digest(reference)
    assert 'version' not in statistics['metadata'] and 'spec' not in statistics['metadata']
    assert statistics['metadata']['labels'] == dataset_json['labels']
    assert sum(updates) == 1
    assert base_calls == [True] and scans == [1]
    extractor.run()
    assert base_calls == [True, True] and scans == [1, 1]


def test_cached_morphology_is_shared_and_not_changed_by_filtering(monkeypatch):
    from nnunetv2.postprocessing import adaptive
    spec = label_spec({'labels': {'background': 0, 'lesion': 1}})
    masks = np.zeros((1, 20, 20), bool)
    masks[0, 3:9, 3:9] = True
    masks[0, 17, 17] = True
    masks.flags.writeable = False
    cache = adaptive.MorphologyCache()
    calls = []
    original = adaptive.repair_masks

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(adaptive, 'repair_masks', counted)
    first = {'direction': 'expand', 'connectivity': {'foreground': 26, 'background': 'full'},
             'settings': [{op: None for op in OPERATIONS}]}
    second = deepcopy(first)
    second['settings'][0]['min_volume'] = 2
    original_result = apply_policy(masks, [1, 1], spec, first, cache)
    filtered = apply_policy(masks, [1, 1], spec, second, cache)
    np.testing.assert_equal(apply_policy(masks, [1, 1], spec, first, cache), original_result)
    assert len(calls) == 1
    assert original_result[17, 17] == 1 and filtered[17, 17] == 0


@pytest.mark.parametrize('spacing,radius', [([1., 1.], 1.), ([2., 1.], 2.5),
                                          ([1., 1., 1.], 1.5), ([3., 2., 1.], 2.5)])
def test_localized_closing_matches_dense_reference(spacing, radius):
    from nnunetv2.postprocessing.adaptive import close_mask, _close_dense
    for seed in range(10):
        shape = (40,) * len(spacing)
        mask = np.random.default_rng(seed).random(shape) < .0005
        mask[(slice(0, 5),) * len(spacing)] = True  # Original-image boundary behavior.
        mask[(slice(15, 21),) * len(spacing)] = True
        mask[(slice(17, 18),) * len(spacing)] = False
        mask[(slice(24, 30),) * len(spacing)] = True
        expected = _close_dense(mask, np.asarray(spacing), radius)
        np.testing.assert_equal(close_mask(mask, spacing, radius), expected)


@pytest.mark.parametrize('spacing', [[2., 1.], [3., 2., 1.]])
def test_spatial_prediction_groups_equal_exact_mst_groups(spacing):
    from nnunetv2.postprocessing.components import (
        component_measurements, grouped_components, group_membership, proximity_membership, boundary_points,
    )
    for seed in range(5):
        mask = np.random.default_rng(seed).random((18,) * len(spacing)) < .015
        mask[(slice(3, 9),) * len(spacing)] = True
        components, volumes, edges = component_measurements(mask, spacing)
        points = boundary_points(components, spacing)
        for threshold in [0., 1., 2., 3., 6., np.inf, *edges[:3, 2]]:
            expected, sizes = group_membership(volumes, edges, threshold)
            mapping, actual_sizes = proximity_membership(volumes, points, threshold)
            np.testing.assert_equal(mapping, expected)
            np.testing.assert_allclose(actual_sizes, sizes, atol=0, rtol=0)
            actual, _ = grouped_components(mask, spacing, 26, threshold)
            np.testing.assert_equal(actual, expected[components])


def test_training_summary_uses_nnunet_region_keys():
    from nnunetv2.postprocessing.workers import summarize_case_metrics
    spec = label_spec({'labels': {'background': 0, 'whole': [1, 2], 'core': [2]},
                       'regions_class_order': [1, 2]})
    summary = summarize_case_metrics(['case'], np.ones((1, 2, 8)), spec)
    assert set(summary['mean']) == {'(1, 2)', '(2,)'}
    assert set(summary['metric_per_case'][0]['metrics']) == set(summary['mean'])
