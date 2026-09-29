"""Configurable connectivity, grouping, weighting, and search acceptance tests."""
from copy import deepcopy
from itertools import product

import numpy as np
import pytest
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist

from nnunetv2.postprocessing.adaptive import (
    apply_policy, case_regions, component_metadata, extract_case_components, fingerprint_summaries,
    fit_policy, fragmentation_gaps, label_spec, load_policy, OPERATIONS, VERSION, close_mask, validate_fingerprint,
)
from nnunetv2.postprocessing.components import (
    boundary_points, component_measurements, component_volumes, distance_tree,
    effective_connectivity, grouped_components, group_membership,
)
from nnunetv2.postprocessing.configuration import (
    DEFAULTS, PERCENTILE_CATALOGUE, distribution_candidates, distribution_quantiles,
    resolve_configuration, resolve_percentiles,
)
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

DATASET = {'labels': {'background': 0, 'lesion': 1}}


def policy(spec, **settings):
    return {'version': VERSION, 'spec': spec, 'direction': 'expand',
            'connectivity': {'foreground': 26, 'background': 'full'},
            'settings': [{op: settings.get(op) for op in OPERATIONS}]}


def fit(references, predictions=None, configuration=None):
    spec = label_spec(DATASET)
    predictions = references if predictions is None else predictions
    cases = {k: extract_case_components(v, (1, 1), spec) for k, v in references.items()}
    fingerprint = {'component_statistics': {'metadata': component_metadata(spec), 'cases': cases}}
    load = lambda k: (predictions[k][None].astype(bool), references[k], (1, 1))
    return fit_policy(sorted(references), load, fingerprint, spec, configuration=configuration)


def test_catalogue_and_signed_selectors():
    assert PERCENTILE_CATALOGUE == [1.] + np.arange(2.5, 100, 2.5).tolist() + [99., 100.]
    assert resolve_percentiles(-3) == [1, 2.5, 5]
    assert resolve_percentiles(3) == [97.5, 99, 100]
    assert resolve_percentiles([50, 0, 50, 1.25, 100]) == [0, 1.25, 50, 100]
    assert resolve_percentiles(iter([25, 50])) == [25, 50]
    assert resolve_configuration({'size_percentiles': iter([10, 25])})['size_percentiles'] == [10, 25]
    assert resolve_percentiles(None) == resolve_percentiles([]) == []


@pytest.mark.parametrize('value', [True, False, 0, 99, -99, [True], [np.nan], [np.inf], [-1], [101],
                                  '3', 3.5, {'a': 1}])
def test_bad_percentile_selectors(value):
    with pytest.raises(ValueError):
        resolve_percentiles(value)


@pytest.mark.parametrize('value', [{'unknown': 1}, {'connectivity': 4}, {'connectivity': True},
                                  {'aggregation': 'median'}, {'hierarchy_repair': 'both'},
                                  {'hyperparameter_search': 'global'}, {'size_percentiles': 0}, []])
def test_bad_configuration(value):
    with pytest.raises(ValueError):
        resolve_configuration(value)


def test_configuration_defaults_inheritance_and_explicit_null():
    plans = PlansManager({'configurations': {'base': {'post_processing': {'size_percentiles': [10],
        'connectivity': 18}}, 'child': {'inherits_from': 'base', 'post_processing': {'size_percentiles': None}},
        'replace': {'inherits_from': 'base', 'nested_override': False,
                    'post_processing': {'count_percentiles': -2}}}})
    inherited = plans._internal_resolve_configuration_inheritance('child')
    assert inherited['post_processing'] == {'size_percentiles': None, 'connectivity': 18}
    # A complete architecture isn't needed to test the configuration property.
    config = ConfigurationManager.__new__(ConfigurationManager)
    config.configuration = inherited
    assert config.post_processing['size_percentiles'] is None
    assert config.post_processing['connectivity'] == 18
    replaced = plans._internal_resolve_configuration_inheritance('replace')['post_processing']
    assert resolve_configuration(replaced)['connectivity'] == 26
    assert resolve_configuration(replaced)['count_percentiles'] == [1, 2.5]
    config.configuration = {}
    assert config.post_processing == DEFAULTS


@pytest.mark.parametrize('connectivity,count', [(6, 3), (18, 2), (26, 1)])
def test_3d_face_edge_corner_connectivity(connectivity, count):
    mask = np.zeros((4, 4, 4), bool)
    mask[0, 0, 0] = mask[1, 1, 0] = mask[2, 2, 1] = True
    components, volumes = component_volumes(mask, (2, 3, 4), connectivity)
    assert len(volumes) == count and volumes.sum() == 72
    assert components.max() == count


@pytest.mark.parametrize('requested,effective', [(6, 4), (8, 8), (18, 8), (26, 8)])
def test_2d_connectivity_and_native_singleton_axis(requested, effective):
    mask = np.eye(3, dtype=bool)[None]
    assert effective_connectivity(2, requested) == effective
    _, volumes = component_volumes(mask, (999, 2, 3), requested)
    assert len(volumes) == (3 if effective == 4 else 1) and volumes.sum() == 18
    with pytest.raises(ValueError, match='invalid'):
        component_volumes(np.ones((3, 3, 3), bool), (1, 1, 1), 8)


def test_exact_tree_distances_against_all_pairs_and_determinism():
    rng = np.random.default_rng(123)
    points = [rng.normal(size=(n, 3)) + 8 * i for i, n in enumerate([2, 12, 1, 4, 6])]
    tree = distance_tree(points)
    weights = np.zeros((len(points), len(points)))
    for i in range(len(points)):
        for j in range(i):
            weights[i, j] = weights[j, i] = cdist(points[i], points[j]).min()
    assert tree.shape == (4, 3)
    assert tree[:, 2].sum() == pytest.approx(minimum_spanning_tree(weights).data.sum())
    np.testing.assert_equal(tree, distance_tree(points))
    tied = [np.asarray([p], dtype=float) for p in product((0, 2), repeat=2)]
    np.testing.assert_equal(distance_tree(tied), distance_tree(tied))


def test_anisotropic_tree_threshold_chains_and_volume_preservation():
    mask = np.zeros((8, 8), bool)
    mask[1, 1] = mask[3, 1] = mask[5, 1] = True
    components, volumes, edges = component_measurements(mask, (2, 1), 26)
    assert edges[:, 2].tolist() == [4, 4]
    separate, sizes = grouped_components(mask, (2, 1), 26, 3.999)
    assert len(sizes) == 3
    combined, sizes = grouped_components(mask, (2, 1), 26, 4)
    assert len(sizes) == 1 and sizes[0] == 6
    np.testing.assert_equal(combined > 0, mask)
    assert np.max(combined) == 1  # The end objects are 8 mm apart: chains are intentional.
    assert boundary_points(components, (2, 1))[0].tolist() == [[2, 1]]
    assert group_membership(np.empty(0), [], 4)[1].size == 0


def test_case_weights_and_single_case_linear_convention():
    observations = [[1, 1, 1, 1], [100]]
    instance = distribution_quantiles(observations, [75], 'instance')[0]
    case = distribution_quantiles(observations, [75], 'case')[0]
    assert instance == 1 and case > 1
    np.testing.assert_allclose(distribution_quantiles([[1, 3, 8]], [1, 25, 50, 99], 'case'),
                               np.percentile([1, 3, 8], [1, 25, 50, 99]))
    assert distribution_candidates([[], [1, 1]], [1, 5, 10], 'case') == [None, 1]
    assert distribution_candidates([[], []], [50], 'case') == [None]
    assert distribution_candidates([[1], [3]], [90, 95, 100], 'case', True) == [None, 3]


def test_fingerprint_variants_tables_summaries_and_empty_classes():
    reference = np.zeros((9, 9, 9), np.uint8)
    reference[1, 1, 1] = reference[2, 2, 2] = 1
    spec = label_spec(DATASET)
    case = extract_case_components(reference, (1, 1, 1), spec)
    assert set(case['connectivities']) == {'6', '18', '26'}
    face, full = case_regions(case, 6)[0], case_regions(case, 26)[0]
    assert face['component_ids'] == [1, 2] and full['component_ids'] == [1]
    assert face['edges'][0][2] == pytest.approx(np.sqrt(3))
    assert len(face['volume_percentiles']) == len(PERCENTILE_CATALOGUE)
    assert full['distance_percentiles'] == {}
    summaries = fingerprint_summaries({'positive': case}, 1)
    assert summaries['26']['case'][0]['count_percentiles']['100.0'] == 1
    empty = extract_case_components(np.zeros_like(reference), (1, 1, 1), spec)
    assert case_regions(empty, 26)[0]['volumes'] == []


def test_grouping_changes_fragmentation_reference_matching():
    reference = np.zeros((10, 15), bool)
    reference[3:6, 2:5] = reference[3:6, 7:10] = True
    assert fragmentation_gaps(reference, reference, (1, 1), 26) == []
    assert fragmentation_gaps(reference, reference, (1, 1), 26, 3) == [3]


@pytest.mark.parametrize('mode', ['greedy', 'exhaustive'])
def test_grouping_is_fitted_through_downstream_filters(mode):
    reference = np.zeros((20, 20), np.uint8)
    reference[5, 5] = reference[5, 7] = 1
    prediction = reference.copy()
    prediction[17, 17] = 1
    config = {'hyperparameter_search': mode, 'closing_percentiles': None,
              'filling_percentiles': None, 'count_percentiles': None, 'hierarchy_repair': 'child'}
    fitted, report = fit({'train': reference}, {'train': prediction}, config)
    assert fitted['objective']['fitted'] == 1
    assert fitted['settings'][0]['grouping_distance'] == 2
    assert fitted['settings'][0]['min_volume'] == 2
    assert len(report['runs']) == 1 and report['runs'][0]['direction'] == 'expand'
    assert report['fold_distributions'][0]['contexts'][0]['group_volumes']['per_case']['train'] == [1, 1]
    assert report['fold_distributions'][0]['contexts'][1]['group_volumes']['per_case']['train'] == [2]
    assert report['fold_distributions'][0]['contexts'][1]['positive_case_counts']['per_case']['train'] == [1]
    assert fitted['warnings']


def test_grouping_recomputed_after_closing_and_count_removal():
    spec = label_spec(DATASET)
    mask = np.zeros((1, 15, 20), bool)
    mask[0, 3:8, 2:5] = mask[0, 3:8, 6:9] = True
    mask[0, 12, 17] = True
    result = apply_policy(mask, (1, 1), spec, policy(spec, grouping_distance=2, closing_radius=1.5,
                                                  min_volume=25, max_count=1))
    assert np.all(result[mask[0] & (np.indices(mask.shape[1:])[0] < 10)])
    assert result[5, 5] == 1 and result[12, 17] == 0


def test_no_measurements_and_all_disabled_steps_leave_identity():
    empty = np.zeros((8, 8), np.uint8)
    config = {key: None for key in DEFAULTS if key.endswith('_percentiles')}
    fitted, report = fit({'train': empty}, configuration=config)
    assert fitted['direction'] == 'identity'
    assert all(v is None for settings in fitted['settings'] for v in settings.values())
    assert fitted['candidate_grids'][0] == {op: [None] for op in OPERATIONS}
    assert len(report['trials']) < 15


def test_fold_isolation_ignores_validation_and_global_summary_changes():
    spec = label_spec(DATASET)
    reference = np.zeros((10, 10), np.uint8)
    reference[2:5, 2:5] = 1
    prediction = reference.copy()
    prediction[8, 8] = 1
    cases = {'train': extract_case_components(reference, (1, 1), spec),
             'validation': extract_case_components(reference, (1, 1), spec)}
    fingerprint = {'component_statistics': {'metadata': component_metadata(spec), 'cases': cases,
                                            'summaries': fingerprint_summaries(cases, 1)}}
    load = lambda k: (prediction[None].astype(bool), reference, (1, 1))
    first, _ = fit_policy(['train'], load, fingerprint, spec)
    fingerprint['component_statistics']['cases']['validation'] = {'unreadable': True}
    fingerprint['component_statistics']['summaries'] = {'volume_percentiles': {'1.0': 1e9}}
    second, _ = fit_policy(['train'], load, fingerprint, spec)
    assert first == second


def test_policy_version_and_saved_connectivity_validation():
    spec = label_spec(DATASET)
    saved = policy(spec, min_volume=2)
    assert load_policy(saved, spec) == saved
    old = deepcopy(saved)
    old['version'] = 1
    with pytest.raises(ValueError, match='Refit'):
        load_policy(old, spec)
    saved['connectivity']['foreground'] = 4
    with pytest.raises(ValueError, match='connectivity'):
        load_policy(saved, spec)
    saved['connectivity']['foreground'] = 26.0
    with pytest.raises(ValueError, match='connectivity'):
        load_policy(saved, spec)


def test_parent_hierarchy_setting_and_deterministic_search():
    reference = np.zeros((12, 12), np.uint8)
    reference[3:6, 3:6] = 1
    prediction = reference.copy()
    prediction[10, 10] = 1
    first, report = fit({'train': reference}, {'train': prediction}, {'hierarchy_repair': 'parent'})
    second, _ = fit({'train': reference}, {'train': prediction}, {'hierarchy_repair': 'parent'})
    assert first == second
    assert [run['direction'] for run in report['runs']] == ['restrict']


def test_joint_search_finds_closing_size_interaction_that_greedy_misses():
    fragmented = np.zeros((80, 80), bool)
    fragmented[3:10, 3:10] = True
    fragmented[3:10, 6:8] = False
    reference = close_mask(fragmented, (1, 1), 1.5).astype(np.uint8)
    prediction = fragmented.astype(np.uint8)
    for i in range(10):
        row, column = 20 + 10 * (i // 5), 5 + 12 * (i % 5)
        prediction[row:row + 5, column] = prediction[row:row + 5, column + 3] = 1
    config = {'grouping_percentiles': None, 'filling_percentiles': None,
              'count_percentiles': None, 'hierarchy_repair': 'child'}
    greedy, _ = fit({'train': reference}, {'train': prediction}, config)
    joint, _ = fit({'train': reference}, {'train': prediction}, dict(config, hyperparameter_search='exhaustive'))
    assert greedy['direction'] == 'identity'
    assert joint['objective']['fitted'] == 1
    assert joint['settings'][0]['closing_radius'] == 1.5
    assert joint['settings'][0]['min_volume'] == int(reference.sum())


def test_parent_filter_recomputes_descendant_groups():
    spec = label_spec({'labels': {'background': 0, 'whole': [1, 2], 'core': 2},
                       'regions_class_order': [1, 2]})
    masks = np.zeros((2, 15, 15), bool)
    masks[0, 2:4, 2:4] = masks[0, 2:4, 8:10] = True
    masks[1, 2:4, 2:10] = True  # Parent restriction cuts this into two parts.
    saved = {'version': VERSION, 'spec': spec, 'direction': 'restrict',
             'connectivity': {'foreground': 26, 'background': 'full'},
             'settings': [{op: None for op in OPERATIONS} for _ in range(2)]}
    saved['settings'][0]['max_count'] = 1
    saved['settings'][1].update(grouping_distance=6, min_volume=4, max_count=1)
    segmentation = apply_policy(masks, (1, 1), spec, saved)
    assert np.all(segmentation[2:4, 2:4] == 2)
    assert not segmentation[2:4, 4:10].any()


def test_connectivity_8_volume_preflight_error_is_actionable():
    spec = label_spec(DATASET)
    segmentation = np.zeros((5, 5, 5), np.uint8)
    statistics = {'component_statistics': {'metadata': component_metadata(spec), 'cases': {
        'train': extract_case_components(segmentation, (1, 1, 1), spec)}}}
    with pytest.raises(ValueError, match='only valid for genuine 2D'):
        validate_fingerprint(statistics, spec, ['train'], 8)


def test_corrupt_component_tree_is_rejected():
    spec = label_spec(DATASET)
    segmentation = np.zeros((10, 10), np.uint8)
    segmentation[1, 1] = segmentation[3, 3] = segmentation[5, 5] = 1
    case = extract_case_components(segmentation, (1, 1), spec)
    case_regions(case, 26)[0]['edges'] = [[1, 2, 2], [1, 2, 2]]
    fingerprint = {'component_statistics': {'metadata': component_metadata(spec), 'cases': {'train': case}}}
    with pytest.raises(ValueError, match='refresh'):
        validate_fingerprint(fingerprint, spec, ['train'])
