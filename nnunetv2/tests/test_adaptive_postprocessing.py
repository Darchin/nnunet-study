import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from nnunetv2.postprocessing.adaptive import (
    VERSION, OPERATIONS, MorphologyCache, apply_policy, candidates, cavity_components, close_mask,
    component_metadata, component_volumes, extract_case_components, fit_policy, fragmentation_gaps,
    label_spec, load_policy, masks_from_segmentation, mean_dice, repair_hierarchy, validate_fingerprint, validate_spec,
    case_regions,
)
from nnunetv2.inference.export_prediction import (
    convert_predicted_logits_to_segmentation_with_correct_shape, export_prediction_from_logits,
)
from nnunetv2.utilities.label_handling.label_handling import LabelManager


LABELS = {'labels': {'background': 0, 'lesion': 1}, 'file_ending': '.npy'}
NESTED = {'labels': {'background': 0, 'whole': [1, 2], 'core': 2}, 'regions_class_order': [1, 2]}


def policy_for(spec, direction='expand', **settings):
    return {'version': VERSION, 'spec': spec, 'direction': direction,
            'connectivity': {'foreground': 26, 'background': 'full'},
            'settings': [{op: settings.get(op) for op in OPERATIONS} for _ in spec['regions']]}


def fingerprint_for(references, spec, spacing=(1., 1.)):
    return {'component_statistics': {'metadata': component_metadata(spec),
            'cases': {k: extract_case_components(r, spacing, spec) for k, r in references.items()}}}


def test_face_connectivity_and_physical_volume():
    mask = np.eye(3, dtype=bool)
    _, volumes = component_volumes(mask, (2, 3), 6)
    np.testing.assert_equal(volumes, [6, 6, 6])
    volume = np.zeros((3, 3, 3), bool)
    volume[0, 0, 0] = volume[1, 1, 1] = True
    np.testing.assert_equal(component_volumes(volume, (2, 3, 4), 6)[1], [24, 24])


def test_cavities_and_complementary_connectivity():
    mask = np.ones((7, 7), bool)
    mask[3, 3] = False
    _, ids, volumes = cavity_components(mask, (2, 3))
    assert len(ids) == 1 and volumes[0] == 6
    for i in range(4):
        mask[i, i] = False
    assert not len(cavity_components(mask, (2, 3))[1])


def test_closing_preserves_boundary_and_respects_anisotropy():
    mask = np.zeros((12, 12), bool)
    mask[:5, :5] = True
    closed = close_mask(mask, (1, 1), 2)
    assert np.all(closed[mask])
    mask = np.zeros((3, 12, 12), bool)
    mask[0, 3:8, 3:8] = mask[2, 3:8, 3:8] = True
    assert not close_mask(mask, (5, 1, 1), 1)[1].any()


@pytest.mark.parametrize('spacing,radius', [((1., 1.), 1.5), ((2., 1.), 2.5), ((2., 1., 1.), 2.)])
def test_distance_closing_matches_explicit_spherical_footprint(spacing, radius):
    from scipy import ndimage as ndi
    mask = np.random.default_rng(2).random((9,) * len(spacing)) > .7
    extent = np.ceil(radius / np.asarray(spacing)).astype(int)
    coordinates = np.ogrid[tuple(slice(-int(e), int(e) + 1) for e in extent)]
    sphere = sum((c * s) ** 2 for c, s in zip(coordinates, spacing)) <= radius ** 2
    padding = [(int(e) + 1, int(e) + 1) for e in extent]
    explicit = ndi.binary_closing(np.pad(mask, padding), structure=sphere)
    crop = tuple(slice(p, -p) for p, _ in padding)
    np.testing.assert_equal(close_mask(mask, spacing, radius), mask | explicit[crop])


def test_natural_2d_dummy_axis_does_not_create_false_exterior_or_volume():
    mask = np.ones((1, 7, 7), bool)
    mask[0, 3, 3] = False
    assert cavity_components(mask, (999, 1, 1))[2].tolist() == [1]
    assert component_volumes(mask, (999, 1, 1))[1].tolist() == [48]
    np.testing.assert_equal(close_mask(mask, (999, 1, 1), 1)[0], close_mask(mask[0], (1, 1), 1))


def test_fragmentation_gaps_exclude_real_neighbors_and_ambiguous_matches():
    reference = np.zeros((10, 15), bool)
    reference[3:7, 2:12] = True
    predicted = reference.copy()
    predicted[:, 6:8] = False
    assert fragmentation_gaps(predicted, reference, (1, 2)) == [6.]
    # Ground truth itself has two objects: this is not fragmentation.
    assert fragmentation_gaps(predicted, predicted, (1, 2)) == []
    reference[:, 7] = False
    predicted = reference.copy()
    predicted[4, 7] = True  # predicted object overlaps both true objects
    assert fragmentation_gaps(predicted, reference, (1, 1)) == []


def test_hierarchy_directions_and_filter_propagation():
    spec = label_spec(NESTED)
    masks = np.zeros((2, 8, 8), bool)
    masks[0, 1:5, 1:5] = True
    masks[1, 3:7, 3:7] = True
    expanded = repair_hierarchy(masks.copy(), spec, 'expand')
    assert np.all(expanded[0][masks[1]])
    restricted = repair_hierarchy(masks.copy(), spec, 'restrict')
    np.testing.assert_equal(restricted[1], masks[0] & masks[1])
    policy = policy_for(spec)
    policy['settings'][0]['min_volume'] = 100
    assert not apply_policy(masks, (1, 1), spec, policy).any()
    policy = policy_for(spec)
    policy['settings'][1]['min_volume'] = 100
    result = apply_policy(masks, (1, 1), spec, policy)
    assert not (result == 2).any()
    assert np.all(result[expanded[0]] == 1)


def test_child_morphology_clips_or_expands_parent():
    spec = label_spec(NESTED)
    masks = np.ones((2, 9, 9), bool)
    masks[:, 4, 4] = False
    policy = policy_for(spec, 'restrict')
    policy['settings'][1]['hole_volume'] = 1
    assert apply_policy(masks, (1, 1), spec, policy)[4, 4] == 0
    policy['direction'] = 'expand'
    assert apply_policy(masks, (1, 1), spec, policy)[4, 4] == 2


def test_disjoint_hole_additions_do_not_overwrite_other_labels():
    spec = label_spec({'labels': {'background': 0, 'a': 1, 'b': 2}})
    masks = np.zeros((2, 9, 9), bool)
    masks[0, 1:8, 1:8] = True
    masks[0, 4, 4] = False
    masks[1, 4, 4] = True
    policy = policy_for(spec, hole_volume=10)
    assert apply_policy(masks, (1, 1), spec, policy)[4, 4] == 2


def test_spec_rejects_crossing_and_invalid_encoding():
    crossing = {'labels': {'background': 0, 'a': [1, 2], 'b': [2, 3]}, 'regions_class_order': [1, 3]}
    with pytest.raises(ValueError, match='cross'):
        validate_spec(label_spec(crossing))
    wrong = deepcopy(NESTED)
    wrong['regions_class_order'] = [2, 1]
    with pytest.raises(ValueError, match='encode'):
        validate_spec(label_spec(wrong))


def test_quantiles_are_deduplicated_and_counts_are_discrete():
    assert candidates([], [1, 5, 10]) == [None]
    assert candidates([2, 2, 2], [1, 5, 10]) == [None, 2]
    assert candidates([1, 3], [90, 95, 100], discrete=True) == [None, 3]


def test_cache_is_bounded_and_returns_identical_results():
    cache = MorphologyCache(max_bytes=32)
    mask = np.ones((4, 4), bool)
    compute = Mock(return_value=mask.copy())
    first = cache.value('test', mask, (1, 1), None, compute)
    second = cache.value('test', mask, (1, 1), None, compute)
    assert first is second and compute.call_count == 1
    for i in range(5):
        cache.value('test', mask, (1, 1), i, compute)
    assert cache.nbytes <= 32


def test_fit_size_and_count_selects_improvement_and_ignores_validation():
    spec = label_spec(LABELS)
    reference = np.zeros((12, 12), np.uint8)
    reference[3:7, 3:7] = 1
    empty = np.zeros_like(reference)
    predictions = {'positive': reference.copy(), 'negative': empty.copy()}
    predictions['positive'][10, 10] = predictions['negative'][10, 10] = 1
    references = {'positive': reference, 'negative': empty, 'validation': np.ones_like(reference)}
    fingerprint = fingerprint_for(references, spec)
    requested = []

    def load_case(k):
        requested.append(k)
        assert k != 'validation'
        return masks_from_segmentation(predictions[k], spec), references[k], (1, 1)

    policy, report = fit_policy(['positive', 'negative'], load_case, fingerprint, spec)
    assert policy['objective']['fitted'] == 1
    assert policy['settings'][0]['min_volume'] == 16
    assert policy['settings'][0]['closing_radius'] is None
    assert report['fixed_candidate_grids'][0]['max_count'] == [None, 1]
    references['validation'][:] = 0
    case_regions(fingerprint['component_statistics']['cases']['validation'], 26)[0]['volumes'] = [1000000]
    second, _ = fit_policy(['positive', 'negative'], load_case, fingerprint, spec)
    assert second == policy and set(requested) == {'positive', 'negative'}


def test_fit_identity_on_perfect_predictions_and_empty_class():
    spec = label_spec(LABELS)
    empty = np.zeros((5, 5), np.uint8)
    fingerprint = fingerprint_for({'empty': empty}, spec)
    policy, report = fit_policy(['empty'], lambda _: (empty[None].astype(bool), empty, (1, 1)), fingerprint, spec)
    assert policy['direction'] == 'identity' and policy['objective']['fitted'] == 1
    assert all(report['fixed_candidate_grids'][0][op] == [None] for op in ('min_volume', 'max_count', 'closing_radius'))


def test_fit_filling_repairs_small_cavity():
    spec = label_spec(LABELS)
    reference = np.zeros((9, 9), np.uint8)
    reference[1:8, 1:8] = 1
    predicted = reference.copy()
    predicted[4, 4] = 0
    fingerprint = fingerprint_for({'train': reference}, spec)
    policy, report = fit_policy(['train'], lambda _: (predicted[None].astype(bool), reference, (1, 1)), fingerprint, spec)
    assert policy['settings'][0]['hole_volume'] == 1
    assert policy['objective']['fitted'] == 1


def test_fit_preserves_genuine_hole():
    spec = label_spec(LABELS)
    reference = np.zeros((9, 9), np.uint8)
    reference[1:8, 1:8] = 1
    reference[4, 4] = 0
    fingerprint = fingerprint_for({'train': reference}, spec)
    policy, _ = fit_policy(['train'], lambda _: (reference[None].astype(bool), reference, (1, 1)), fingerprint, spec)
    assert policy['direction'] == 'identity'


def test_fit_closing_reconnects_fragments_before_volume_filtering():
    spec = label_spec(LABELS)
    reference = np.zeros((12, 16), np.uint8)
    reference[3:9, 2:13] = 1
    predicted = reference.copy()
    predicted[:, 7:9] = 0
    fingerprint = fingerprint_for({'train': reference}, spec)
    fitted, report = fit_policy(['train'], lambda _: (predicted[None].astype(bool), reference, (1, 1)),
                                fingerprint, spec)
    assert fitted['settings'][0]['closing_radius'] == 1.5
    assert fitted['objective']['fitted'] > fitted['objective']['raw']
    assert report['fixed_candidate_grids'][0]['closing_radius'] == [None, 1.5]


def test_fragmentation_uses_tree_edges_instead_of_all_pairs():
    reference = np.zeros((7, 20), bool)
    reference[2:5, 1:18] = True
    predicted = reference.copy()
    predicted[:, 5:7] = predicted[:, 11:13] = False
    assert fragmentation_gaps(predicted, reference, (1, 1)) == [3., 3.]


def test_first_disjoint_region_wins_competing_morphology_additions():
    from nnunetv2.postprocessing.adaptive import repair_masks
    spec = label_spec({'labels': {'background': 0, 'a': 1, 'b': 2}})
    masks = np.zeros((2, 12, 12), bool)
    masks[0, 2:10, 3] = masks[0, 2:10, 5] = True
    masks[1, 3, 2:10] = masks[1, 5, 2:10] = True
    # Remove initial overlaps; the two closures still both propose the center (4,4).
    masks[1] &= ~masks[0]
    policy = policy_for(spec, closing_radius=1.5)
    repaired = repair_masks(masks, (1, 1), spec, policy['settings'], 'expand')
    assert repaired[0, 4, 4] and not repaired[1, 4, 4]


def test_three_level_hierarchy_propagates_through_all_ancestors():
    spec = label_spec({'labels': {'background': 0, 'whole': [1, 2, 3], 'core': [2, 3], 'tip': 3},
                       'regions_class_order': [1, 2, 3]})
    masks = np.zeros((3, 5, 5), bool)
    masks[2, 2, 2] = True
    assert np.all(repair_hierarchy(masks.copy(), spec, 'expand')[:, 2, 2])
    assert not repair_hierarchy(masks.copy(), spec, 'restrict').any()


def test_policy_round_trip_validation_and_tta_warning(tmp_path):
    spec = label_spec(LABELS)
    policy = policy_for(spec, min_volume=5)
    policy['metadata'] = {'enable_tta': True}
    path = tmp_path / 'postprocessing.json'
    path.write_text(json.dumps(policy))
    assert load_policy(path, spec, True) == policy
    with pytest.warns(UserWarning, match='TTA'):
        load_policy(path, spec, False)
    policy['version'] += 1
    with pytest.raises(ValueError, match='version'):
        load_policy(policy, spec)
    policy = policy_for(spec, max_count=-1)
    with pytest.raises(ValueError, match='max_count'):
        load_policy(policy, spec)


def test_missing_fingerprint_and_ignore_are_actionable():
    spec = label_spec(LABELS)
    with pytest.raises(ValueError, match='Refresh'):
        validate_fingerprint({}, spec)
    ignored = label_spec({'labels': {'background': 0, 'lesion': 1, 'ignore': 2}})
    with pytest.raises(ValueError, match='ignore'):
        validate_fingerprint({}, ignored)


class NumpyReaderWriter:
    def read_seg(self, filename):
        return np.load(filename)[None], {'spacing': [1., 1.]}

    def write_seg(self, segmentation, filename, properties):
        np.save(filename, segmentation)


def export_fixture(dataset_json=LABELS):
    label_manager = LabelManager(dataset_json['labels'], dataset_json.get('regions_class_order'))
    plans = SimpleNamespace(transpose_forward=[1, 0], transpose_backward=[1, 0],
                            get_label_manager=lambda _: label_manager, image_reader_writer_class=NumpyReaderWriter)
    config = SimpleNamespace(spacing=[1., 1.], resampling_fn_probabilities=lambda logits, *args: logits)
    properties = {'spacing': [1., 1.], 'shape_after_cropping_and_before_resampling': [5, 6],
                  'shape_before_cropping': [9, 10], 'bbox_used_for_cropping': [[2, 7], [1, 7]]}
    return plans, config, label_manager, properties


def test_export_retains_preconversion_masks_and_raw_probabilities(tmp_path):
    dataset = dict(NESTED, file_ending='.npy')
    plans, config, lm, properties = export_fixture(dataset)
    logits = torch.full((2, 5, 6), -10.)
    logits[1, 2, 2] = 10  # child outside its predicted parent
    raw, masks, probabilities = convert_predicted_logits_to_segmentation_with_correct_shape(
        logits, plans, config, lm, properties, return_probabilities=True, return_masks=True)
    assert raw[3, 4] == 2 and not masks[0].any() and masks[1].sum() == 1
    policy = policy_for(label_spec(dataset), direction='restrict')
    export_prediction_from_logits(logits, properties, config, plans, dataset, str(tmp_path / 'raw'), True,
                                  postprocessing_policy=policy, processed_output_file_truncated=str(tmp_path / 'processed'))
    np.testing.assert_equal(np.load(tmp_path / 'raw.npy'), raw)
    assert not np.load(tmp_path / 'processed.npy').any()
    with np.load(tmp_path / 'raw.npz') as saved:
        np.testing.assert_equal(saved['probabilities'], probabilities)
    processed, returned = convert_predicted_logits_to_segmentation_with_correct_shape(
        logits, plans, config, lm, properties, return_probabilities=True, postprocessing_policy=policy)
    assert not processed.any()
    np.testing.assert_equal(returned, probabilities)


def test_disabled_export_matches_original_and_mean_dice_convention():
    plans, config, lm, properties = export_fixture()
    logits = torch.randn((2, 5, 6))
    raw = convert_predicted_logits_to_segmentation_with_correct_shape(logits, plans, config, lm, properties)
    retained, _, _ = convert_predicted_logits_to_segmentation_with_correct_shape(
        logits, plans, config, lm, properties, return_masks=True)
    np.testing.assert_equal(raw, retained)
    empty = np.zeros((9, 10), np.uint8)
    assert mean_dice(empty, empty[None].astype(bool), label_spec(LABELS)) == 1


def test_batch_cli_and_worker_command_propagation():
    from nnunetv2.run.batch_train import parse_args, make_worker_command
    args = parse_args(['--worker', '--plans-file', 'p.json', '--configuration', 'c', '--fold', '0',
                       '--trainer', 'nnUNetTrainer', '--postprocess'])
    assert args.postprocess
    command = make_worker_command('p.json', 'c', 0, 'nnUNetTrainer', True, 50, True, postprocess=True)
    assert '--postprocess' in command and '--disable-tta' in command and '--disable_train_val' in command
    assert '--postprocess' not in make_worker_command('p.json', 'c', 0, 'nnUNetTrainer', False, 50, False)


def test_coordinated_fit_success_and_failure(monkeypatch):
    from nnunetv2.postprocessing import validation
    calls = []
    monkeypatch.setattr(validation.dist, 'broadcast_object_list', lambda status, **kwargs: calls.append(status))
    assert validation.coordinated_fit(lambda: {'direction': 'identity'}, True, 0, .001) == {'direction': 'identity'}
    assert calls

    def fail():
        raise ValueError('bad annotations')

    with pytest.raises(RuntimeError, match='bad annotations'):
        validation.coordinated_fit(fail, True, 0, .001)
    # Nonzero ranks receive the fitted object without invoking the fitter.
    monkeypatch.setattr(validation.dist, 'broadcast_object_list',
                        lambda status, **kwargs: status.__setitem__(0, {'policy': {'direction': 'expand'}}))
    assert validation.coordinated_fit(fail, True, 1, .001) == {'direction': 'expand'}


class ImmediatePool:
    """Exercise worker arguments with deterministic synchronous fixture execution."""
    def __init__(self, *args):
        self._pool = [SimpleNamespace(is_alive=lambda: True)]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def starmap(self, function, args):
        return [function(*a) for a in args]

    def apply_async(self, function, args, kwds=None):
        value = function(*args, **(kwds or {}))
        return SimpleNamespace(get=lambda: value, ready=lambda: True)

    def starmap_async(self, function, args):
        value = self.starmap(function, args)
        return SimpleNamespace(get=lambda: value, ready=lambda: True)


def test_fingerprint_refresh_adds_statistics_without_reextracting_images(tmp_path, monkeypatch):
    from nnunetv2.experiment_planning.dataset_fingerprint import percentile_fingerprint_extractor as module
    label = tmp_path / 'case.npy'
    segmentation = np.zeros((7, 7), np.uint8)
    segmentation[2:5, 2:5] = 1
    np.save(label, segmentation)
    existing = {'spacings': [[1, 1]], 'foreground_intensity_properties_per_channel': {'0': {'mean': 13}}}
    monkeypatch.setattr(module.DatasetFingerprintExtractor, 'run', lambda *args: existing)
    monkeypatch.setattr(module, 'determine_reader_writer_from_dataset_json', lambda *args: NumpyReaderWriter)
    monkeypatch.setattr(module.multiprocessing, 'get_context', lambda *args: SimpleNamespace(Pool=ImmediatePool))
    monkeypatch.setattr(module, 'nnUNet_preprocessed', str(tmp_path))
    (tmp_path / 'Dataset001_Test').mkdir()
    extractor = module.PercentileFingerprintExtractor.__new__(module.PercentileFingerprintExtractor)
    extractor.dataset_json, extractor.dataset_name, extractor.num_processes = LABELS, 'Dataset001_Test', 1
    extractor.dataset = {'case': {'images': ['unused.npy'], 'label': str(label)}}
    refreshed = extractor.run()
    assert refreshed['foreground_intensity_properties_per_channel']['0']['mean'] == 13
    assert case_regions(refreshed['component_statistics']['cases']['case'], 26)[0]['volumes'] == [9]
    np.save(label, np.zeros((8, 8), np.uint8))  # source size changes, triggering a refresh
    assert case_regions(extractor.run()['component_statistics']['cases']['case'], 26)[0]['volumes'] == []


def test_full_volume_validation_fits_then_exports_both_results(tmp_path, monkeypatch):
    import importlib
    module = importlib.import_module('nnunetv2.training.nnUNetTrainer.nnUNetTrainer')
    shape = (12, 12)
    reference = np.zeros(shape, np.uint8)
    reference[3:7, 3:7] = 1
    predicted = reference.copy()
    predicted[10, 10] = 1
    logits = torch.stack((torch.where(torch.from_numpy(predicted) > 0, -10., 10.),
                          torch.where(torch.from_numpy(predicted) > 0, 10., -10.)))
    calls = []

    class Predictor:
        def __init__(self, **kwargs):
            assert kwargs['use_mirroring'] is False

        def manual_initialization(self, *args):
            pass

        def predict_sliding_window_return_logits(self, data):
            calls.append(int(data[0, 0, 0]))
            return logits.clone()

    properties = {'spacing': [1., 1.], 'shape_before_cropping': list(shape),
                  'shape_after_cropping_and_before_resampling': list(shape),
                  'bbox_used_for_cropping': [[0, 12], [0, 12]]}

    class Dataset:
        def __init__(self, folder, identifiers, **kwargs):
            self.identifiers = identifiers

        def load_case(self, identifier):
            data = np.full((1, *shape), 1 if identifier == 'train' else 2, np.float32)
            return data, None, None, properties

    output = tmp_path / 'fold_0'
    output.mkdir()
    (tmp_path / 'gt_segmentations').mkdir()
    for identifier in ('train', 'val'):
        np.save(tmp_path / 'gt_segmentations' / f'{identifier}.npy', reference)
    spec = label_spec(LABELS)
    (tmp_path / 'dataset_fingerprint.json').write_text(json.dumps(fingerprint_for({'train': reference}, spec)))
    trainer = module.nnUNetTrainer.__new__(module.nnUNetTrainer)
    trainer.set_deep_supervision_enabled = Mock()
    trainer.enable_deep_supervision = False
    trainer.network = torch.nn.Identity()
    trainer.is_ddp, trainer.is_cascaded, trainer.local_rank = False, False, 0
    trainer.output_folder = str(output)
    trainer.preprocessed_dataset_folder_base = str(tmp_path)
    trainer.preprocessed_dataset_folder = 'unused'
    trainer.folder_with_segs_from_previous_stage = None
    trainer.dataset_class = Dataset
    trainer.do_split = lambda: (['train'], ['val'])
    trainer.device = torch.device('cpu')
    trainer.dataset_json = LABELS
    trainer.configuration_name, trainer.fold, trainer.current_epoch = 'test', 0, 1
    trainer.label_manager = LabelManager(LABELS['labels'], None)
    trainer.configuration_manager = SimpleNamespace(spacing=[1, 1], next_stage_names=None,
        post_processing={'hyperparameter_search': 'exhaustive', 'connectivity': 6, 'aggregation': 'instance',
                         'hierarchy_repair': 'parent', 'grouping_percentiles': None},
        resampling_fn_probabilities=lambda data, *args: data)
    trainer.plans_manager = SimpleNamespace(transpose_forward=[0, 1], transpose_backward=[0, 1],
        get_label_manager=lambda _: trainer.label_manager, image_reader_writer_class=NumpyReaderWriter)
    trainer.inference_allowed_mirroring_axes = None
    trainer.print_to_log_file = Mock()
    trainer.logger = SimpleNamespace(log_summary=Mock())
    monkeypatch.setattr(module, 'nnUNetPredictor', Predictor)
    monkeypatch.setattr(module.multiprocessing, 'get_context', lambda *args: SimpleNamespace(Pool=ImmediatePool))
    trainer.perform_actual_validation(save_probabilities=True, enable_tta=False, postprocess=True)
    assert calls == [1, 2]  # exactly one training inference, then validation inference
    saved_policy = load_policy(output / 'postprocessing.json', spec, False)
    assert saved_policy['training_identifiers'] == ['train']
    assert saved_policy['configuration']['hyperparameter_search'] == 'exhaustive'
    assert saved_policy['configuration']['aggregation'] == 'instance'
    assert saved_policy['connectivity']['foreground'] == 6
    assert not list(output.glob('.postprocessing-fit-*'))
    np.testing.assert_equal(np.load(output / 'validation' / 'val.npy'), predicted)
    np.testing.assert_equal(np.load(output / 'validation_postprocessed' / 'val.npy'), reference)
    raw_summary = json.loads((output / 'validation' / 'summary.json').read_text())
    processed_summary = json.loads((output / 'validation_postprocessed' / 'summary.json').read_text())
    assert raw_summary['foreground_mean']['Dice'] < processed_summary['foreground_mean']['Dice'] == 1
    assert (output / 'validation' / 'val.npz').is_file()
    assert (output / 'postprocessing_search.json').is_file()
    assert any(a.args[0] == 'final_val_postprocessed/foreground_dice' for a in trainer.logger.log_summary.call_args_list)


def test_predictor_saved_policy_is_reused_for_returned_predictions(monkeypatch, tmp_path):
    import importlib
    module = importlib.import_module('nnunetv2.inference.predict_from_raw_data')
    plans, config, lm, properties = export_fixture()
    spec = label_spec(LABELS)
    policy = policy_for(spec, min_volume=2)
    path = tmp_path / 'postprocessing.json'
    path.write_text(json.dumps(policy))
    predictor = module.nnUNetPredictor(device=torch.device('cpu'), postprocessing_policy=str(path), use_mirroring=False)
    monkeypatch.setenv('nnUNet_compile', 'false')
    predictor.manual_initialization(torch.nn.Identity(), plans, config, None, LABELS, 'test', None)
    logits = torch.full((2, 5, 6), -10.)
    logits[0] = 10
    logits[0, 2, 2] = -10
    logits[1, 2, 2] = 10
    predictor.predict_logits_from_preprocessed_data = lambda _: logits
    monkeypatch.setattr(module, 'PreprocessAdapterFromNpy', lambda *args, **kwargs: iter([
        {'data': torch.zeros((1, 5, 6)), 'data_properties': properties}]))
    # The one-image API should route through the same export hook as bulk inference.
    segmentation = predictor.predict_single_npy_array(np.zeros((1, 9, 10)), properties)
    assert not segmentation.any()
    predictor.set_postprocessing_policy(None)
    assert predictor.predict_single_npy_array(np.zeros((1, 9, 10)), properties).sum() == 1


def test_ensemble_applies_policy_after_averaging_and_preserves_probabilities(tmp_path):
    from nnunetv2.ensembling.ensemble import merge_files
    from batchgenerators.utilities.file_and_folder_operations import save_pickle
    lm = LabelManager(NESTED['labels'], NESTED['regions_class_order'])
    # Each input has a different parent region, but aggregation predicts only the child.
    a = np.zeros((2, 7, 7), np.float32)
    b = a.copy()
    a[0, 3, 3] = .9
    a[1, 3, 3] = b[1, 3, 3] = .9
    for name, value in [('a', a), ('b', b)]:
        np.savez_compressed(tmp_path / f'{name}.npz', probabilities=value)
        save_pickle({'spacing': [1., 1.]}, str(tmp_path / f'{name}.pkl'))
    merge_files([str(tmp_path / 'a.npz'), str(tmp_path / 'b.npz')], str(tmp_path / 'ensemble'), '.npy',
                NumpyReaderWriter(), lm, True, policy_for(label_spec(NESTED), 'restrict'))
    assert not np.load(tmp_path / 'ensemble.npy').any()
    with np.load(tmp_path / 'ensemble.npz') as saved:
        np.testing.assert_allclose(saved['probabilities'], (a + b) / 2)


def test_count_filter_keeps_largest_and_breaks_equal_size_ties_deterministically():
    spec = label_spec(LABELS)
    masks = np.zeros((1, 12, 12), bool)
    masks[0, 1:3, 1:3] = masks[0, 8:10, 8:10] = True
    result = apply_policy(masks, (1, 1), spec, policy_for(spec, max_count=1))
    assert result[1:3, 1:3].all() and not result[8:10, 8:10].any()
    masks[0, 7:10, 7:10] = True
    result = apply_policy(masks, (1, 1), spec, policy_for(spec, max_count=1))
    assert result[7:10, 7:10].all() and not result[1:3, 1:3].any()


@pytest.mark.parametrize('enabled', [False, True])
def test_training_python_interface_preflights_before_training_and_preserves_disabled_signature(monkeypatch, enabled):
    from nnunetv2.run import run_training as module
    from nnunetv2.postprocessing import validation
    events = []
    trainer = SimpleNamespace(run_training=lambda: events.append('train'),
                              perform_actual_validation=lambda *args, **kwargs: events.append(('validate', args, kwargs)))
    monkeypatch.setattr(module, 'get_trainer_from_args', lambda *args, **kwargs: trainer)
    monkeypatch.setattr(module, 'maybe_load_checkpoint', lambda *args: events.append('checkpoint'))
    monkeypatch.setattr(module.torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(validation, 'check_postprocessing_prerequisites', lambda t: events.append('preflight'))
    module.run_training('Dataset001_Test', 'test', 0, plans_identifier='test', device=torch.device('cpu'),
                        disable_tta=True, postprocess=enabled)
    if enabled:
        assert events == ['preflight', 'checkpoint', 'train', ('validate', (False, False), {'postprocess': True})]
    else:
        assert events == ['checkpoint', 'train', ('validate', (False, False), {})]


def test_training_cli_propagates_postprocess_and_tta_alias(monkeypatch):
    import sys
    from nnunetv2.run import run_training as module
    run = Mock()
    monkeypatch.setattr(module, 'run_training', run)
    monkeypatch.setattr(module.torch, 'set_num_threads', lambda *args: None)
    monkeypatch.setattr(sys, 'argv', ['train', '123', 'test', '0', '--postprocess', '--disable-tta', '-device', 'cpu'])
    module.run_training_entry()
    assert run.call_args.kwargs['postprocess'] and run.call_args.kwargs['disable_tta']


class WorkerPlans:
    transpose_forward = [1, 0]
    transpose_backward = [1, 0]
    image_reader_writer_class = NumpyReaderWriter

    def get_label_manager(self, dataset):
        return LabelManager(dataset['labels'], dataset.get('regions_class_order'))


def identity_resample(array, *args):
    return array


def test_real_spawn_export_worker_round_trip(tmp_path):
    import multiprocessing
    from nnunetv2.inference.export_prediction import export_fitting_masks
    _, _, _, properties = export_fixture()
    config = SimpleNamespace(spacing=[1., 1.], resampling_fn_probabilities=identity_resample)
    logits = np.full((2, 5, 6), -10., np.float32)
    logits[0] = 10
    logits[0, 2, 2], logits[1, 2, 2] = -10, 10
    policy = policy_for(label_spec(LABELS), min_volume=2)
    with multiprocessing.get_context('spawn').Pool(1) as pool:
        pool.apply_async(export_fitting_masks,
                         (logits, properties, config, WorkerPlans(), LABELS, str(tmp_path / 'fit.npz'))).get(timeout=60)
        pool.apply_async(export_prediction_from_logits,
                         (logits, properties, config, WorkerPlans(), LABELS, str(tmp_path / 'raw'), True),
                         {'postprocessing_policy': policy,
                          'processed_output_file_truncated': str(tmp_path / 'processed')}).get(timeout=60)
    with np.load(tmp_path / 'fit.npz') as saved:
        assert saved['masks'].sum() == 1 and saved['masks'].shape == (1, 10, 9)
    assert np.load(tmp_path / 'raw.npy').sum() == 1
    assert not np.load(tmp_path / 'processed.npy').any()
