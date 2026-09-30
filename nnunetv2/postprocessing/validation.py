"""Persistent training predictions and fold-isolated distributed fitting."""
import hashlib
from contextlib import nullcontext
import json
from pathlib import Path
import threading
import time

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from nnunetv2.configuration import default_num_postprocessing_processes
from nnunetv2.inference.export_prediction import export_fitting_masks
from nnunetv2.postprocessing.adaptive import fit_policy, label_spec, validate_fingerprint
from nnunetv2.postprocessing.configuration import resolve_configuration
from nnunetv2.postprocessing.runtime import (
    CachedCaseLoader, CaseExecutor, annotation_digest, atomic_array, atomic_json, file_digest, trainer_progress,
)
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy


def check_postprocessing_prerequisites(trainer):
    path = Path(trainer.preprocessed_dataset_folder_base) / 'dataset_fingerprint.json'
    if not path.is_file():
        raise ValueError('Adaptive fitting requires a fingerprint from PercentileFingerprintExtractor.')
    fingerprint = json.loads(path.read_text(encoding='utf-8'))
    spec = label_spec(trainer.dataset_json, trainer.label_manager)
    training_ids, _ = trainer.do_split()
    resolve_configuration(getattr(trainer.configuration_manager, 'post_processing', None))
    validate_fingerprint(fingerprint, spec, training_ids)
    return fingerprint, spec


def coordinated_fit(fit, distributed, rank, poll_seconds=5):
    """Keep NCCL peers alive and broadcast fitting failures instead of deadlocking."""
    if not distributed:
        return fit()
    result = {}

    def run():
        try:
            result['policy'] = fit()
        except Exception as error:
            result['error'] = f'{type(error).__name__}: {error}'

    worker = threading.Thread(target=run) if rank == 0 else None
    if worker is not None:
        worker.start()
    while True:
        status = [dict(result) if rank == 0 else None]
        dist.broadcast_object_list(status, src=0)
        if 'error' in status[0]:
            if worker is not None:
                worker.join()
            raise RuntimeError(f"Adaptive post-processing fitting failed: {status[0]['error']}")
        if 'policy' in status[0]:
            if worker is not None:
                worker.join()
            return status[0]['policy']
        time.sleep(poll_seconds)


def file_identity(path):
    path = Path(path)
    stat = path.stat()
    return {'path': str(path.resolve()), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def coordinated_prediction(work, trainer, poll_seconds=5):
    """Uneven cached-case assignments still participate in periodic collectives."""
    stop = threading.Event()
    if not trainer.is_ddp:
        return work(stop)
    result = {}

    def run():
        try:
            with torch.cuda.device(trainer.device) if trainer.device.type == 'cuda' else nullcontext():
                work(stop)
            result['done'] = True
        except Exception as error:
            result.update(done=True, error=f'{type(error).__name__}: {error}')

    worker = threading.Thread(target=run)
    worker.start()
    errors = []
    while True:
        statuses = [None] * dist.get_world_size()
        dist.all_gather_object(statuses, dict(result))
        errors = [s['error'] for s in statuses if 'error' in s]
        if errors:
            stop.set()
        if all(s.get('done') for s in statuses):
            worker.join()
            if errors:
                raise RuntimeError('Training prediction/export failed: ' + '; '.join(errors))
            return
        time.sleep(poll_seconds)


def input_identity(folder, identifier):
    folder = Path(folder)
    # nnU-Net datasets use these exact files; avoid re-enumerating thousands of directory entries per case.
    result = []
    for suffix in ('.b2nd', '_seg.b2nd', '.npz', '.npy', '_seg.npy', '.pkl'):
        try:
            result.append(file_identity(folder / (identifier + suffix)))
        except FileNotFoundError:
            pass
    return sorted(result, key=lambda item: item['path'])


def model_identity(trainer):
    digest = hashlib.sha256()
    network = getattr(trainer.network, 'module', trainer.network)
    network = getattr(network, '_orig_mod', network)
    for name, value in sorted(network.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy())
    return digest.hexdigest()


def inspect_prediction(directory, prediction_path, identity):
    directory = Path(directory)
    try:
        saved = json.loads((directory / 'prediction.json').read_text(encoding='utf-8'))
        geometry = json.loads((directory / 'geometry.json').read_text(encoding='utf-8'))
        masks = np.load(directory / 'masks.npy', mmap_mode='r', allow_pickle=False)
        valid = (saved['identity'] == identity and masks.dtype == np.uint8 and masks.ndim == 1
                 and geometry['shape'][0] == len(identity['labels']['regions'])
                 and masks.size == (int(np.prod(geometry['shape'])) + 7) // 8
                 and saved['geometry_digest'] == file_digest(directory / 'geometry.json')
                 and saved['mask_digest'] == file_digest(directory / 'masks.npy')
                 and saved['prediction_digest'] == file_digest(prediction_path))
        return bool(valid)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def prepare_reference(identifier, directory, reference_path, reader_writer_class, recorded):
    directory = Path(directory)
    identity = file_identity(reference_path)
    cache_file, record_file = directory / 'reference.npy', directory / 'reference.json'
    reference, digest, cached = None, None, None
    try:
        saved = json.loads(record_file.read_text(encoding='utf-8'))
        if saved['source'] == identity and saved['annotation_digest'] == recorded['annotation_digest']:
            cached = np.load(cache_file, mmap_mode='r', allow_pickle=False)
            digest = annotation_digest(cached)
            if (digest == recorded['annotation_digest'] and list(cached.shape) == recorded['shape']
                and np.allclose(saved['spacing'], recorded['spacing'])):
                reference, spacing = cached, saved['spacing']
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if reference is None and isinstance(cached, np.memmap):
        cached._mmap.close()
        cached = None
    if reference is None:
        segmentation, properties = reader_writer_class().read_seg(reference_path)
        reference, spacing = segmentation[0], properties['spacing']
        digest = annotation_digest(reference)
    geometry = json.loads((directory / 'geometry.json').read_text(encoding='utf-8'))
    if (list(reference.shape) != geometry['shape'][1:] or list(reference.shape) != recorded['shape']
        or not np.allclose(spacing, geometry['spacing']) or not np.allclose(spacing, recorded['spacing'])):
        raise ValueError(f'Prediction/reference/fingerprint grid mismatch for training case {identifier}.')
    if digest != recorded['annotation_digest']:
        raise ValueError(f'Stale component statistics for {identifier}; refresh the fingerprint and ensure '
                         'gt_segmentations match the current annotations (preprocessing may need updating).')
    if not isinstance(reference, np.memmap):
        max_label = int(reference.max(initial=0))
        dtype = np.uint8 if max_label <= 255 else np.uint16 if max_label <= 65535 else np.int32
        atomic_array(cache_file, reference.astype(dtype))
        atomic_json(record_file, {'source': identity, 'annotation_digest': recorded['annotation_digest'],
                                 'spacing': list(map(float, spacing))})
    return identifier


def fit_training_postprocessing(trainer, predictor, export_pool, enable_tta):
    fingerprint, spec = check_postprocessing_prerequisites(trainer)
    training_ids, validation_ids = trainer.do_split()
    training_ids = sorted(training_ids)
    rank = trainer.local_rank
    directory = Path(trainer.output_folder) / 'training'
    cache_directory = directory / '.postprocessing_cache'
    directory.mkdir(parents=True, exist_ok=True)
    progress = trainer_progress(trainer)
    # Export workers can be reused only if their count matches the independently configured fitting count.
    shared_pool = export_pool if len(export_pool._pool) == default_num_postprocessing_processes else None
    identity = [None]
    if rank == 0:
        with progress.phase('Identifying model and prediction inputs'):
            identity[0] = {'model_digest': model_identity(trainer),
                          'enable_tta': enable_tta, 'mirroring_axes': trainer.inference_allowed_mirroring_axes,
                          'labels': spec, 'configuration_name': trainer.configuration_name,
                          'configuration': getattr(trainer.configuration_manager, 'configuration', {}),
                          'plans': getattr(trainer.plans_manager, 'plans', {})}
            # Fitting settings do not affect model predictions.
            identity[0]['configuration'] = dict(identity[0]['configuration'])
            identity[0]['configuration'].pop('post_processing', None)
            identity[0]['configuration'].pop('trainer', None)
            identity[0]['plans'] = dict(identity[0]['plans'])
            identity[0]['plans'].pop('configurations', None)
            identity[0] = json.loads(json.dumps(identity[0], default=str))
    if trainer.is_ddp:
        dist.broadcast_object_list(identity, src=0)
    identities = {k: dict(identity[0], inputs=input_identity(trainer.preprocessed_dataset_folder, k),
                         previous_stage=input_identity(trainer.folder_with_segs_from_previous_stage, k)
                         if trainer.folder_with_segs_from_previous_stage else []) for k in training_ids}
    pending = [None]

    def inspect():
        with CaseExecutor(shared_pool, default_num_postprocessing_processes, progress) as executor:
            present = executor.map(inspect_prediction,
                [(str(cache_directory / k), str(directory / (k + trainer.dataset_json['file_ending'])), identities[k])
                 for k in training_ids], 'Checking training prediction cache')
        return {'pending': [k for k, valid in zip(training_ids, present) if not valid]}

    pending[0] = coordinated_fit(inspect, trainer.is_ddp, rank)['pending']
    if rank == 0:
        progress(f'Training prediction cache: {len(training_ids) - len(pending[0])} reused; '
                 f'{len(pending[0])} require prediction')
    world_size = dist.get_world_size() if trainer.is_ddp else 1
    local_keys = pending[0][rank::world_size]
    dataset = trainer.dataset_class(trainer.preprocessed_dataset_folder, local_keys,
                                   folder_with_segs_from_previous_stage=trainer.folder_with_segs_from_previous_stage)
    workers = list(export_pool._pool)

    def predict(stop):
        jobs = []
        for identifier in dataset.identifiers:
            if stop.is_set():
                break
            while check_workers_alive_and_busy(export_pool, workers, jobs, allowed_num_queued=2):
                if stop.is_set():
                    return
                time.sleep(.1)
            progress(f'Predicting training case for post-processing: {identifier}')
            data, _, previous, properties = dataset.load_case(identifier)
            data = data[:]
            if trainer.is_cascaded:
                data = np.vstack((data, convert_labelmap_to_one_hot(previous[:],
                                  trainer.label_manager.foreground_labels, output_dtype=data.dtype)))
            prediction = predictor.predict_sliding_window_return_logits(torch.from_numpy(np.array(data, copy=True)))
            jobs.append(export_pool.apply_async(export_fitting_masks,
                (prediction.cpu(), properties, trainer.configuration_manager, trainer.plans_manager,
                 trainer.dataset_json, str(cache_directory / identifier),
                 str(directory / (identifier + trainer.dataset_json['file_ending'])), identities[identifier])))
        with progress.phase(f'Completing training exports, rank {rank}'):
            with tqdm(total=len(jobs), desc=f'Training exports (rank {rank})') as bar:
                remaining = list(jobs)
                while remaining:
                    if not all(worker.is_alive() for worker in workers):
                        raise RuntimeError('A training export worker died.')
                    ready = [job for job in remaining if job.ready()]
                    for job in ready:
                        job.get()
                        remaining.remove(job)
                        bar.update()
                    if not ready:
                        time.sleep(.1)

    original_network = getattr(predictor, 'network', None)
    if trainer.is_ddp and original_network is not None:
        # Independent inference must not broadcast DDP buffers from uneven cached-case assignments.
        predictor.network = getattr(original_network, 'module', original_network)
    try:
        coordinated_prediction(predict, trainer)
    finally:
        if original_network is not None:
            predictor.network = original_network

    def fit():
        configuration = resolve_configuration(getattr(trainer.configuration_manager, 'post_processing', None))
        ending = trainer.dataset_json['file_ending']
        reference_folder = Path(trainer.preprocessed_dataset_folder_base) / 'gt_segmentations'
        progress(f'Post-processing CPU workers: {default_num_postprocessing_processes} '
                 '(controlled by nnUNet_n_post_proc)')
        with CaseExecutor(shared_pool, default_num_postprocessing_processes, progress) as executor:
            executor.map(prepare_reference, [(k, str(cache_directory / k), str(reference_folder / (k + ending)),
                trainer.plans_manager.image_reader_writer_class, fingerprint['component_statistics']['cases'][k])
                for k in training_ids], 'Preparing and verifying training references')
            checkpoint = Path(trainer.output_folder) / 'checkpoint_last.pth'
            overlap = sorted(set(training_ids) & set(validation_ids))
            metadata = {'enable_tta': enable_tta, 'checkpoint': {'name': checkpoint.name,
                'model_digest': identity[0]['model_digest'], 'epoch': trainer.current_epoch}, 'fold': trainer.fold,
                'configuration': trainer.configuration_name, 'overlapping_identifiers': overlap,
                'evaluation': 'in-sample' if overlap else 'held-out',
                'fitting_predictions': 'in-sample training predictions'}
            policy, report = fit_policy(training_ids, CachedCaseLoader(str(cache_directory)), fingerprint, spec,
                metadata, progress=progress, configuration=configuration, executor=executor,
                report_path=directory / 'postprocessing_search.json')
        with progress.phase('Writing training metrics and fitted policy'):
            summary = report['case_summaries']['raw']
            for name in ('raw', 'selected'):
                for case in report['case_summaries'][name]['metric_per_case']:
                    case['prediction_file'] = str(directory / (case['case'] + ending))
                    case['reference_file'] = str(reference_folder / (case['case'] + ending))
            report['case_summaries']['selected']['policy_file'] = str(Path(trainer.output_folder) / 'postprocessing.json')
            summary['evaluation'] = 'in-sample training predictions'
            summary['postprocessing'] = {'objective': policy['objective'], 'selected': report['case_summaries']['selected'],
                                        'tested_configurations': report['trials']}
            atomic_json(directory / 'summary.json', summary)
            atomic_json(Path(trainer.output_folder) / 'postprocessing.json', policy)
            atomic_json(directory / 'postprocessing_search.json', report)
        progress(f"Post-processing training Dice: {policy['objective']['raw']:.6f} -> "
                 f"{policy['objective']['fitted']:.6f}; direction {policy['direction']}")
        if overlap:
            progress('Post-processing evaluation is in-sample: training/validation identifiers overlap.')
        return policy

    return coordinated_fit(fit, trainer.is_ddp, rank)
