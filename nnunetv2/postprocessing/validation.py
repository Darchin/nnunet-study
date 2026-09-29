"""Full-volume training prediction and distributed coordination for adaptive fitting."""
import json
from pathlib import Path
import shutil
import tempfile
import threading
import time

import numpy as np
import torch
import torch.distributed as dist

from nnunetv2.inference.export_prediction import export_fitting_masks
from nnunetv2.postprocessing.adaptive import (
    extract_case_components, fit_policy, label_spec, validate_fingerprint,
)
from nnunetv2.postprocessing.configuration import resolve_configuration
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy


def check_postprocessing_prerequisites(trainer):
    path = Path(trainer.preprocessed_dataset_folder_base) / 'dataset_fingerprint.json'
    if not path.is_file():
        raise ValueError('Adaptive fitting requires a fingerprint from PercentileFingerprintExtractor.')
    with path.open(encoding='utf-8') as f:
        fingerprint = json.load(f)
    spec = label_spec(trainer.dataset_json, trainer.label_manager)
    training_ids, _ = trainer.do_split()
    configuration = resolve_configuration(getattr(trainer.configuration_manager, 'post_processing', None))
    validate_fingerprint(fingerprint, spec, training_ids, configuration['connectivity'])
    return fingerprint, spec


def coordinated_fit(fit, distributed, rank, poll_seconds=5):
    """Broadcast heartbeats while rank zero fits, avoiding idle NCCL timeouts.

    Failure on the fitting rank is broadcast to all ranks instead of leaving peers
    blocked in a collective. The fitting thread uses only CPU masks/annotations.
    """
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


def fit_training_postprocessing(trainer, predictor, export_pool, enable_tta):
    fingerprint, spec = check_postprocessing_prerequisites(trainer)
    training_ids, validation_ids = trainer.do_split()
    rank = trainer.local_rank
    directory = [tempfile.mkdtemp(prefix='.postprocessing-fit-', dir=trainer.output_folder) if rank == 0 else None]
    if trainer.is_ddp:
        dist.broadcast_object_list(directory, src=0)
    directory = Path(directory[0])
    world_size = dist.get_world_size() if trainer.is_ddp else 1
    local_keys = training_ids[rank::world_size]
    dataset = trainer.dataset_class(trainer.preprocessed_dataset_folder, local_keys,
                                    folder_with_segs_from_previous_stage=trainer.folder_with_segs_from_previous_stage)
    workers = list(export_pool._pool)
    jobs = []
    last_barrier = len(training_ids) // world_size - 1
    for index, identifier in enumerate(dataset.identifiers):
        while check_workers_alive_and_busy(export_pool, workers, jobs, allowed_num_queued=2):
            time.sleep(.1)
        trainer.print_to_log_file(f'Predicting training case for post-processing: {identifier}')
        data, _, previous, properties = dataset.load_case(identifier)
        data = data[:]
        if trainer.is_cascaded:
            data = np.vstack((data, convert_labelmap_to_one_hot(previous[:], trainer.label_manager.foreground_labels,
                                                               output_dtype=data.dtype)))
        prediction = predictor.predict_sliding_window_return_logits(torch.from_numpy(np.array(data, copy=True)))
        jobs.append(export_pool.apply_async(export_fitting_masks,
                    (prediction.cpu(), properties, trainer.configuration_manager, trainer.plans_manager,
                     trainer.dataset_json, str(directory / f'{identifier}.npz'))))
        if trainer.is_ddp and index < last_barrier and (index + 1) % 20 == 0:
            dist.barrier()
    for job in jobs:
        job.get()
    if trainer.is_ddp:
        dist.barrier()

    def fit():
        configuration = resolve_configuration(getattr(trainer.configuration_manager, 'post_processing', None))
        rw = trainer.plans_manager.image_reader_writer_class()
        reference_folder = Path(trainer.preprocessed_dataset_folder_base) / 'gt_segmentations'
        ending = trainer.dataset_json['file_ending']

        def load_case(identifier):
            with np.load(directory / f'{identifier}.npz') as saved:
                masks, spacing = saved['masks'], saved['spacing']
            reference, properties = rw.read_seg(str(reference_folder / (identifier + ending)))
            reference = reference[0]
            if masks.shape[1:] != reference.shape or not np.allclose(spacing, properties['spacing']):
                raise ValueError(f'Prediction/reference grid mismatch for training case {identifier}.')
            return masks, reference, spacing

        # Verify original geometry/statistics once; never inspect validation annotations.
        for identifier in training_ids:
            _, reference, spacing = load_case(identifier)
            measured = extract_case_components(reference, spacing, spec)
            recorded = fingerprint['component_statistics']['cases'][identifier]
            if (measured['shape'] != recorded['shape'] or
                not np.allclose(measured['spacing'], recorded['spacing']) or
                measured['connectivities'] != recorded['connectivities']):
                raise ValueError(f'Stale component statistics for {identifier}; refresh fingerprint and '
                                 'ensure preprocessed gt_segmentations match the current annotations.')
        checkpoint = Path(trainer.output_folder) / 'checkpoint_last.pth'
        identity = {'name': checkpoint.name, 'epoch': trainer.current_epoch, 'present': checkpoint.is_file()}
        if checkpoint.is_file():
            identity.update(size=checkpoint.stat().st_size, mtime_ns=checkpoint.stat().st_mtime_ns)
        overlap = sorted(set(training_ids) & set(validation_ids))
        metadata = {'enable_tta': enable_tta, 'checkpoint': identity, 'fold': trainer.fold,
                    'configuration': trainer.configuration_name, 'overlapping_identifiers': overlap,
                    'evaluation': 'in-sample' if overlap else 'held-out',
                    'fitting_predictions': 'in-sample training predictions'}
        policy, report = fit_policy(training_ids, load_case, fingerprint, spec, metadata,
                                    progress=trainer.print_to_log_file, configuration=configuration)
        for name, value in [('postprocessing_search.json', report), ('postprocessing.json', policy)]:
            destination = Path(trainer.output_folder) / name
            temporary = destination.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
            temporary.replace(destination)
        trainer.print_to_log_file(f"Post-processing training Dice: {policy['objective']['raw']:.6f} -> "
                                  f"{policy['objective']['fitted']:.6f}; direction {policy['direction']}")
        if overlap:
            trainer.print_to_log_file('Post-processing evaluation is in-sample: training/validation identifiers overlap.')
        # This directory is exclusively owned by this invocation.
        if directory.resolve().parent != Path(trainer.output_folder).resolve():
            raise RuntimeError('Unexpected fitting temporary directory parent.')
        shutil.rmtree(directory)
        return policy

    return coordinated_fit(fit, trainer.is_ddp, rank)
