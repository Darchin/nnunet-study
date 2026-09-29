"""Isolated GPU smoke test on two real preprocessed crops and an existing checkpoint.

Run with --model-folder MODEL --fold 0. Original datasets, fingerprints and result
folders are read-only inputs. Temporary smoke outputs are automatically removed.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.postprocessing.adaptive import component_metadata, extract_case_components, label_spec
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-folder', required=True)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--patch-size', type=int, default=64)
    args = parser.parse_args()
    os.environ['nnUNet_compile'] = 'false'
    torch.set_num_threads(2)
    predictor = nnUNetPredictor(device=torch.device('cuda'), use_mirroring=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(args.model_folder, [args.fold])
    plans = predictor.plans_manager
    configuration = predictor.configuration_manager
    configuration.configuration['patch_size'] = [args.patch_size] * len(configuration.spacing)
    dataset_folder = Path(os.fspath(nnUNet_preprocessed)) / plans.dataset_name
    preprocessed_folder = dataset_folder / configuration.data_identifier
    split = json.loads((dataset_folder / 'splits_final.json').read_text())[args.fold]
    source = infer_dataset_class(str(preprocessed_folder))(str(preprocessed_folder),
                                                          [split['train'][0], split['val'][0]])
    spec = label_spec(predictor.dataset_json, predictor.label_manager)
    cases, references = {}, {}
    for name, identifier in [('training_smoke', split['train'][0]), ('validation_smoke', split['val'][0])]:
        data, segmentation, previous, properties = source.load_case(identifier)
        if previous is not None or configuration.previous_stage_name is not None:
            raise ValueError('This isolated crop smoke test currently requires a non-cascaded model.')
        if len(configuration.spacing) != 3 or not np.allclose(configuration.spacing, properties['spacing']):
            raise ValueError('Use a 3D configuration with original-grid spacing for this crop smoke test.')
        data, segmentation = data[:], segmentation[0]
        segmentation = np.asarray(segmentation)
        points = np.argwhere(segmentation > 0)
        center = points.mean(0).astype(int) if len(points) else np.asarray(segmentation.shape) // 2
        start = np.maximum(0, np.minimum(center - args.patch_size // 2,
                                        np.asarray(segmentation.shape) - args.patch_size))
        slices = tuple(slice(int(s), int(s) + args.patch_size) for s in start)
        cropped = np.array(data[(slice(None), *slices)], copy=True)
        properties = dict(properties, shape_before_cropping=list(cropped.shape[1:]),
                          shape_after_cropping_and_before_resampling=list(cropped.shape[1:]),
                          bbox_used_for_cropping=[[0, int(s)] for s in cropped.shape[1:]])
        cases[name] = (cropped, None, None, properties)
        references[name] = np.maximum(segmentation[slices], 0).transpose(plans.transpose_backward)

    class CroppedDataset:
        def __init__(self, folder, identifiers, **kwargs):
            self.identifiers = identifiers

        def load_case(self, identifier):
            return cases[identifier]

    with tempfile.TemporaryDirectory(prefix='adaptive-smoke-') as temporary:
        root = Path(temporary)
        output = root / 'fold_0'
        output.mkdir()
        (root / 'gt_segmentations').mkdir()
        rw = plans.image_reader_writer_class()
        ending = predictor.dataset_json['file_ending']
        for name, reference in references.items():
            rw.write_seg(reference, str(root / 'gt_segmentations' / (name + ending)), cases[name][3])
        fingerprint = {'component_statistics': {'metadata': component_metadata(spec), 'cases': {
            'training_smoke': extract_case_components(references['training_smoke'],
                                                      cases['training_smoke'][3]['spacing'], spec)}}}
        (root / 'dataset_fingerprint.json').write_text(json.dumps(fingerprint))
        metrics = {}
        trainer = SimpleNamespace(network=predictor.network, enable_deep_supervision=False,
            set_deep_supervision_enabled=lambda value: None, is_ddp=False, is_cascaded=False, local_rank=0,
            output_folder=str(output), preprocessed_dataset_folder_base=str(root),
            preprocessed_dataset_folder=str(root), folder_with_segs_from_previous_stage=None,
            dataset_class=CroppedDataset, do_split=lambda: (['training_smoke'], ['validation_smoke']),
            device=torch.device('cuda'), dataset_json=predictor.dataset_json,
            configuration_name='isolated_crop_smoke', fold=args.fold, current_epoch=0,
            label_manager=predictor.label_manager, configuration_manager=configuration, plans_manager=plans,
            inference_allowed_mirroring_axes=None, print_to_log_file=lambda *a, **kw: print(*a, flush=True),
            logger=SimpleNamespace(log_summary=lambda name, value: metrics.__setitem__(name, value)))
        # Keep the existing GPU/result setup untouched and bound smoke-test CPU workers.
        import importlib
        module = importlib.import_module('nnunetv2.training.nnUNetTrainer.nnUNetTrainer')
        module.default_num_processes = 1
        nnUNetTrainer.perform_actual_validation(trainer, save_probabilities=True, enable_tta=False, postprocess=True)
        policy = json.loads((output / 'postprocessing.json').read_text())
        assert policy['training_identifiers'] == ['training_smoke']
        assert (output / 'validation_postprocessed' / 'summary.json').is_file()
        assert (output / 'validation' / 'validation_smoke.npz').is_file()
        print('SMOKE PASSED:', json.dumps({'training_objective': policy['objective'], 'validation': metrics}), flush=True)


if __name__ == '__main__':
    main()
