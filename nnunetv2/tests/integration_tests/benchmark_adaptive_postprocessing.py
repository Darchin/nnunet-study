"""Read-only native-grid benchmark using reproducibly perturbed training annotations.

This measures preparation and fitting, not model inference or validation accuracy.
Original annotations, fingerprints, and model outputs are never modified.
"""
import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnunetv2.postprocessing.adaptive import (
    case_regions, component_metadata, extract_case_components, fit_policy, label_spec, masks_from_segmentation,
)
from nnunetv2.postprocessing.runtime import CachedCaseLoader, Progress, atomic_array, atomic_json, store_masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--dataset', default='226')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--cases', type=int, default=4)
    parser.add_argument('--processes', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.cases < 1:
        raise ValueError('--cases must be positive.')
    dataset = maybe_convert_to_dataset_name(int(args.dataset) if args.dataset.isdigit() else args.dataset)
    root = Path(nnUNet_preprocessed) / dataset
    dataset_json = json.loads((root / 'dataset.json').read_text())
    split = json.loads((root / 'splits_final.json').read_text())[args.fold]
    previous = json.loads((root / 'dataset_fingerprint.json').read_text())['component_statistics']['cases']
    keys = sorted(split['train'])
    most_fragmented = max(keys, key=lambda k: sum(len(r['volumes']) for r in case_regions(previous[k], 26)))
    keys = [most_fragmented] + [k for k in keys if k != most_fragmented][:args.cases - 1]
    spec = label_spec(dataset_json)
    ending = dataset_json['file_ending']
    reader = determine_reader_writer_from_dataset_json(dataset_json, str(root / 'gt_segmentations' / (keys[0] + ending)))()
    result = {'dataset': dataset, 'fold': args.fold, 'cases': keys,
              'prediction_source': 'deterministically perturbed training annotations', 'runs': []}
    with tempfile.TemporaryDirectory(prefix='postprocessing-benchmark-') as temporary:
        cache = Path(temporary)
        records, extraction = {}, []
        for identifier in keys:
            start = time.monotonic()
            reference, properties = reader.read_seg(str(root / 'gt_segmentations' / (identifier + ending)))
            reference = reference[0]
            records[identifier] = extract_case_components(reference, properties['spacing'], spec)
            extraction.append(time.monotonic() - start)
            masks = masks_from_segmentation(reference, spec)
            # Sparse false positives plus one removed plane simulate inexpensive reproducible prediction errors.
            rng = np.random.default_rng(20260929)
            for region, mask in enumerate(masks):
                for point in rng.integers(np.asarray(mask.shape), size=(12, mask.ndim)):
                    mask[tuple(point)] = True
                occupied = np.flatnonzero(mask.sum(axis=tuple(range(1, mask.ndim))))
                if len(occupied):
                    mask[occupied[len(occupied) // 2]] = False
            store_masks(cache / identifier, masks, properties['spacing'])
            atomic_array(cache / identifier / 'reference.npy', reference)
        fingerprint = {'component_statistics': {'metadata': component_metadata(spec), 'cases': records}}
        result['preparation_seconds'] = extraction
        result['cache_bytes'] = sum(path.stat().st_size for path in cache.rglob('*') if path.is_file())
        first = None
        for processes in args.processes:
            start = time.monotonic()
            policy, report = fit_policy(keys, CachedCaseLoader(str(cache)), fingerprint, spec,
                progress=Progress(), num_processes=processes)
            if first is None:
                first = policy
            elif policy != first:
                raise AssertionError('Worker count changed the fitted policy.')
            result['runs'].append({'processes': processes, 'seconds': time.monotonic() - start,
                'execution': report['execution'], 'distinct_trials': len(report['trials']),
                'objective': policy['objective']})
            atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
