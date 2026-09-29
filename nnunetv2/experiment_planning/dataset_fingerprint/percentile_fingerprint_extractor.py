from typing import Dict, List
import multiprocessing
import os

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, save_json

from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import DatasetFingerprintExtractor
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnunetv2.postprocessing.adaptive import component_metadata, extract_case_components, label_spec, fingerprint_summaries


def analyze_components(segmentation_file, reader_writer_class, spec):
    segmentation, properties = reader_writer_class().read_seg(segmentation_file)
    result = extract_case_components(segmentation[0], properties['spacing'], spec)
    stat = os.stat(segmentation_file)
    result['source'] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    return result


class PercentileFingerprintExtractor(DatasetFingerprintExtractor):
    spacing_percentile_step = 5

    @staticmethod
    def compute_spacing_percentiles(spacings: List[List[float]], percentile_step: int = 5) -> Dict[str, List[float]]:
        spacings = np.vstack(spacings)
        return {
            str(p): [float(i) for i in np.percentile(spacings, p, axis=0)]
            for p in range(0, 101, percentile_step)
        }

    def run(self, overwrite_existing: bool = False) -> dict:
        fingerprint = super().run(overwrite_existing)
        changed = False
        if "spacing_percentiles" not in fingerprint:
            fingerprint["spacing_percentiles"] = self.compute_spacing_percentiles(
                fingerprint["spacings"], self.spacing_percentile_step
            )
            changed = True
        spec = label_spec(self.dataset_json)
        metadata = component_metadata(spec)
        statistics = fingerprint.get('component_statistics', {})
        if statistics.get('metadata') != metadata:
            statistics = {'metadata': metadata, 'cases': {}}
            changed = True
        pending = []
        for identifier, files in self.dataset.items():
            stat = os.stat(files['label'])
            source = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
            if overwrite_existing or statistics['cases'].get(identifier, {}).get('source') != source:
                pending.append(identifier)
        if pending:
            rw = determine_reader_writer_from_dataset_json(self.dataset_json,
                                                           self.dataset[pending[0]]['images'][0])
            with multiprocessing.get_context('spawn').Pool(self.num_processes) as pool:
                results = pool.starmap(analyze_components,
                                       [(self.dataset[k]['label'], rw, spec) for k in pending])
            statistics['cases'].update(zip(pending, results))
            changed = True
        removed = set(statistics['cases']) - self.dataset.keys()
        for identifier in removed:
            del statistics['cases'][identifier]
        changed |= bool(removed)
        if changed or 'summaries' not in statistics:
            statistics['summaries'] = fingerprint_summaries(statistics['cases'], len(spec['regions']))
            changed = True
        fingerprint['component_statistics'] = statistics
        if changed:
            save_json(
                fingerprint,
                join(nnUNet_preprocessed, self.dataset_name, "dataset_fingerprint.json"),
                sort_keys=False,
            )
        return fingerprint
