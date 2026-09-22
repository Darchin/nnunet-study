import warnings
from fractions import Fraction
from typing import Tuple, Union

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from threadpoolctl import threadpool_limits

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.utilities.split_resolution import SplitResolutionGeometry


class SplitResolutionDataLoader(nnUNetDataLoader):
    """3D loader for image and target arrays sampled on different voxel lattices."""

    def __init__(self, *args, segmentation_patch_size, geometry: SplitResolutionGeometry, **kwargs):
        super().__init__(*args, **kwargs)
        if self.patch_size_was_2d:
            raise ValueError("split-resolution loading supports only 3D configurations")
        self.geometry = geometry
        self.segmentation_patch_size = tuple(int(i) for i in segmentation_patch_size)
        expected = geometry.input_extent_to_target(self.patch_size)
        if self.segmentation_patch_size != expected:
            raise ValueError(
                f"segmentation_patch_size={self.segmentation_patch_size} does not match "
                f"the exact spacing-derived size {expected}"
            )

    def get_bbox(self, data_shape: np.ndarray, force_fg: bool, class_locations: Union[dict, None],
                 overwrite_class: Union[int, Tuple[int, ...]] = None, verbose: bool = False):
        need_to_pad = self.need_to_pad.copy()
        for axis in range(len(data_shape)):
            if need_to_pad[axis] + data_shape[axis] < self.patch_size[axis]:
                need_to_pad[axis] = self.patch_size[axis] - data_shape[axis]

        raw_lbs = [-need_to_pad[i] // 2 for i in range(len(data_shape))]
        raw_ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i]
                   for i in range(len(data_shape))]
        bounds = self.geometry.align_input_origin_bounds(raw_lbs, raw_ubs)

        selected_class = None
        if force_fg or self.has_ignore:
            if class_locations is None:
                raise ValueError("class_locations are required for foreground oversampling")
            if not force_fg and self.has_ignore:
                selected_class = self.annotated_classes_key
                if len(class_locations[selected_class]) == 0:
                    warnings.warn("Warning! No annotated pixels in image!")
                    selected_class = None
            else:
                eligible = [key for key, locations in class_locations.items() if len(locations) > 0]
                annotated = [key == self.annotated_classes_key if isinstance(key, tuple) else False
                             for key in eligible]
                if any(annotated) and len(eligible) > 1:
                    eligible.pop(int(np.where(annotated)[0][0]))
                if overwrite_class is not None and overwrite_class in eligible:
                    selected_class = overwrite_class
                elif eligible:
                    selected_class = eligible[np.random.choice(len(eligible))]
                elif verbose:
                    print("case does not contain any foreground classes")

        if selected_class is None:
            bbox_lbs = [
                np.random.randint(lb // denominator, ub // denominator + 1) * denominator
                for (lb, ub), denominator in zip(bounds, self.geometry.denominators)
            ]
        else:
            locations = class_locations[selected_class]
            target_center = locations[np.random.choice(len(locations))][1:]
            input_center = self.geometry.target_center_to_input(target_center)
            bbox_lbs = []
            for center, patch, (lb, ub), denominator in zip(
                input_center, self.patch_size, bounds, self.geometry.denominators
            ):
                ideal = center - Fraction(int(patch), 2)
                aligned = (int(ideal.numerator // ideal.denominator) // denominator) * denominator
                bbox_lbs.append(min(ub, max(lb, aligned)))

        return bbox_lbs, [lb + patch for lb, patch in zip(bbox_lbs, self.patch_size)]

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = None
        seg_all = None
        with torch.no_grad(), threadpool_limits(limits=1, user_api=None):
            for batch_index, key in enumerate(selected_keys):
                data, seg, seg_prev, properties = self._data.load_case(key)
                if seg_prev is not None:
                    raise RuntimeError("split-resolution training does not support cascades")
                bbox_lbs, bbox_ubs = self.get_bbox(
                    data.shape[1:], self.get_do_oversample(batch_index), properties["class_locations"]
                )
                data_bbox = [[lb, ub] for lb, ub in zip(bbox_lbs, bbox_ubs)]
                target_lbs = self.geometry.input_boundary_to_target(bbox_lbs)
                target_ubs = self.geometry.input_boundary_to_target(bbox_ubs)
                target_bbox = [[lb, ub] for lb, ub in zip(target_lbs, target_ubs)]
                data_sample = torch.from_numpy(crop_and_pad_nd(data, data_bbox, 0)).float()
                seg_sample = torch.from_numpy(
                    crop_and_pad_nd(seg, target_bbox, -1, cast_cropped_to=np.int16)
                ).to(torch.int16)
                if self.transforms is not None:
                    transformed = self.transforms(image=data_sample, segmentation=seg_sample)
                    data_sample, seg_sample = transformed["image"], transformed["segmentation"]
                if data_all is None:
                    data_all = torch.empty((self.batch_size, *data_sample.shape), dtype=torch.float32)
                data_all[batch_index] = data_sample
                if isinstance(seg_sample, list):
                    if seg_all is None:
                        seg_all = [torch.empty((self.batch_size, *item.shape), dtype=item.dtype)
                                   for item in seg_sample]
                    for target_index, target in enumerate(seg_sample):
                        seg_all[target_index][batch_index] = target
                else:
                    if seg_all is None:
                        seg_all = torch.empty((self.batch_size, *seg_sample.shape), dtype=seg_sample.dtype)
                    seg_all[batch_index] = seg_sample
        return {"data": data_all, "target": seg_all, "keys": selected_keys}
