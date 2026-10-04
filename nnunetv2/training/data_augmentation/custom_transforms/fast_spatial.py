"""Opt-in 3D label resampling; original image interpolation and sampled grid."""

import torch
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform


class FastSpatialTransform(SpatialTransform):
    def _apply_to_segmentation(self, segmentation, **params):
        grid = params["grid"]
        if (
            grid is None
            or segmentation.ndim != 4
            or segmentation.device.type != "cpu"
            or grid.device.type != "cpu"
            or grid.dtype != torch.float32
            or self.bg_style_seg_sampling
            or self.mode_seg != "bilinear"
            or self.border_mode_seg not in ("zeros", "constant")
            or segmentation.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
        ):
            return super()._apply_to_segmentation(segmentation, **params)
        from nnunetv2.training.data_augmentation.custom_transforms._fused_segmentation import resample_labels

        values = resample_labels(
            segmentation.contiguous().numpy(), grid.contiguous().numpy().reshape(-1, 3), self.align_corners
        )
        result = torch.from_numpy(values.reshape(segmentation.shape[0], *grid.shape[:-1]))
        if self._requires_constant_padding_fixup(self.border_mode_seg, self.padding_value_seg):
            mask = self._compute_out_of_bounds_mask(grid, segmentation.shape[1:])
            result.masked_fill_(mask.unsqueeze(0), self.padding_value_seg)
        return result


def replace_spatial_transforms(transform):
    if type(transform) is SpatialTransform:
        replacement = FastSpatialTransform.__new__(FastSpatialTransform)
        replacement.__dict__.update(transform.__dict__)
        return replacement
    if hasattr(transform, "transforms"):
        transform.transforms = [replace_spatial_transforms(t) for t in transform.transforms]
    if hasattr(transform, "transform"):
        transform.transform = replace_spatial_transforms(transform.transform)
    return transform
