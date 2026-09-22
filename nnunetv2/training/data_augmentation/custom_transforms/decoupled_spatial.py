from typing import Tuple, Union, List, Callable
import numpy as np
import torch
import torch.nn.functional as F
from batchgeneratorsv2.transforms.spatial.spatial import (
    SpatialTransform,
    _convert_my_grid_to_grid_sample_grid,
    _create_centered_identity_grid2,
)


class PairedSpatialTransform(SpatialTransform):
    def __init__(self,
                 patch_size: Tuple[int, ...],
                 patch_center_dist_from_border: Union[int, List[int], Tuple[int, ...]],
                 random_crop: bool,
                 p_elastic_deform: float = 0,
                 elastic_deform_scale: Union[int, float, Tuple[float, float], Callable[..., Union[int, float]]] = (0, 0.2),
                 elastic_deform_magnitude: Union[int, float, Tuple[float, float], Callable[..., Union[int, float]]] = (0, 0.2),
                 p_synchronize_def_scale_across_axes: float = 0,
                 p_rotation: float = 0,
                 rotation: Union[int, float, Tuple[float, float], Callable[..., Union[int, float]]] = (0, 2 * np.pi),
                 p_rot_per_axis: float = 1,
                 p_scaling: float = 0,
                 scaling: Union[int, float, Tuple[float, float], Callable[..., Union[int, float]]] = (0.7, 1.3),
                 p_synchronize_scaling_across_axes: float = 0,
                 bg_style_seg_sampling: bool = True,
                 mode_seg: str = 'bilinear',
                 border_mode_seg: str = 'zeros',
                 center_deformation: bool = True,
                 mode_image: str = 'bilinear',
                 padding_mode_image: str = 'zeros',
                 padding_value_seg: float = 0,
                 padding_value_image: float = 0,
                 align_corners: bool = False,
                 patch_size_seg: Tuple[int, ...] = None):
        super().__init__(
            patch_size=patch_size,
            patch_center_dist_from_border=patch_center_dist_from_border,
            random_crop=random_crop,
            p_elastic_deform=p_elastic_deform,
            elastic_deform_scale=elastic_deform_scale,
            elastic_deform_magnitude=elastic_deform_magnitude,
            p_synchronize_def_scale_across_axes=p_synchronize_def_scale_across_axes,
            p_rotation=p_rotation,
            rotation=rotation,
            p_rot_per_axis=p_rot_per_axis,
            p_scaling=p_scaling,
            scaling=scaling,
            p_synchronize_scaling_across_axes=p_synchronize_scaling_across_axes,
            bg_style_seg_sampling=bg_style_seg_sampling,
            mode_seg=mode_seg,
            border_mode_seg=border_mode_seg,
            center_deformation=center_deformation,
            mode_image=mode_image,
            padding_mode_image=padding_mode_image,
            padding_value_seg=padding_value_seg,
            padding_value_image=padding_value_image,
            align_corners=align_corners,
        )
        self.target_patch_size = tuple(int(i) for i in patch_size_seg) if patch_size_seg is not None else self.patch_size
        self._target_scale = torch.tensor(
            [target / image for image, target in zip(self.patch_size, self.target_patch_size)],
            dtype=torch.float32,
        )
        self._target_transform = SpatialTransform(
            patch_size=self.target_patch_size,
            patch_center_dist_from_border=patch_center_dist_from_border,
            random_crop=random_crop,
            p_elastic_deform=0,
            p_rotation=0,
            p_scaling=0,
            bg_style_seg_sampling=bg_style_seg_sampling,
            mode_seg=mode_seg,
            border_mode_seg=border_mode_seg,
            center_deformation=center_deformation,
            mode_image=mode_image,
            padding_mode_image=padding_mode_image,
            padding_value_seg=padding_value_seg,
            padding_value_image=padding_value_image,
            align_corners=align_corners,
        )

    def get_parameters(self, **data_dict) -> dict:
        params = super().get_parameters(**data_dict)
        params["image_shape"] = tuple(data_dict["image"].shape[1:])
        return params

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        if self.target_patch_size == self.patch_size:
            return super()._apply_to_segmentation(segmentation, **params)

        segmentation = segmentation.contiguous()
        grid = params.get('grid')
        if grid is not None:
            scale = self._target_scale
            grid_seg = _create_centered_identity_grid2(self.target_patch_size).float()
            grid_seg /= scale
            offsets = params["elastic_offsets"]
            if offsets is not None:
                dim = offsets.shape[-1]
                channel_first = offsets.movedim(-1, 0)[None]
                resized_offsets = F.interpolate(
                    channel_first,
                    size=self.target_patch_size,
                    mode="trilinear" if dim == 3 else "bilinear",
                    align_corners=True,
                )[0].movedim(0, -1)
                grid_seg += resized_offsets
            affine = params["affine"]
            if affine is not None:
                grid_seg = torch.matmul(grid_seg, torch.from_numpy(affine.T).float())
            grid_seg *= scale
            if self.center_deformation and offsets is not None:
                mean = grid_seg.mean(dim=tuple(range(len(self.target_patch_size))))
            else:
                mean = 0
            center = torch.tensor(params["center_location_in_pixels"], dtype=torch.float32) * scale
            source_shape = torch.tensor(segmentation.shape[1:], dtype=torch.float32)
            grid_seg += center - source_shape / 2 - mean
            grid_seg = _convert_my_grid_to_grid_sample_grid(grid_seg, segmentation.shape[1:])
            params_seg = {**params, 'grid': grid_seg}
        else:
            ratio = [s / d for s, d in zip(segmentation.shape[1:], params["image_shape"])]
            center_seg = [i * r for i, r in zip(params['center_location_in_pixels'], ratio)]
            params_seg = {**params, 'center_location_in_pixels': center_seg}

        return self._target_transform._apply_to_segmentation(segmentation, **params_seg)


from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform


class PairedMaskImageTransform(MaskImageTransform):
    def apply(self, data_dict, **params):
        if len(self.apply_to_channels) == 0:
            return data_dict
        mask = data_dict['segmentation'][self.channel_idx_in_seg] < 0
        if not mask.any():
            return data_dict
        img = data_dict['image']
        if mask.shape != img.shape[1:]:
            mask_float = mask.float().contiguous()[None, None]
            mask = F.interpolate(
                mask_float,
                size=img.shape[1:],
                mode='nearest-exact',
            )[0, 0] > 0.5
        for a in self.apply_to_channels:
            img[a, mask] = self.set_outside_to
        return data_dict
