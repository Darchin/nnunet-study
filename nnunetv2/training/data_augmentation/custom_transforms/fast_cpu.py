"""CPU Gaussian blur and region conversion with the original transform parameters."""

import numpy as np
import torch
from scipy.ndimage import correlate1d
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform, _build_kernel
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform


class FastGaussianBlurTransform(GaussianBlurTransform):
    def _apply_to_image(self, img, **params):
        if img.device.type != "cpu" or img.dtype != torch.float32 or img.requires_grad:
            return super()._apply_to_image(img, **params)
        indices = np.flatnonzero(params["apply_to_channel"].numpy())
        if not len(indices):
            return img
        if self.synchronize_channels:
            sub = img[params["apply_to_channel"]].numpy()
            for dim in range(img.ndim - 1):
                kernel = _build_kernel(params["sigmas"][dim], truncate=6).numpy()
                sub = correlate1d(sub, kernel, axis=dim + 1, mode="mirror")
            img[params["apply_to_channel"]] = torch.from_numpy(sub)
        else:
            for j, channel in enumerate(indices):
                sub = img[channel : channel + 1].numpy()
                for dim in range(img.ndim - 1):
                    kernel = _build_kernel(params["sigmas"][j][dim], truncate=6).numpy()
                    sub = correlate1d(sub, kernel, axis=dim + 1, mode="mirror")
                img[channel : channel + 1] = torch.from_numpy(sub)
        return img


class FastRegionsTransform(ConvertSegmentationToRegionsTransform):
    def _apply_to_segmentation(self, segmentation, **params):
        if segmentation.device.type != "cpu" or any(len(r) == 0 for r in self.regions):
            return super()._apply_to_segmentation(segmentation, **params)
        seg = segmentation[self.channel_in_seg].numpy()
        output = torch.empty((len(self.regions), *seg.shape), dtype=torch.bool)
        for i, labels in enumerate(self.regions):
            target = output[i].numpy()
            np.equal(seg, labels[0].item(), out=target)
            for label in labels[1:]:
                np.logical_or(target, seg == label.item(), out=target)
        return output


def replace_cpu_transforms(transform):
    replacements = {
        GaussianBlurTransform: FastGaussianBlurTransform,
        ConvertSegmentationToRegionsTransform: FastRegionsTransform,
    }
    replacement = replacements.get(type(transform))
    if replacement:
        new = replacement.__new__(replacement)
        new.__dict__.update(transform.__dict__)
        if isinstance(new, FastGaussianBlurTransform):
            new.benchmark = False
        return new
    if hasattr(transform, "transforms"):
        transform.transforms = [replace_cpu_transforms(t) for t in transform.transforms]
    if hasattr(transform, "transform"):
        transform.transform = replace_cpu_transforms(transform.transform)
    return transform
