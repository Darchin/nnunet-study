from typing import Union, List

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from batchgenerators.utilities.file_and_folder_operations import load_json, save_pickle

from nnunetv2.configuration import default_num_processes
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2, comp_blosc2_params
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.postprocessing.adaptive import (
    apply_policy, label_spec, load_policy, masks_from_probabilities,
)


def convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits: Union[torch.Tensor, np.ndarray],
                                                                plans_manager: PlansManager,
                                                                configuration_manager: ConfigurationManager,
                                                                label_manager: LabelManager,
                                                                properties_dict: dict,
                                                                return_probabilities: bool = False,
                                                                num_threads_torch: int = default_num_processes,
                                                                postprocessing_policy: dict = None,
                                                                return_masks: bool = False):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    # resample to original shape
    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    predicted_logits = configuration_manager.resampling_fn_probabilities(predicted_logits,
                                            properties_dict['shape_after_cropping_and_before_resampling'],
                                            current_spacing,
                                            [properties_dict['spacing'][i] for i in plans_manager.transpose_forward])
    # return value of resampling_fn_probabilities can be ndarray or Tensor but that does not matter because
    # apply_inference_nonlin will convert to torch
    need_masks = return_masks or postprocessing_policy is not None
    if not return_probabilities and not need_masks:
        # this has a faster computation path because we can skip the softmax in regular (not region based) training
        segmentation = label_manager.convert_logits_to_segmentation(predicted_logits)
    else:
        predicted_probabilities = label_manager.apply_inference_nonlin(predicted_logits)
        segmentation = label_manager.convert_probabilities_to_segmentation(predicted_probabilities)
    if need_masks:
        spec = label_spec({'labels': label_manager.label_dict,
                           'regions_class_order': label_manager.regions_class_order}, label_manager)
        cropped_masks = masks_from_probabilities(predicted_probabilities.cpu().numpy(), spec)
        masks = np.zeros((len(cropped_masks), *properties_dict['shape_before_cropping']), dtype=bool)
        for i, mask in enumerate(cropped_masks):
            masks[i] = insert_crop_into_image(masks[i], mask, properties_dict['bbox_used_for_cropping'])
        masks = masks.transpose([0] + [i + 1 for i in plans_manager.transpose_backward])
        if not return_probabilities:
            del predicted_probabilities
    del predicted_logits

    # put segmentation in bbox (revert cropping)
    segmentation_reverted_cropping = np.zeros(properties_dict['shape_before_cropping'],
                                              dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16)
    segmentation_reverted_cropping = insert_crop_into_image(segmentation_reverted_cropping, segmentation, properties_dict['bbox_used_for_cropping'])
    del segmentation

    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation_reverted_cropping, torch.Tensor):
        segmentation_reverted_cropping = segmentation_reverted_cropping.cpu().numpy()

    # revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(plans_manager.transpose_backward)
    if postprocessing_policy is not None:
        policy = load_policy(postprocessing_policy, spec)
        segmentation_reverted_cropping = apply_policy(masks, properties_dict['spacing'], spec, policy)
    if return_probabilities:
        # revert cropping
        predicted_probabilities = label_manager.revert_cropping_on_probabilities(predicted_probabilities,
                                                                                 properties_dict[
                                                                                     'bbox_used_for_cropping'],
                                                                                 properties_dict[
                                                                                     'shape_before_cropping'])
        predicted_probabilities = predicted_probabilities.cpu().numpy()
        # revert transpose
        predicted_probabilities = predicted_probabilities.transpose([0] + [i + 1 for i in
                                                                           plans_manager.transpose_backward])
        torch.set_num_threads(old_threads)
        if return_masks:
            return segmentation_reverted_cropping, masks, predicted_probabilities
        return segmentation_reverted_cropping, predicted_probabilities
    else:
        torch.set_num_threads(old_threads)
        if return_masks:
            return segmentation_reverted_cropping, masks, None
        return segmentation_reverted_cropping


def export_prediction_from_logits(predicted_array_or_file: Union[np.ndarray, torch.Tensor], properties_dict: dict,
                                  configuration_manager: ConfigurationManager,
                                  plans_manager: PlansManager,
                                  dataset_json_dict_or_file: Union[dict, str], output_file_truncated: str,
                                  save_probabilities: bool = False,
                                  num_threads_torch: int = default_num_processes,
                                  postprocessing_policy: dict = None,
                                  processed_output_file_truncated: str = None):
    # if isinstance(predicted_array_or_file, str):
    #     tmp = deepcopy(predicted_array_or_file)
    #     if predicted_array_or_file.endswith('.npy'):
    #         predicted_array_or_file = np.load(predicted_array_or_file)
    #     elif predicted_array_or_file.endswith('.npz'):
    #         predicted_array_or_file = np.load(predicted_array_or_file)['softmax']
    #     os.remove(tmp)

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    ret = convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_array_or_file, plans_manager, configuration_manager, label_manager, properties_dict,
        return_probabilities=save_probabilities, num_threads_torch=num_threads_torch,
        postprocessing_policy=postprocessing_policy if processed_output_file_truncated is None else None,
        return_masks=processed_output_file_truncated is not None
    )
    del predicted_array_or_file

    # save
    if processed_output_file_truncated is not None:
        segmentation_final, masks, probabilities_final = ret
        spec = label_spec(dataset_json_dict_or_file, label_manager)
        processed = apply_policy(masks, properties_dict['spacing'], spec, load_policy(postprocessing_policy, spec))
        plans_manager.image_reader_writer_class().write_seg(
            processed, processed_output_file_truncated + dataset_json_dict_or_file['file_ending'], properties_dict)
        ret = (segmentation_final, probabilities_final) if save_probabilities else segmentation_final
        del masks, processed
    if save_probabilities:
        segmentation_final, probabilities_final = ret
        np.savez_compressed(output_file_truncated + '.npz', probabilities=probabilities_final)
        save_pickle(properties_dict, output_file_truncated + '.pkl')
        del probabilities_final, ret
    else:
        segmentation_final = ret
        del ret

    rw = plans_manager.image_reader_writer_class()
    rw.write_seg(segmentation_final, output_file_truncated + dataset_json_dict_or_file['file_ending'],
                 properties_dict)


def export_fitting_masks(logits, properties, configuration_manager, plans_manager, dataset_json, output_path,
                         prediction_path=None, identity=None):
    """Worker export retaining the independent region decisions used for fitting."""
    segmentation, masks, _ = convert_predicted_logits_to_segmentation_with_correct_shape(
        logits, plans_manager, configuration_manager, plans_manager.get_label_manager(dataset_json),
        properties, return_masks=True)
    if prediction_path is None:
        np.savez_compressed(output_path, masks=masks, spacing=np.asarray(properties['spacing']))
        return
    from pathlib import Path
    from nnunetv2.postprocessing.runtime import store_masks, atomic_json, file_digest
    directory = Path(output_path)
    directory.mkdir(parents=True, exist_ok=True)
    completion = directory / 'prediction.json'
    completion.unlink(missing_ok=True)
    store_masks(directory, masks, properties['spacing'])
    path = Path(prediction_path)
    ending = dataset_json['file_ending']
    temporary = path.with_name(path.name[:-len(ending)] + '.tmp' + ending)
    plans_manager.image_reader_writer_class().write_seg(segmentation, str(temporary), properties)
    temporary.replace(path)
    atomic_json(completion, {'identity': identity, 'mask_digest': file_digest(directory / 'masks.npy'),
                            'geometry_digest': file_digest(directory / 'geometry.json'),
                            'prediction_digest': file_digest(path)})


def resample_and_save(predicted: Union[torch.Tensor, np.ndarray], target_shape: List[int], output_file: str,
                      plans_manager: PlansManager, configuration_manager: ConfigurationManager, properties_dict: dict,
                      dataset_json_dict_or_file: Union[dict, str], num_threads_torch: int = default_num_processes,
                      dataset_class=None) \
        -> None:

    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    # resample to original shape
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    target_spacing = configuration_manager.spacing if len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    predicted_array_or_file = configuration_manager.resampling_fn_probabilities(predicted,
                                                                                target_shape,
                                                                                current_spacing,
                                                                                target_spacing)

    # create segmentation (argmax, regions, etc)
    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    segmentation = label_manager.convert_logits_to_segmentation(predicted_array_or_file)
    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()

    if dataset_class is None or dataset_class == nnUNetDatasetBlosc2:
        block_size, chunk_size = comp_blosc2_params(
            (1, *segmentation.shape),
            tuple(configuration_manager.patch_size),
            bytes_per_pixel=1 if len(label_manager.foreground_labels) < 255 else 2
        )
        block_size = [int(i) for i in block_size[1:]]
        chunk_size = [int(i) for i in chunk_size[1:]]
        nnUNetDatasetBlosc2.save_seg(
            segmentation.astype(dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16),
            output_file,
            chunks_seg=chunk_size,
            blocks_seg=block_size)
    else:
        dataset_class.save_seg(segmentation.astype(dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16), output_file)
    torch.set_num_threads(old_threads)
