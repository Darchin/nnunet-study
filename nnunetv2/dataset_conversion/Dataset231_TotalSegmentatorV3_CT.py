import multiprocessing
import os
import shutil
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import (
    join, isfile, isdir, maybe_mkdir_p, subdirs, subfiles
)

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw

# Canonical 117 classes for TotalSegmentator v3 (CT)
TOTAL_V3_CT_LABELS: Dict[str, int] = {
    "background": 0,
    "spleen": 1,
    "kidney_right": 2,
    "kidney_left": 3,
    "gallbladder": 4,
    "liver": 5,
    "stomach": 6,
    "pancreas": 7,
    "adrenal_gland_right": 8,
    "adrenal_gland_left": 9,
    "lung_upper_lobe_left": 10,
    "lung_lower_lobe_left": 11,
    "lung_upper_lobe_right": 12,
    "lung_middle_lobe_right": 13,
    "lung_lower_lobe_right": 14,
    "esophagus": 15,
    "trachea": 16,
    "thyroid_gland": 17,
    "small_bowel": 18,
    "duodenum": 19,
    "colon": 20,
    "urinary_bladder": 21,
    "prostate": 22,
    "kidney_cyst_left": 23,
    "kidney_cyst_right": 24,
    "sacrum": 25,
    "vertebrae_L6": 26,
    "vertebrae_L5": 27,
    "vertebrae_L4": 28,
    "vertebrae_L3": 29,
    "vertebrae_L2": 30,
    "vertebrae_L1": 31,
    "vertebrae_T12": 32,
    "vertebrae_T11": 33,
    "vertebrae_T10": 34,
    "vertebrae_T9": 35,
    "vertebrae_T8": 36,
    "vertebrae_T7": 37,
    "vertebrae_T6": 38,
    "vertebrae_T5": 39,
    "vertebrae_T4": 40,
    "vertebrae_T3": 41,
    "vertebrae_T2": 42,
    "vertebrae_T1": 43,
    "vertebrae_C7": 44,
    "vertebrae_C6": 45,
    "vertebrae_C5": 46,
    "vertebrae_C4": 47,
    "vertebrae_C3": 48,
    "vertebrae_C2": 49,
    "vertebrae_C1": 50,
    "heart": 51,
    "aorta": 52,
    "pulmonary_vein": 53,
    "brachiocephalic_trunk": 54,
    "subclavian_artery_right": 55,
    "subclavian_artery_left": 56,
    "common_carotid_artery_right": 57,
    "common_carotid_artery_left": 58,
    "brachiocephalic_vein_left": 59,
    "brachiocephalic_vein_right": 60,
    "atrial_appendage_left": 61,
    "superior_vena_cava": 62,
    "inferior_vena_cava": 63,
    "portal_vein_and_splenic_vein": 64,
    "iliac_artery_left": 65,
    "iliac_artery_right": 66,
    "iliac_vena_left": 67,
    "iliac_vena_right": 68,
    "humerus_left": 69,
    "humerus_right": 70,
    "scapula_left": 71,
    "scapula_right": 72,
    "clavicula_left": 73,
    "clavicula_right": 74,
    "femur_left": 75,
    "femur_right": 76,
    "hip_left": 77,
    "hip_right": 78,
    "spinal_cord": 79,
    "gluteus_maximus_left": 80,
    "gluteus_maximus_right": 81,
    "gluteus_medius_left": 82,
    "gluteus_medius_right": 83,
    "gluteus_minimus_left": 84,
    "gluteus_minimus_right": 85,
    "autochthon_left": 86,
    "autochthon_right": 87,
    "iliopsoas_left": 88,
    "iliopsoas_right": 89,
    "brain": 90,
    "skull": 91,
    "rib_left_1": 92,
    "rib_left_2": 93,
    "rib_left_3": 94,
    "rib_left_4": 95,
    "rib_left_5": 96,
    "rib_left_6": 97,
    "rib_left_7": 98,
    "rib_left_8": 99,
    "rib_left_9": 100,
    "rib_left_10": 101,
    "rib_left_11": 102,
    "rib_left_12": 103,
    "rib_right_1": 104,
    "rib_right_2": 105,
    "rib_right_3": 106,
    "rib_right_4": 107,
    "rib_right_5": 108,
    "rib_right_6": 109,
    "rib_right_7": 110,
    "rib_right_8": 111,
    "rib_right_9": 112,
    "rib_right_10": 113,
    "rib_right_11": 114,
    "rib_right_12": 115,
    "sternum": 116,
    "costal_cartilages": 117
}


def _find_image_file(case_dir: str, case_id: str, tags: List[str]) -> str:
    for tag in tags:
        for candidate in [
            f"{tag}.nii.gz", f"{case_id}_{tag}.nii.gz", f"{case_id}-{tag}.nii.gz",
            f"{tag}.nii", f"{case_id}_{tag}.nii", f"{case_id}-{tag}.nii"
        ]:
            p = join(case_dir, candidate)
            if isfile(p):
                return p
    # Fallback: any nifti file in case_dir not matching segmentations/seg
    all_niftis = subfiles(case_dir, suffix='.nii.gz', join=False) + subfiles(case_dir, suffix='.nii', join=False)
    for f in all_niftis:
        fn_lower = f.lower()
        if not any(k in fn_lower for k in ['seg', 'label', 'mask']):
            return join(case_dir, f)
    raise FileNotFoundError(f"Could not locate CT image file in {case_dir} for case {case_id}")


def _process_case(case_info: Tuple[str, str, str, str, Dict[str, int]]) -> None:
    case_dir, case_id, imagestr, labelstr, label_dict = case_info
    image_file = _find_image_file(case_dir, case_id, ['ct', 'imaging', 'image'])
    shutil.copy(image_file, join(imagestr, f"{case_id}_0000.nii.gz"))

    # Check if a merged segmentation file already exists
    for candidate in [
        f"{case_id}.nii.gz", "segmentation.nii.gz", "seg.nii.gz",
        f"{case_id}_seg.nii.gz", f"{case_id}-seg.nii.gz",
        f"{case_id}.nii", "segmentation.nii", "seg.nii"
    ]:
        p = join(case_dir, candidate)
        if isfile(p):
            shutil.copy(p, join(labelstr, f"{case_id}.nii.gz"))
            return

    # Check segmentations directory
    seg_dir = join(case_dir, "segmentations")
    if not isdir(seg_dir):
        seg_dir = case_dir

    mask_files = subfiles(seg_dir, suffix='.nii.gz', join=False) + subfiles(seg_dir, suffix='.nii', join=False)
    # Exclude the image file itself if scanning case_dir directly
    img_basename = os.path.basename(image_file)
    mask_files = [f for f in mask_files if f != img_basename]

    if len(mask_files) == 0:
        raise FileNotFoundError(f"No segmentation masks found for case {case_id} in {case_dir}")

    ref_img = nib.load(image_file)
    max_label_id = max(label_dict.values())
    dtype = np.uint8 if max_label_id <= 255 else np.uint16
    combined_mask = np.zeros(ref_img.shape, dtype=dtype)

    for mf in mask_files:
        struct_name = mf[:-7] if mf.endswith('.nii.gz') else mf[:-4]
        if struct_name in label_dict:
            label_val = label_dict[struct_name]
            if label_val > 0:
                mask_nib = nib.load(join(seg_dir, mf))
                mask_data = np.asanyarray(mask_nib.dataobj)
                combined_mask[mask_data > 0] = label_val

    out_nib = nib.Nifti1Image(combined_mask, affine=ref_img.affine, header=ref_img.header)
    nib.save(out_nib, join(labelstr, f"{case_id}.nii.gz"))


def convert_totalsegmentator_v3_ct(input_folder: str,
                                   nnunet_dataset_id: int = 231,
                                   task_name: str = "TotalSegmentatorV3_CT",
                                   num_processes: int = 8) -> None:
    foldername = "Dataset%03.0d_%s" % (nnunet_dataset_id, task_name)

    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    cases = [d for d in subdirs(input_folder, join=False) if not d.startswith('.') and d != '__MACOSX']
    if len(cases) == 0:
        raise RuntimeError(f"No case subdirectories found in {input_folder}")
    cases.sort()

    label_dict = dict(TOTAL_V3_CT_LABELS)

    tasks = [(join(input_folder, tr), tr, imagestr, labelstr, label_dict) for tr in cases]
    if num_processes > 1 and len(tasks) > 1:
        with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
            pool.map(_process_case, tasks)
    else:
        for t in tasks:
            _process_case(t)

    generate_dataset_json(
        out_base,
        channel_names={0: "CT"},
        labels=label_dict,
        num_training_cases=len(cases),
        file_ending='.nii.gz',
        regions_class_order=None,
        dataset_name=task_name,
        reference='https://zenodo.org/records/22688904',
        release='3.0.0',
        license='see https://zenodo.org/records/22688904',
        overwrite_image_reader_writer='NibabelIOWithReorient',
        description="TotalSegmentator CT v3.0.0 dataset containing 117 anatomical structures across 1939 CT scans."
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Convert TotalSegmentator V3 CT dataset to nnU-Net format.")
    parser.add_argument('input_folder', type=str,
                        help="Path to the extracted TotalSegmentator V3 CT dataset (containing sXXXX subject folders).")
    parser.add_argument('-d', required=False, type=int, default=231,
                        help='nnU-Net Dataset ID, default: 231')
    parser.add_argument('-t', '--task_name', required=False, type=str, default='TotalSegmentatorV3_CT',
                        help='Task name for the dataset, default: TotalSegmentatorV3_CT')
    parser.add_argument('-np', '--num_processes', required=False, type=int, default=8,
                        help='Number of processes for parallel conversion, default: 8')
    args = parser.parse_args()
    convert_totalsegmentator_v3_ct(args.input_folder, args.d, task_name=args.task_name, num_processes=args.num_processes)
