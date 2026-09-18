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

# Canonical 50 classes for TotalSegmentator v3 (MRI)
TOTAL_MR_LABELS: Dict[str, int] = {
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
    "lung_left": 10,
    "lung_right": 11,
    "esophagus": 12,
    "small_bowel": 13,
    "duodenum": 14,
    "colon": 15,
    "urinary_bladder": 16,
    "prostate": 17,
    "sacrum": 18,
    "vertebrae": 19,
    "intervertebral_discs": 20,
    "spinal_cord": 21,
    "heart": 22,
    "aorta": 23,
    "inferior_vena_cava": 24,
    "portal_vein_and_splenic_vein": 25,
    "iliac_artery_left": 26,
    "iliac_artery_right": 27,
    "iliac_vena_left": 28,
    "iliac_vena_right": 29,
    "humerus_left": 30,
    "humerus_right": 31,
    "scapula_left": 32,
    "scapula_right": 33,
    "clavicula_left": 34,
    "clavicula_right": 35,
    "femur_left": 36,
    "femur_right": 37,
    "hip_left": 38,
    "hip_right": 39,
    "gluteus_maximus_left": 40,
    "gluteus_maximus_right": 41,
    "gluteus_medius_left": 42,
    "gluteus_medius_right": 43,
    "gluteus_minimus_left": 44,
    "gluteus_minimus_right": 45,
    "autochthon_left": 46,
    "autochthon_right": 47,
    "iliopsoas_left": 48,
    "iliopsoas_right": 49,
    "brain": 50
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
    raise FileNotFoundError(f"Could not locate MRI image file in {case_dir} for case {case_id}")


def _process_case(case_info: Tuple[str, str, str, str, Dict[str, int]]) -> None:
    case_dir, case_id, imagestr, labelstr, label_dict = case_info
    image_file = _find_image_file(case_dir, case_id, ['mri', 'imaging', 'image'])
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


def convert_totalsegmentator_v3_mri(input_folder: str,
                                    nnunet_dataset_id: int = 232,
                                    task_name: str = "TotalSegmentatorV3_MRI",
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

    label_dict = dict(TOTAL_MR_LABELS)

    tasks = [(join(input_folder, tr), tr, imagestr, labelstr, label_dict) for tr in cases]
    if num_processes > 1 and len(tasks) > 1:
        with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
            pool.map(_process_case, tasks)
    else:
        for t in tasks:
            _process_case(t)

    generate_dataset_json(
        out_base,
        channel_names={0: "MRI"},
        labels=label_dict,
        num_training_cases=len(cases),
        file_ending='.nii.gz',
        regions_class_order=None,
        dataset_name=task_name,
        reference='https://zenodo.org/records/22688334',
        release='3.0.0',
        license='see https://zenodo.org/records/22688334',
        overwrite_image_reader_writer='NibabelIOWithReorient',
        description="TotalSegmentator MRI v3.0.0 dataset containing 50 anatomical structures across 1296 MR scans."
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Convert TotalSegmentator V3 MRI dataset to nnU-Net format.")
    parser.add_argument('input_folder', type=str,
                        help="Path to the extracted TotalSegmentator V3 MRI dataset (containing sXXXX subject folders).")
    parser.add_argument('-d', required=False, type=int, default=232,
                        help='nnU-Net Dataset ID, default: 232')
    parser.add_argument('-t', '--task_name', required=False, type=str, default='TotalSegmentatorV3_MRI',
                        help='Task name for the dataset, default: TotalSegmentatorV3_MRI')
    parser.add_argument('-np', '--num_processes', required=False, type=int, default=8,
                        help='Number of processes for parallel conversion, default: 8')
    args = parser.parse_args()
    convert_totalsegmentator_v3_mri(args.input_folder, args.d, task_name=args.task_name, num_processes=args.num_processes)
