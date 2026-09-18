import os
import re
import shutil
from typing import List, Tuple, Optional
from batchgenerators.utilities.file_and_folder_operations import (
    join, isdir, isfile, subfiles, maybe_mkdir_p
)
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed


def _find_case_file(case_dir: str, tags: list) -> Optional[str]:
    """
    Search for a file matching any of the given tags within a case directory.
    First tries exact match, then suffix match, then substring match.
    """
    for tag in tags:
        for candidate in [f"{tag}.nii.gz", f"{tag}.nii"]:
            p = join(case_dir, candidate)
            if isfile(p):
                return p

    # Look through nifti files in the directory
    nii_files = subfiles(case_dir, suffix='.nii.gz', join=False)
    if not nii_files:
        nii_files = subfiles(case_dir, suffix='.nii', join=False)

    for f in nii_files:
        f_lower = f.lower()
        for tag in tags:
            tag_lower = tag.lower()
            if (f_lower == f"{tag_lower}.nii.gz" or
                    f_lower.endswith(f"-{tag_lower}.nii.gz") or
                    f_lower.endswith(f"_{tag_lower}.nii.gz")):
                return join(case_dir, f)

    for f in nii_files:
        f_lower = f.lower()
        for tag in tags:
            if tag.lower() in f_lower:
                return join(case_dir, f)

    return None


def _find_autopet_cases(autopet_base_dir: str) -> List[Tuple[str, str, str, str]]:
    """
    Find all AutoPET III cases in autopet_base_dir.
    Returns a list of tuples: (case_id, ct_path, pet_path, seg_path).

    Supports:
    1. Pre-organized nnU-Net format with imagesTr/ and labelsTr/.
    2. Hierarchical patient/study directory structures (e.g. TCIA / FDAT / Grand Challenge).
    3. Flat case subfolders (e.g. case_id/ containing CT, SUV, SEG).
    """
    cases = []

    # Case 1: Pre-organized nnU-Net format
    imagestr_in = join(autopet_base_dir, "imagesTr")
    labelstr_in = join(autopet_base_dir, "labelsTr")
    if isdir(imagestr_in) and isdir(labelstr_in):
        ct_files = subfiles(imagestr_in, suffix='_0000.nii.gz', join=False)
        ct_files.sort()
        for f in ct_files:
            case_id = f[:-12]
            ct_path = join(imagestr_in, f"{case_id}_0000.nii.gz")
            pet_path = join(imagestr_in, f"{case_id}_0001.nii.gz")
            seg_path = join(labelstr_in, f"{case_id}.nii.gz")
            if isfile(pet_path) and isfile(seg_path):
                cases.append((case_id, ct_path, pet_path, seg_path))
        if cases:
            return cases

    # Case 2: Traverse directory tree for case directories containing CT, PET/SUV, and SEG
    ct_tags = ['CTres', 'CT', 'ct_res', '0000']
    pet_tags = ['SUV', 'PET', '0001']
    seg_tags = ['SEG', 'segmentation', 'seg', 'mask', 'labels', 'label']

    for root, dirs, files in os.walk(autopet_base_dir):
        # Check if root contains CT, PET, and SEG files
        ct_file = _find_case_file(root, ct_tags)
        pet_file = _find_case_file(root, pet_tags)
        seg_file = _find_case_file(root, seg_tags)

        if ct_file and pet_file and seg_file:
            rel_path = os.path.relpath(root, autopet_base_dir)
            # Create a clean case_id from the relative path
            clean_id = re.sub(r'[^a-zA-Z0-9_]+', '_', rel_path.replace(os.sep, '_')).strip('_')
            if not clean_id:
                clean_id = f"case_{len(cases):04d}"
            cases.append((clean_id, ct_file, pet_file, seg_file))
            # Don't descend into subdirectories of an identified case
            dirs.clear()

    cases.sort(key=lambda x: x[0])
    return cases


def convert_autopet3(autopet_base_dir: str, nnunet_dataset_id: int = 229):
    task_name = "AutoPETIII"

    foldername = "Dataset%03.0d_%s" % (nnunet_dataset_id, task_name)

    # Setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    cases = _find_autopet_cases(autopet_base_dir)
    if len(cases) == 0:
        raise RuntimeError(
            f"Could not find any valid AutoPET cases in {autopet_base_dir}. "
            f"Expected either imagesTr/ and labelsTr/ directories or subdirectories "
            f"containing CT (e.g. CTres.nii.gz), PET (e.g. SUV.nii.gz), and SEG (e.g. SEG.nii.gz) files."
        )

    print(f"Found {len(cases)} cases. Copying to {out_base}...")
    for case_id, ct_file, pet_file, seg_file in cases:
        out_ct = join(imagestr, f"{case_id}_0000.nii.gz")
        out_pet = join(imagestr, f"{case_id}_0001.nii.gz")
        out_seg = join(labelstr, f"{case_id}.nii.gz")

        if not isfile(out_ct):
            shutil.copy(ct_file, out_ct)
        if not isfile(out_pet):
            shutil.copy(pet_file, out_pet)
        if not isfile(out_seg):
            shutil.copy(seg_file, out_seg)

    generate_dataset_json(
        out_base,
        channel_names={
            0: "CT",
            1: "CT"
        },
        labels={
            "background": 0,
            "tumor": 1
        },
        num_training_cases=len(cases),
        file_ending='.nii.gz',
        regions_class_order=None,
        dataset_name=task_name,
        reference='https://autopet-iii.grand-challenge.org/',
        release='1.0',
        description="AutoPET III: Automated Lesion Segmentation in Whole-Body PET/CT - Multitracer Multicenter generalization (FDG & PSMA). "
                    "Channels: 0: CT, 1: PET (SUV), both using CT normalization."
    )

    # If splits_final.json is present in the input directory, copy it to nnUNet_preprocessed
    splits_src = join(autopet_base_dir, "splits_final.json")
    if isfile(splits_src):
        pp_out_dir = join(nnUNet_preprocessed, foldername)
        maybe_mkdir_p(pp_out_dir)
        splits_dst = join(pp_out_dir, "splits_final.json")
        if not isfile(splits_dst):
            shutil.copy(splits_src, splits_dst)
            print(f"Copied splits_final.json to {pp_out_dir}")

    print(f"Dataset conversion complete for {foldername} ({len(cases)} cases).")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Convert AutoPET III dataset to nnU-Net format.")
    parser.add_argument('input_folder', type=str,
                        help="The downloaded and extracted AutoPET III dataset directory.")
    parser.add_argument('-d', required=False, type=int, default=229,
                        help='nnU-Net Dataset ID, default: 229')
    args = parser.parse_args()
    convert_autopet3(args.input_folder, args.d)
