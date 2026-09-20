import multiprocessing
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw


def _find_case_file(case_dir: str, case_id: str, tags: list) -> str:
    for tag in tags:
        for candidate in [f"{case_id}-{tag}.nii.gz", f"{case_id}_{tag}.nii.gz", f"{tag}.nii.gz"]:
            p = join(case_dir, candidate)
            if isfile(p):
                return p
    # Fallback to suffix search
    for f in subfiles(case_dir, suffix='.nii.gz', join=False):
        for tag in tags:
            if f.endswith(f"-{tag}.nii.gz") or f.endswith(f"_{tag}.nii.gz") or f == f"{tag}.nii.gz":
                return join(case_dir, f)
    raise FileNotFoundError(f"Could not find file with tags {tags} in {case_dir}")


def _copy_case(case_info: tuple) -> None:
    case_dir, tr, imagestr, labelstr = case_info
    t1n = _find_case_file(case_dir, tr, ['t1n', 't1'])
    t1c = _find_case_file(case_dir, tr, ['t1c', 't1ce'])
    t2w = _find_case_file(case_dir, tr, ['t2w', 't2'])
    t2f = _find_case_file(case_dir, tr, ['t2f', 'flair'])
    seg = _find_case_file(case_dir, tr, ['seg', 'segmentation'])

    shutil.copy(t1n, join(imagestr, f"{tr}_0000.nii.gz"))
    shutil.copy(t1c, join(imagestr, f"{tr}_0001.nii.gz"))
    shutil.copy(t2w, join(imagestr, f"{tr}_0002.nii.gz"))
    shutil.copy(t2f, join(imagestr, f"{tr}_0003.nii.gz"))
    shutil.copy(seg, join(labelstr, f"{tr}.nii.gz"))


def convert_brats_ped(brats_base_dir: str,
                      nnunet_dataset_id: int = 228,
                      task_name: str = "BraTS_PED",
                      num_processes: int = 8,
                      use_regions: bool = True):
    foldername = "Dataset%03.0d_%s" % (nnunet_dataset_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    cases = subdirs(brats_base_dir, prefix='BraTS', join=False)
    if len(cases) == 0:
        cases = [d for d in subdirs(brats_base_dir, join=False) if not d.startswith('.')]
    cases.sort()

    tasks = [(join(brats_base_dir, tr), tr, imagestr, labelstr) for tr in cases]
    if num_processes > 1 and len(tasks) > 1:
        with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
            pool.map(_copy_case, tasks)
    else:
        for t in tasks:
            _copy_case(t)

    if use_regions:
        labels = {
            "background": 0,
            "whole tumor": (1, 2, 3, 4),
            "tumor core": (1, 2, 3),
            "cystic component": (3,),
            "enhancing tumor": (1,)
        }
        regions_class_order = (4, 2, 3, 1)
    else:
        labels = {
            "background": 0,
            "ET": 1,
            "NET": 2,
            "CC": 3,
            "ED": 4
        }
        regions_class_order = None

    generate_dataset_json(
        out_base,
        channel_names={
            0: "T1n",
            1: "T1c",
            2: "T2w",
            3: "T2f"
        },
        labels=labels,
        num_training_cases=len(cases),
        file_ending='.nii.gz',
        regions_class_order=regions_class_order,
        dataset_name=task_name,
        reference='https://www.synapse.org/Synapse:syn51156910/wiki/622536',
        release='1.0',
        license='see https://www.synapse.org/Synapse:syn51156910/wiki/622536',
        description="ASNR-MICCAI BraTS-PED: Pediatric Brain Tumor Segmentation Challenge. "
                    "Target regions: Whole Tumor (WT: 1+2+3+4), Tumor Core (TC: 1+2+3), "
                    "Cystic Component (CC: 3), Enhancing Tumor (ET: 1). "
                    "Underlying classes: 1: Enhancing Tumor (ET), 2: Non-enhancing Tumor (NET), "
                    "3: Cystic Component (CC), 4: Peritumoral Edema (ED)."
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Convert BraTS-PED dataset to nnU-Net format.")
    parser.add_argument('input_folder', type=str,
                        help="The downloaded and extracted BraTS-PED dataset directory containing case subfolders "
                             "(e.g., BraTS-PED-XXXXX-XXX).")
    parser.add_argument('-d', required=False, type=int, default=228,
                        help='nnU-Net Dataset ID, default: 228')
    parser.add_argument('-t', '--task_name', required=False, type=str, default='BraTS_PED',
                        help='Task name for the dataset, default: BraTS_PED')
    parser.add_argument('-np', '--num_processes', required=False, type=int, default=8,
                        help='Number of processes to copy case files, default: 8')
    parser.add_argument('--no_regions', action='store_true', default=False,
                        help='Disable region-based training and use mutually exclusive class labels (ET: 1, NET: 2, CC: 3, ED: 4).')
    args = parser.parse_args()
    convert_brats_ped(args.input_folder, args.d, task_name=args.task_name,
                      num_processes=args.num_processes, use_regions=not args.no_regions)
