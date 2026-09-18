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


def convert_brats_ped(brats_base_dir: str, nnunet_dataset_id: int = 228):
    task_name = "BraTS_PED"

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

    for tr in cases:
        case_dir = join(brats_base_dir, tr)

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

    generate_dataset_json(
        out_base,
        channel_names={
            0: "T1n",
            1: "T1c",
            2: "T2w",
            3: "T2f"
        },
        labels={
            "background": 0,
            "ET": 1,
            "NET": 2,
            "CC": 3,
            "ED": 4
        },
        num_training_cases=len(cases),
        file_ending='.nii.gz',
        regions_class_order=None,
        dataset_name=task_name,
        reference='https://www.synapse.org/Synapse:syn51156910/wiki/622536',
        release='1.0',
        description="ASNR-MICCAI BraTS-PED: Pediatric Brain Tumor Segmentation Challenge. "
                    "Labels: 1: Enhancing Tumor (ET), 2: Non-enhancing Tumor (NET), "
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
    args = parser.parse_args()
    convert_brats_ped(args.input_folder, args.d)
