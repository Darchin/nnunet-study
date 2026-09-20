import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import nibabel as nib

from batchgenerators.utilities.file_and_folder_operations import join, isfile, isdir, subfiles
from nnunetv2.dataset_conversion.Dataset228_BraTS_PED import convert_brats_ped


class TestDataset228BraTSPED(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.raw_dir = join(self.temp_dir.name, "nnUNet_raw")
        os.makedirs(self.raw_dir, exist_ok=True)
        self.input_dir = join(self.temp_dir.name, "raw_brats_ped")
        os.makedirs(self.input_dir, exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_dummy_nifti(self, path: str, shape=(5, 5, 5), dtype=np.uint8):
        data = np.zeros(shape, dtype=dtype)
        img = nib.Nifti1Image(data, np.eye(4))
        nib.save(img, path)

    def test_convert_brats_ped(self):
        # Case 1 with standard challenge naming: <case>-<modality>.nii.gz
        case1_id = "BraTS-PED-00001-000"
        case1_dir = join(self.input_dir, case1_id)
        os.makedirs(case1_dir, exist_ok=True)
        for mod in ["t1n", "t1c", "t2w", "t2f", "seg"]:
            self._create_dummy_nifti(join(case1_dir, f"{case1_id}-{mod}.nii.gz"))

        # Case 2 with alternative naming: <case>_<modality>.nii.gz (t1, t1ce, t2, flair)
        case2_id = "BraTS-PED-00002-000"
        case2_dir = join(self.input_dir, case2_id)
        os.makedirs(case2_dir, exist_ok=True)
        for mod in ["t1", "t1ce", "t2", "flair", "seg"]:
            self._create_dummy_nifti(join(case2_dir, f"{case2_id}_{mod}.nii.gz"))

        with patch.dict(os.environ, {'nnUNet_raw': self.raw_dir}, clear=False):
            # Also patch nnUNet_raw imported in Dataset228_BraTS_PED
            with patch("nnunetv2.dataset_conversion.Dataset228_BraTS_PED.nnUNet_raw", self.raw_dir):
                convert_brats_ped(self.input_dir, nnunet_dataset_id=228)

        output_dataset_dir = join(self.raw_dir, "Dataset228_BraTS_PED")
        self.assertTrue(isdir(output_dataset_dir))

        imagestr = join(output_dataset_dir, "imagesTr")
        labelstr = join(output_dataset_dir, "labelsTr")
        self.assertTrue(isdir(imagestr))
        self.assertTrue(isdir(labelstr))

        # Check imagesTr files
        for c in [case1_id, case2_id]:
            for ch in ["0000", "0001", "0002", "0003"]:
                img_file = join(imagestr, f"{c}_{ch}.nii.gz")
                self.assertTrue(isfile(img_file), f"Missing image file: {img_file}")
            lbl_file = join(labelstr, f"{c}.nii.gz")
            self.assertTrue(isfile(lbl_file), f"Missing label file: {lbl_file}")

        # Check dataset.json
        dataset_json_path = join(output_dataset_dir, "dataset.json")
        self.assertTrue(isfile(dataset_json_path))
        with open(dataset_json_path, 'r') as f:
            dj = json.load(f)

        self.assertEqual(dj["numTraining"], 2)
        self.assertEqual(dj["file_ending"], ".nii.gz")
        self.assertEqual(dj["channel_names"], {"0": "T1n", "1": "T1c", "2": "T2w", "3": "T2f"})
        self.assertEqual(dj["labels"], {
            "background": 0,
            "whole tumor": [1, 2, 3, 4],
            "tumor core": [1, 2, 3],
            "cystic component": [3],
            "enhancing tumor": [1]
        })
        self.assertEqual(dj.get("regions_class_order"), [4, 2, 3, 1])

        # Verify dataset integrity using nnU-Net's built-in verifier
        from nnunetv2.experiment_planning.verify_dataset_integrity import verify_dataset_integrity
        verify_dataset_integrity(output_dataset_dir, num_processes=1)

    def test_convert_brats_ped_multiprocessing(self):
        case_id = "BraTS-PED-00003-000"
        case_dir = join(self.input_dir, case_id)
        os.makedirs(case_dir, exist_ok=True)
        for mod in ["t1n", "t1c", "t2w", "t2f", "seg"]:
            self._create_dummy_nifti(join(case_dir, f"{case_id}-{mod}.nii.gz"))

        with patch.dict(os.environ, {'nnUNet_raw': self.raw_dir}, clear=False):
            with patch("nnunetv2.dataset_conversion.Dataset228_BraTS_PED.nnUNet_raw", self.raw_dir):
                convert_brats_ped(self.input_dir, nnunet_dataset_id=228, num_processes=2)

        output_dataset_dir = join(self.raw_dir, "Dataset228_BraTS_PED")
        self.assertTrue(isdir(join(output_dataset_dir, "imagesTr")))
        self.assertTrue(isfile(join(output_dataset_dir, "imagesTr", f"{case_id}_0000.nii.gz")))
        self.assertTrue(isfile(join(output_dataset_dir, "labelsTr", f"{case_id}.nii.gz")))

    def test_convert_brats_ped_no_regions(self):
        case_id = "BraTS-PED-00004-000"
        case_dir = join(self.input_dir, case_id)
        os.makedirs(case_dir, exist_ok=True)
        for mod in ["t1n", "t1c", "t2w", "t2f", "seg"]:
            self._create_dummy_nifti(join(case_dir, f"{case_id}-{mod}.nii.gz"))

        with patch.dict(os.environ, {'nnUNet_raw': self.raw_dir}, clear=False):
            with patch("nnunetv2.dataset_conversion.Dataset228_BraTS_PED.nnUNet_raw", self.raw_dir):
                convert_brats_ped(self.input_dir, nnunet_dataset_id=228, num_processes=1, use_regions=False)

        output_dataset_dir = join(self.raw_dir, "Dataset228_BraTS_PED")
        dataset_json_path = join(output_dataset_dir, "dataset.json")
        with open(dataset_json_path, 'r') as f:
            dj = json.load(f)

        self.assertEqual(dj["labels"], {"background": 0, "ET": 1, "NET": 2, "CC": 3, "ED": 4})
        self.assertIsNone(dj.get("regions_class_order"))


if __name__ == '__main__':
    unittest.main()
