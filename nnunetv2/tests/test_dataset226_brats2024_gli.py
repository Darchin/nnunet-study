import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import nibabel as nib

from batchgenerators.utilities.file_and_folder_operations import join, isfile, isdir
import importlib
brats_gli_module = importlib.import_module("nnunetv2.dataset_conversion.Dataset226_BraTS2024-GLI")
convert_brats2024_gli = brats_gli_module.convert_brats2024_gli


class TestDataset226BraTS2024GLI(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.raw_dir = join(self.temp_dir.name, "nnUNet_raw")
        os.makedirs(self.raw_dir, exist_ok=True)
        self.input_dir = join(self.temp_dir.name, "raw_brats_gli")
        os.makedirs(self.input_dir, exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_dummy_nifti(self, path: str, shape=(5, 5, 5), dtype=np.uint8, values=None):
        if values is not None:
            data = np.array(values, dtype=dtype)
        else:
            data = np.zeros(shape, dtype=dtype)
        img = nib.Nifti1Image(data, np.eye(4))
        nib.save(img, path)

    def test_convert_brats2024_gli(self):
        # Case 1: Standard challenge naming: <case>-<modality>.nii.gz
        case1_id = "BraTS-GLI-00001-000"
        case1_dir = join(self.input_dir, case1_id)
        os.makedirs(case1_dir, exist_ok=True)
        for mod in ["t1n", "t1c", "t2w", "t2f"]:
            self._create_dummy_nifti(join(case1_dir, f"{case1_id}-{mod}.nii.gz"))
        # Create segmentation with labels 0, 1, 2, 3, 4
        seg_data1 = np.zeros((5, 5, 5), dtype=np.uint8)
        seg_data1[1, 1, 1] = 1  # NETC
        seg_data1[2, 2, 2] = 2  # SNFH
        seg_data1[3, 3, 3] = 3  # ET
        seg_data1[4, 4, 4] = 4  # RC
        self._create_dummy_nifti(join(case1_dir, f"{case1_id}-seg.nii.gz"), values=seg_data1)

        # Case 2: Alternative naming: <case>_<modality>.nii.gz (t1, t1ce, t2, flair, seg)
        case2_id = "BraTS-GLI-00002-000"
        case2_dir = join(self.input_dir, case2_id)
        os.makedirs(case2_dir, exist_ok=True)
        for mod in ["t1", "t1ce", "t2", "flair"]:
            self._create_dummy_nifti(join(case2_dir, f"{case2_id}_{mod}.nii.gz"))
        seg_data2 = np.zeros((5, 5, 5), dtype=np.uint8)
        seg_data2[2, 2, 2] = 1
        self._create_dummy_nifti(join(case2_dir, f"{case2_id}_seg.nii.gz"), values=seg_data2)

        with patch.dict(os.environ, {'nnUNet_raw': self.raw_dir}, clear=False):
            with patch.object(brats_gli_module, "nnUNet_raw", self.raw_dir):
                convert_brats2024_gli(self.input_dir, nnunet_dataset_id=226, num_processes=1)

        output_dataset_dir = join(self.raw_dir, "Dataset226_BraTS2024_GLI")
        self.assertTrue(isdir(output_dataset_dir))

        imagestr = join(output_dataset_dir, "imagesTr")
        labelstr = join(output_dataset_dir, "labelsTr")
        self.assertTrue(isdir(imagestr))
        self.assertTrue(isdir(labelstr))

        # Check imagesTr and labelsTr files
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
        self.assertEqual(dj["labels"], {"background": 0, "NETC": 1, "SNFH": 2, "ET": 3, "RC": 4})
        self.assertIsNone(dj.get("regions_class_order"))

        # Verify dataset integrity using nnU-Net's built-in verifier
        from nnunetv2.experiment_planning.verify_dataset_integrity import verify_dataset_integrity
        verify_dataset_integrity(output_dataset_dir, num_processes=1)

    def test_convert_brats2024_gli_multiprocessing(self):
        case_id = "BraTS-GLI-00003-000"
        case_dir = join(self.input_dir, case_id)
        os.makedirs(case_dir, exist_ok=True)
        for mod in ["t1n", "t1c", "t2w", "t2f", "seg"]:
            self._create_dummy_nifti(join(case_dir, f"{case_id}-{mod}.nii.gz"))

        with patch.dict(os.environ, {'nnUNet_raw': self.raw_dir}, clear=False):
            with patch.object(brats_gli_module, "nnUNet_raw", self.raw_dir):
                convert_brats2024_gli(self.input_dir, nnunet_dataset_id=226, num_processes=2)

        output_dataset_dir = join(self.raw_dir, "Dataset226_BraTS2024_GLI")
        self.assertTrue(isdir(join(output_dataset_dir, "imagesTr")))
        self.assertTrue(isfile(join(output_dataset_dir, "imagesTr", f"{case_id}_0000.nii.gz")))
        self.assertTrue(isfile(join(output_dataset_dir, "labelsTr", f"{case_id}.nii.gz")))


if __name__ == '__main__':
    unittest.main()

