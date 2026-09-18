import json
import os
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, isfile, isdir

from nnunetv2.dataset_conversion.Dataset231_TotalSegmentatorV3_CT import (
    convert_totalsegmentator_v3_ct, TOTAL_V3_CT_LABELS
)
from nnunetv2.dataset_conversion.Dataset232_TotalSegmentatorV3_MRI import (
    convert_totalsegmentator_v3_mri, TOTAL_MR_LABELS
)
from nnunetv2.experiment_planning.verify_dataset_integrity import verify_dataset_integrity


class TestDataset231And232TotalSegmentatorV3(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.raw_dir = join(self.temp_dir.name, "nnUNet_raw")
        os.makedirs(self.raw_dir, exist_ok=True)
        self.input_dir_ct = join(self.temp_dir.name, "raw_ts_ct")
        os.makedirs(self.input_dir_ct, exist_ok=True)
        self.input_dir_mri = join(self.temp_dir.name, "raw_ts_mri")
        os.makedirs(self.input_dir_mri, exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_dummy_nifti(self, path: str, shape=(8, 8, 8), dtype=np.float32, values=None):
        if values is not None:
            data = np.array(values, dtype=dtype)
        else:
            data = np.zeros(shape, dtype=dtype)
        img = nib.Nifti1Image(data, np.eye(4))
        nib.save(img, path)

    def test_convert_totalsegmentator_v3_ct(self):
        case1_id = "s0001"
        case1_dir = join(self.input_dir_ct, case1_id)
        seg1_dir = join(case1_dir, "segmentations")
        os.makedirs(seg1_dir, exist_ok=True)

        self._create_dummy_nifti(join(case1_dir, "ct.nii.gz"), shape=(8, 8, 8))

        # Binary masks for specific structures
        spleen_mask = np.zeros((8, 8, 8), dtype=np.uint8)
        spleen_mask[1, 1, 1] = 1
        self._create_dummy_nifti(join(seg1_dir, "spleen.nii.gz"), values=spleen_mask, dtype=np.uint8)

        vert_l6_mask = np.zeros((8, 8, 8), dtype=np.uint8)
        vert_l6_mask[2, 2, 2] = 1
        self._create_dummy_nifti(join(seg1_dir, "vertebrae_L6.nii.gz"), values=vert_l6_mask, dtype=np.uint8)

        costal_mask = np.zeros((8, 8, 8), dtype=np.uint8)
        costal_mask[3, 3, 3] = 1
        self._create_dummy_nifti(join(seg1_dir, "costal_cartilages.nii.gz"), values=costal_mask, dtype=np.uint8)

        # Case 2: test pre-merged single segmentation file fallback
        case2_id = "s0002"
        case2_dir = join(self.input_dir_ct, case2_id)
        os.makedirs(case2_dir, exist_ok=True)
        self._create_dummy_nifti(join(case2_dir, "ct.nii.gz"), shape=(8, 8, 8))
        case2_seg = np.zeros((8, 8, 8), dtype=np.uint8)
        case2_seg[4, 4, 4] = TOTAL_V3_CT_LABELS["liver"]
        self._create_dummy_nifti(join(case2_dir, "segmentation.nii.gz"), values=case2_seg, dtype=np.uint8)

        with patch.dict(os.environ, {'nnUNet_raw': self.raw_dir}, clear=False):
            with patch("nnunetv2.dataset_conversion.Dataset231_TotalSegmentatorV3_CT.nnUNet_raw", self.raw_dir):
                convert_totalsegmentator_v3_ct(self.input_dir_ct, nnunet_dataset_id=231, num_processes=1)

        output_dir = join(self.raw_dir, "Dataset231_TotalSegmentatorV3_CT")
        self.assertTrue(isdir(output_dir))

        imagestr = join(output_dir, "imagesTr")
        labelstr = join(output_dir, "labelsTr")
        self.assertTrue(isdir(imagestr))
        self.assertTrue(isdir(labelstr))

        # Check images and labels
        for cid in [case1_id, case2_id]:
            self.assertTrue(isfile(join(imagestr, f"{cid}_0000.nii.gz")))
            self.assertTrue(isfile(join(labelstr, f"{cid}.nii.gz")))

        # Check merged label values for case 1
        lbl_img = nib.load(join(labelstr, f"{case1_id}.nii.gz"))
        lbl_data = np.asanyarray(lbl_img.dataobj)
        self.assertEqual(lbl_data[1, 1, 1], TOTAL_V3_CT_LABELS["spleen"])
        self.assertEqual(lbl_data[2, 2, 2], TOTAL_V3_CT_LABELS["vertebrae_L6"])
        self.assertEqual(lbl_data[3, 3, 3], TOTAL_V3_CT_LABELS["costal_cartilages"])
        self.assertEqual(lbl_data[0, 0, 0], 0)

        # Check dataset.json
        with open(join(output_dir, "dataset.json"), 'r') as f:
            dj = json.load(f)

        self.assertEqual(dj["numTraining"], 2)
        self.assertEqual(dj["channel_names"], {"0": "CT"})
        self.assertEqual(dj["labels"]["spleen"], 1)
        self.assertEqual(dj["labels"]["vertebrae_L6"], 26)
        self.assertEqual(dj["labels"]["costal_cartilages"], 117)
        self.assertEqual(dj["file_ending"], ".nii.gz")

        # Run verify_dataset_integrity
        verify_dataset_integrity(output_dir, num_processes=1)

    def test_convert_totalsegmentator_v3_mri(self):
        case1_id = "s0001"
        case1_dir = join(self.input_dir_mri, case1_id)
        seg1_dir = join(case1_dir, "segmentations")
        os.makedirs(seg1_dir, exist_ok=True)

        self._create_dummy_nifti(join(case1_dir, "mri.nii.gz"), shape=(8, 8, 8))

        # Binary masks for MRI structures
        spleen_mask = np.zeros((8, 8, 8), dtype=np.uint8)
        spleen_mask[1, 1, 1] = 1
        self._create_dummy_nifti(join(seg1_dir, "spleen.nii.gz"), values=spleen_mask, dtype=np.uint8)

        lung_mask = np.zeros((8, 8, 8), dtype=np.uint8)
        lung_mask[2, 2, 2] = 1
        self._create_dummy_nifti(join(seg1_dir, "lung_left.nii.gz"), values=lung_mask, dtype=np.uint8)

        vert_mask = np.zeros((8, 8, 8), dtype=np.uint8)
        vert_mask[3, 3, 3] = 1
        self._create_dummy_nifti(join(seg1_dir, "vertebrae.nii.gz"), values=vert_mask, dtype=np.uint8)

        brain_mask = np.zeros((8, 8, 8), dtype=np.uint8)
        brain_mask[4, 4, 4] = 1
        self._create_dummy_nifti(join(seg1_dir, "brain.nii.gz"), values=brain_mask, dtype=np.uint8)

        with patch.dict(os.environ, {'nnUNet_raw': self.raw_dir}, clear=False):
            with patch("nnunetv2.dataset_conversion.Dataset232_TotalSegmentatorV3_MRI.nnUNet_raw", self.raw_dir):
                convert_totalsegmentator_v3_mri(self.input_dir_mri, nnunet_dataset_id=232, num_processes=1)

        output_dir = join(self.raw_dir, "Dataset232_TotalSegmentatorV3_MRI")
        self.assertTrue(isdir(output_dir))

        imagestr = join(output_dir, "imagesTr")
        labelstr = join(output_dir, "labelsTr")
        self.assertTrue(isdir(imagestr))
        self.assertTrue(isdir(labelstr))

        self.assertTrue(isfile(join(imagestr, f"{case1_id}_0000.nii.gz")))
        self.assertTrue(isfile(join(labelstr, f"{case1_id}.nii.gz")))

        # Check merged label values for MRI case 1
        lbl_img = nib.load(join(labelstr, f"{case1_id}.nii.gz"))
        lbl_data = np.asanyarray(lbl_img.dataobj)
        self.assertEqual(lbl_data[1, 1, 1], TOTAL_MR_LABELS["spleen"])
        self.assertEqual(lbl_data[2, 2, 2], TOTAL_MR_LABELS["lung_left"])
        self.assertEqual(lbl_data[3, 3, 3], TOTAL_MR_LABELS["vertebrae"])
        self.assertEqual(lbl_data[4, 4, 4], TOTAL_MR_LABELS["brain"])

        # Check dataset.json
        with open(join(output_dir, "dataset.json"), 'r') as f:
            dj = json.load(f)

        self.assertEqual(dj["numTraining"], 1)
        self.assertEqual(dj["channel_names"], {"0": "MRI"})
        self.assertEqual(dj["labels"]["spleen"], 1)
        self.assertEqual(dj["labels"]["lung_left"], 10)
        self.assertEqual(dj["labels"]["vertebrae"], 19)
        self.assertEqual(dj["labels"]["brain"], 50)
        self.assertEqual(dj["file_ending"], ".nii.gz")

        # Run verify_dataset_integrity
        verify_dataset_integrity(output_dir, num_processes=1)

    def test_multiprocessing(self):
        # Create two cases for CT
        for cid in ["s0010", "s0011"]:
            cdir = join(self.input_dir_ct, cid)
            sdir = join(cdir, "segmentations")
            os.makedirs(sdir, exist_ok=True)
            self._create_dummy_nifti(join(cdir, "ct.nii.gz"), shape=(8, 8, 8))
            self._create_dummy_nifti(join(sdir, "spleen.nii.gz"), shape=(8, 8, 8))

        with patch.dict(os.environ, {'nnUNet_raw': self.raw_dir}, clear=False):
            with patch("nnunetv2.dataset_conversion.Dataset231_TotalSegmentatorV3_CT.nnUNet_raw", self.raw_dir):
                convert_totalsegmentator_v3_ct(self.input_dir_ct, nnunet_dataset_id=231, num_processes=2)

        output_dir = join(self.raw_dir, "Dataset231_TotalSegmentatorV3_CT")
        for cid in ["s0010", "s0011"]:
            self.assertTrue(isfile(join(output_dir, "imagesTr", f"{cid}_0000.nii.gz")))
            self.assertTrue(isfile(join(output_dir, "labelsTr", f"{cid}.nii.gz")))


if __name__ == '__main__':
    unittest.main()
