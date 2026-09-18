import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import nibabel as nib

from batchgenerators.utilities.file_and_folder_operations import join, isfile, isdir
from nnunetv2.dataset_conversion.Dataset229_AutoPETIII import convert_autopet3
from nnunetv2.experiment_planning.verify_dataset_integrity import verify_dataset_integrity


class TestDataset229AutoPETIII(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.raw_dir = join(self.temp_dir.name, "nnUNet_raw")
        self.pp_dir = join(self.temp_dir.name, "nnUNet_preprocessed")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.pp_dir, exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_dummy_nifti(self, path: str, shape=(6, 6, 6), dtype=np.uint8, is_seg=False):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if is_seg:
            data = np.zeros(shape, dtype=np.uint8)
            data[2:4, 2:4, 2:4] = 1
        else:
            data = np.random.randint(0, 100, size=shape, dtype=dtype)
        img = nib.Nifti1Image(data, np.eye(4))
        nib.save(img, path)

    def test_convert_hierarchical_tcia_structure(self):
        """Test conversion on typical TCIA/FDAT hierarchical structure: patient/study/"""
        input_dir = join(self.temp_dir.name, "tcia_autopet")
        case1_dir = join(input_dir, "PETCT_01", "study_01")
        self._create_dummy_nifti(join(case1_dir, "CTres.nii.gz"))
        self._create_dummy_nifti(join(case1_dir, "SUV.nii.gz"))
        self._create_dummy_nifti(join(case1_dir, "SEG.nii.gz"), is_seg=True)

        case2_dir = join(input_dir, "PETCT_02", "study_01")
        self._create_dummy_nifti(join(case2_dir, "CTres.nii.gz"))
        self._create_dummy_nifti(join(case2_dir, "SUV.nii.gz"))
        self._create_dummy_nifti(join(case2_dir, "SEG.nii.gz"), is_seg=True)

        # Include a splits_final.json
        dummy_splits = [{"train": ["PETCT_01_study_01"], "val": ["PETCT_02_study_01"]}]
        with open(join(input_dir, "splits_final.json"), "w") as f:
            json.dump(dummy_splits, f)

        with patch("nnunetv2.dataset_conversion.Dataset229_AutoPETIII.nnUNet_raw", self.raw_dir):
            with patch("nnunetv2.dataset_conversion.Dataset229_AutoPETIII.nnUNet_preprocessed", self.pp_dir):
                convert_autopet3(input_dir, nnunet_dataset_id=229)

        out_dataset_dir = join(self.raw_dir, "Dataset229_AutoPETIII")
        self.assertTrue(isdir(out_dataset_dir))

        imagestr = join(out_dataset_dir, "imagesTr")
        labelstr = join(out_dataset_dir, "labelsTr")
        self.assertTrue(isdir(imagestr))
        self.assertTrue(isdir(labelstr))

        # Check images and labels
        for case_id in ["PETCT_01_study_01", "PETCT_02_study_01"]:
            self.assertTrue(isfile(join(imagestr, f"{case_id}_0000.nii.gz")))
            self.assertTrue(isfile(join(imagestr, f"{case_id}_0001.nii.gz")))
            self.assertTrue(isfile(join(labelstr, f"{case_id}.nii.gz")))

        # Check dataset.json
        with open(join(out_dataset_dir, "dataset.json"), "r") as f:
            dj = json.load(f)

        self.assertEqual(dj["numTraining"], 2)
        self.assertEqual(dj["file_ending"], ".nii.gz")
        self.assertEqual(dj["channel_names"], {"0": "CT", "1": "CT"})
        self.assertEqual(dj["labels"], {"background": 0, "tumor": 1})
        self.assertIsNone(dj.get("regions_class_order"))

        # Check splits copied to preprocessed folder
        self.assertTrue(isfile(join(self.pp_dir, "Dataset229_AutoPETIII", "splits_final.json")))

        # Verify dataset integrity using nnU-Net's built-in verifier
        verify_dataset_integrity(out_dataset_dir, num_processes=1)

    def test_convert_preorganized_imagesTr_labelsTr(self):
        """Test conversion on pre-organized Grand Challenge format with imagesTr and labelsTr"""
        input_dir = join(self.temp_dir.name, "gc_autopet")
        imagestr_in = join(input_dir, "imagesTr")
        labelstr_in = join(input_dir, "labelsTr")

        case_id = "psma_patient001_study01"
        self._create_dummy_nifti(join(imagestr_in, f"{case_id}_0000.nii.gz"))
        self._create_dummy_nifti(join(imagestr_in, f"{case_id}_0001.nii.gz"))
        self._create_dummy_nifti(join(labelstr_in, f"{case_id}.nii.gz"), is_seg=True)

        with patch("nnunetv2.dataset_conversion.Dataset229_AutoPETIII.nnUNet_raw", self.raw_dir):
            with patch("nnunetv2.dataset_conversion.Dataset229_AutoPETIII.nnUNet_preprocessed", self.pp_dir):
                convert_autopet3(input_dir, nnunet_dataset_id=229)

        out_dataset_dir = join(self.raw_dir, "Dataset229_AutoPETIII")
        self.assertTrue(isfile(join(out_dataset_dir, "imagesTr", f"{case_id}_0000.nii.gz")))
        self.assertTrue(isfile(join(out_dataset_dir, "imagesTr", f"{case_id}_0001.nii.gz")))
        self.assertTrue(isfile(join(out_dataset_dir, "labelsTr", f"{case_id}.nii.gz")))

        verify_dataset_integrity(out_dataset_dir, num_processes=1)

    def test_convert_tracer_grouped_subdirectories(self):
        """Test conversion when FDG and PSMA folders are in subdirectories"""
        input_dir = join(self.temp_dir.name, "tracer_grouped")

        fdg_case = join(input_dir, "FDG-PET-CT-Lesions", "PETCT_10", "study_01")
        self._create_dummy_nifti(join(fdg_case, "CTres.nii.gz"))
        self._create_dummy_nifti(join(fdg_case, "SUV.nii.gz"))
        self._create_dummy_nifti(join(fdg_case, "SEG.nii.gz"), is_seg=True)

        psma_case = join(input_dir, "PSMA-PET-CT-Lesions", "psma_20", "study_01")
        self._create_dummy_nifti(join(psma_case, "CTres.nii.gz"))
        self._create_dummy_nifti(join(psma_case, "SUV.nii.gz"))
        self._create_dummy_nifti(join(psma_case, "SEG.nii.gz"), is_seg=True)

        with patch("nnunetv2.dataset_conversion.Dataset229_AutoPETIII.nnUNet_raw", self.raw_dir):
            with patch("nnunetv2.dataset_conversion.Dataset229_AutoPETIII.nnUNet_preprocessed", self.pp_dir):
                convert_autopet3(input_dir, nnunet_dataset_id=229)

        out_dataset_dir = join(self.raw_dir, "Dataset229_AutoPETIII")
        with open(join(out_dataset_dir, "dataset.json"), "r") as f:
            dj = json.load(f)
        self.assertEqual(dj["numTraining"], 2)
        verify_dataset_integrity(out_dataset_dir, num_processes=1)


if __name__ == '__main__':
    unittest.main()
