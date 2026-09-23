import os
import unittest
from unittest.mock import patch
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class _CompileTrainerStub(nnUNetTrainer):
    def __init__(self, device: torch.device):
        self.device = device
        self.log_messages: list[str] = []

    def print_to_log_file(self, *args, **kwargs):
        self.log_messages.append(" ".join(str(a) for a in args))


class TestCompileDecision(unittest.TestCase):
    def test_mps_device_disables_compile(self):
        trainer = _CompileTrainerStub(torch.device("mps"))
        with patch.dict(os.environ, {"nnUNet_compile": "true"}, clear=True):
            self.assertFalse(trainer._do_i_compile())
            self.assertTrue(any("unsupported mps" in m for m in trainer.log_messages))

    def test_cpu_device_disables_compile(self):
        trainer = _CompileTrainerStub(torch.device("cpu"))
        with patch.dict(os.environ, {"nnUNet_compile": "true"}, clear=True):
            self.assertFalse(trainer._do_i_compile())
            self.assertTrue(any("device is CPU" in m for m in trainer.log_messages))

    def test_windows_without_triton_disables_compile(self):
        trainer = _CompileTrainerStub(torch.device("cuda:0"))
        with patch("os.name", "nt"), patch.dict("sys.modules", {"triton": None}):
            # When nnUNet_compile is unset
            with patch.dict(os.environ, {}, clear=True):
                self.assertFalse(trainer._do_i_compile())

            # When nnUNet_compile is true, logs guidance
            with patch.dict(os.environ, {"nnUNet_compile": "true"}, clear=True):
                self.assertFalse(trainer._do_i_compile())
                self.assertTrue(
                    any("triton is not installed on Windows" in m for m in trainer.log_messages)
                )

    def test_windows_with_triton_enables_compile(self):
        trainer = _CompileTrainerStub(torch.device("cuda:0"))
        # Mock triton module existence
        dummy_triton = object()
        with patch("os.name", "nt"), patch.dict("sys.modules", {"triton": dummy_triton}):
            # Default (unset) should enable compile
            with patch.dict(os.environ, {}, clear=True):
                self.assertTrue(trainer._do_i_compile())

            # Explicit true
            with patch.dict(os.environ, {"nnUNet_compile": "1"}, clear=True):
                self.assertTrue(trainer._do_i_compile())

            # Explicit false
            with patch.dict(os.environ, {"nnUNet_compile": "false"}, clear=True):
                self.assertFalse(trainer._do_i_compile())

    def test_posix_compile_behavior(self):
        trainer = _CompileTrainerStub(torch.device("cuda:0"))
        with patch("os.name", "posix"):
            # Default enables compile
            with patch.dict(os.environ, {}, clear=True):
                self.assertTrue(trainer._do_i_compile())

            # Explicit disable
            with patch.dict(os.environ, {"nnUNet_compile": "0"}, clear=True):
                self.assertFalse(trainer._do_i_compile())


if __name__ == "__main__":
    unittest.main()
