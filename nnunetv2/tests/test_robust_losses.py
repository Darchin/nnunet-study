import unittest
import torch
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_topk_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, SoftDiceLoss


class TestRobustLosses(unittest.TestCase):
    def setUp(self):
        self.devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            self.devices.append(torch.device("cuda:0"))

    def test_robust_ce_handles_negative_and_oob_targets(self):
        for device in self.devices:
            num_classes = 5
            logits = torch.randn(2, num_classes, 8, 8, 8, device=device, requires_grad=True)
            target = torch.randint(0, num_classes, (2, 1, 8, 8, 8), device=device)
            # Inject out-of-bound labels
            target[0, 0, 0, 0, 0] = -1
            target[0, 0, 0, 0, 1] = -100
            target[0, 0, 0, 0, 2] = 5
            target[0, 0, 0, 0, 3] = 99

            loss_fn = RobustCrossEntropyLoss().to(device)
            loss = loss_fn(logits, target)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertIsNotNone(logits.grad)

    def test_robust_ce_respects_explicit_ignore_index(self):
        for device in self.devices:
            num_classes = 5
            logits = torch.randn(2, num_classes, 8, 8, 8, device=device)
            target = torch.randint(0, num_classes, (2, 1, 8, 8, 8), device=device)
            target[0, 0, 0, 0, 0] = -1  # padding artifact
            target[0, 0, 0, 0, 1] = 3   # designated ignore label

            loss_fn = RobustCrossEntropyLoss(ignore_index=3).to(device)
            loss = loss_fn(logits, target)
            self.assertTrue(torch.isfinite(loss))

    def test_topk_handles_negative_and_oob_targets(self):
        for device in self.devices:
            num_classes = 5
            logits = torch.randn(2, num_classes, 8, 8, 8, device=device)
            target = torch.randint(0, num_classes, (2, 1, 8, 8, 8), device=device)
            target[0, 0, 0, 0, 0] = -1
            target[0, 0, 0, 0, 1] = 10

            loss_fn = TopKLoss(k=20).to(device)
            loss = loss_fn(logits, target)
            self.assertTrue(torch.isfinite(loss))

    def test_dc_and_ce_handles_negative_labels_without_ignore_label(self):
        for device in self.devices:
            num_classes = 5
            logits = torch.randn(2, num_classes, 8, 8, 8, device=device, requires_grad=True)
            target = torch.randint(0, num_classes, (2, 1, 8, 8, 8), device=device)
            target[0, 0, 0, 0, 0] = -1
            target[0, 0, 0, 0, 1] = -1

            loss_fn = DC_and_CE_loss(
                {'batch_dice': False, 'smooth': 1e-5, 'do_bg': False, 'ddp': False},
                {},
                dice_class=MemoryEfficientSoftDiceLoss
            ).to(device)
            loss = loss_fn(logits, target)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertIsNotNone(logits.grad)

    def test_dc_and_ce_handles_negative_labels_with_ignore_label(self):
        for device in self.devices:
            num_classes = 5
            logits = torch.randn(2, num_classes, 8, 8, 8, device=device, requires_grad=True)
            target = torch.randint(0, num_classes, (2, 1, 8, 8, 8), device=device)
            target[0, 0, 0, 0, 0] = -1  # padding artifact
            target[0, 0, 0, 0, 1] = 4   # ignore label

            loss_fn = DC_and_CE_loss(
                {'batch_dice': False, 'smooth': 1e-5, 'do_bg': False, 'ddp': False},
                {},
                ignore_label=4,
                dice_class=MemoryEfficientSoftDiceLoss
            ).to(device)
            loss = loss_fn(logits, target)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertIsNotNone(logits.grad)


if __name__ == '__main__':
    unittest.main()
