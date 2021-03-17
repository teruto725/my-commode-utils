import unittest

import torch

from commode_utils.metrics import SequentialF1Score, ClassificationMetrics


class TestMetrics(unittest.TestCase):
    def test_update(self):
        predicted = torch.tensor([[1, 1, 1, 0], [2, 2, 0, -1], [3, 3, -1, -1], [-1, -1, -1, -1]])
        target = torch.tensor([[2, 4, 1, 0], [4, 5, 2, 0], [1, 6, 3, 0], [5, -1, -1, -1]])
        ignore_idx = [-1, 0]

        metric = SequentialF1Score(mask_after_pad=False, ignore_idx=ignore_idx)
        _ = metric(predicted, target)

        self.assertEqual(3, metric.true_positive)
        self.assertEqual(4, metric.false_positive)
        self.assertEqual(7, metric.false_negative)

    def test_computing_metrics(self):
        metric = SequentialF1Score(mask_after_pad=False)
        metric.true_positive = 3
        metric.false_positive = 4
        metric.false_negative = 7

        classification_metrics: ClassificationMetrics = metric.compute()
        self.assertAlmostEqual(3 / 7, classification_metrics.precision)
        self.assertAlmostEqual(3 / 10, classification_metrics.recall)
        self.assertAlmostEqual(6 / 17, classification_metrics.f1_score)

    def test_computing_zero_metrics(self):
        metric = SequentialF1Score(mask_after_pad=False)
        metric.true_positive = 0
        metric.false_positive = 0
        metric.false_negative = 0

        classification_metrics: ClassificationMetrics = metric.compute()
        self.assertAlmostEqual(0, classification_metrics.precision)
        self.assertAlmostEqual(0, classification_metrics.recall)
        self.assertAlmostEqual(0, classification_metrics.f1_score)

    def test_update_equal_tensors(self):
        predicted = torch.tensor([1, 2, 3, 4, 5, 0, -1]).view(-1, 1)
        target = torch.tensor([1, 2, 3, 4, 5, 0, -1]).view(-1, 1)
        ignore_idx = [-1, 0]

        metric = SequentialF1Score(mask_after_pad=False, ignore_idx=ignore_idx)
        _ = metric(predicted, target)

        self.assertEqual(metric.true_positive, 5)
        self.assertEqual(metric.false_positive, 0)
        self.assertEqual(metric.false_negative, 0)

    def test_masking(self):
        tokens = torch.tensor([[1, 1, 1, 1], [1, 1, 1, -1], [1, 1, -1, 2], [1, -1, 2, 2]])
        true_mask = torch.tensor(
            [
                [False, False, False, False],
                [False, False, False, True],
                [False, False, True, True],
                [False, True, True, True],
            ],
            dtype=torch.bool,
        )

        metric = SequentialF1Score(mask_after_pad=True, pad_idx=-1)
        pred_mask = metric._get_after_pad_mask(tokens)

        torch.testing.assert_allclose(pred_mask, true_mask)

    def test_update_with_masking(self):
        target = torch.tensor([1, 2, 3, 6, 7, 8, 0, 0, 0]).view(-1, 1)
        predicted = torch.tensor([1, 2, 3, 4, 5, 0, 6, 0, 8]).view(-1, 1)

        metric = SequentialF1Score(mask_after_pad=True, pad_idx=0)
        _ = metric(predicted, target)

        self.assertEqual(metric.true_positive, 3)
        self.assertEqual(metric.false_positive, 2)
        self.assertEqual(metric.false_negative, 3)

    def test_update_with_masking_long_sequence(self):
        target = torch.tensor([1, 2, 3, 6, 7, 8, 0, 0, 0]).view(-1, 1)
        predicted = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).view(-1, 1)

        metric = SequentialF1Score(mask_after_pad=True, pad_idx=0)
        _ = metric(predicted, target)

        self.assertEqual(metric.true_positive, 6)
        self.assertEqual(metric.false_positive, 2)
        self.assertEqual(metric.false_negative, 0)
