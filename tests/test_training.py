from unittest import TestCase

import torch

from commode_utils.training import segment_sizes_to_slices, cut_into_segments


class TestTrainingUtils(TestCase):
    def test_segment_sizes_to_slices(self):
        sizes = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        true_slices = [slice(0, 1), slice(1, 3), slice(3, 6), slice(6, 10), slice(10, 15)]
        calculated_slices = segment_sizes_to_slices(sizes)

        self.assertListEqual(true_slices, calculated_slices)

    def test_segment_sizes_to_slices_short(self):
        sizes = torch.tensor([1], dtype=torch.long)
        true_slices = [slice(0, 1)]
        calculated_slices = segment_sizes_to_slices(sizes)

        self.assertListEqual(true_slices, calculated_slices)

    def test_cut_into_segments(self):
        data = torch.arange(18).reshape(-1, 3) + 1
        sizes = torch.tensor([1, 2, 3], dtype=torch.long)

        true_cut_data = torch.tensor(
            [
                [[1, 2, 3], [0, 0, 0], [0, 0, 0]],
                [[4, 5, 6], [7, 8, 9], [0, 0, 0]],
                [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
            ]
        )
        true_mask = torch.tensor([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
        calculated_cut_data, calculated_mask = cut_into_segments(data, sizes, 1)

        torch.testing.assert_allclose(true_cut_data, calculated_cut_data)
        torch.testing.assert_allclose(true_mask, calculated_mask)
