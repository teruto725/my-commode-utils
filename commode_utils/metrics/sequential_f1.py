from typing import Optional, List

import torch
from torchmetrics import Metric

from commode_utils.metrics import ClassificationMetrics


class SequentialF1Score(Metric):
    def __init__(
        self,
        mask_after_pad: bool = True,
        pad_idx: Optional[int] = None,
        ignore_idx: Optional[List[int]] = None,
        **kwargs,
    ):
        """Initialize the metric for computing f1-score for sequential data.
        This metric is used in many works about code summarization.

        :param mask_after_pad: if True then ignore tokens after first pad index
        :param pad_idx: index of PAD token, required for masking after pad
        :param ignore_idx: indexes that are ignored during computing occurrences
        """
        if mask_after_pad and pad_idx is None:
            raise ValueError("Pass padding index in order to find it first occurrence.")
        super().__init__(**kwargs)
        self._mask_after_pad = mask_after_pad
        self._pad_idx = pad_idx
        self._ignore_idx = ignore_idx if ignore_idx is not None else []

        # metric states
        self.add_state("true_positive", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_positive", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_negative", default=torch.tensor(0), dist_reduce_fx="sum")

    def _get_after_pad_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """For each sequence, create mask with all tokens after first PAD

        :param tokens: [seq len; batch size] tensor with tokens to be replaced
        :return: mask with the same shape as tokens
        """
        mask_max_value, mask_max_indices = torch.max(tokens == self._pad_idx, dim=0)
        # if no pad token use len+1 position
        mask_max_indices[~mask_max_value] = tokens.shape[0]
        mask = torch.arange(tokens.shape[0], device=tokens.device).view(-1, 1) >= mask_max_indices
        return mask

    def update(self, predicted: torch.Tensor, target: torch.Tensor):
        """Calculated token occurrence statistic in predicted tensor w.r.t. target tensor.

        :param predicted: [pred seq len; batch size] -- tensor with predicted tokens
        :param target: [target seq len; batch size] -- tensor with ground truth tokens
        :return:
        """
        batch_size = target.shape[1]
        if predicted.shape[1] != batch_size:
            raise ValueError(f"Wrong batch size for prediction (expected: {batch_size}, actual: {predicted.shape[1]})")
        if self._mask_after_pad and self._pad_idx is not None:
            after_pad_mask = self._get_after_pad_mask(predicted)
            predicted[after_pad_mask] = self._pad_idx
            if self._pad_idx not in self._ignore_idx:
                self._ignore_idx.append(self._pad_idx)

        for batch_idx in range(batch_size):
            target_seq = [token for token in target[:, batch_idx] if token not in self._ignore_idx]
            predicted_seq = [token for token in predicted[:, batch_idx] if token not in self._ignore_idx]

            for predicted_token in predicted_seq:
                if predicted_token in target_seq:
                    self.true_positive += 1
                else:
                    self.false_positive += 1
            for target_token in target_seq:
                if target_token not in predicted_seq:
                    self.false_negative += 1

    def compute(self) -> ClassificationMetrics:
        """Calculate precision, recall, and F1-score based on stored token occurrences

        :return: calculated metrics aggregated in data class
        """
        precision = self.true_positive
        if self.true_positive + self.false_positive > 0:
            precision = self.true_positive / (self.true_positive + self.false_positive)
        recall = self.true_positive
        if self.true_positive + self.false_negative > 0:
            recall = self.true_positive / (self.true_positive + self.false_negative)
        f1_score = 2 * precision * recall
        if precision + recall > 0:
            f1_score /= precision + recall
        return ClassificationMetrics(f1_score=f1_score, precision=precision, recall=recall)
