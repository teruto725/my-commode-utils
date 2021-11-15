from typing import Dict, Optional, List

import torch
from sacrebleu import CHRF
from torchmetrics import Metric

from commode_utils.common import decode


class ChrF(Metric):
    def __init__(self, id2label: Dict[int, str], ignore_indexes: Optional[List[int]] = None, **kwargs):
        super().__init__(**kwargs)
        self.__id2label = id2label
        if ignore_indexes is None:
            ignore_indexes = []
        self.__ignore_indexes = ignore_indexes
        self.__chrf = CHRF()

        # Metric states
        self.add_state("chrf", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predicted: torch.Tensor, target: torch.Tensor):
        """Calculated ChrF metric on predicted tensor w.r.t. target tensor.

        :param predicted: [pred seq len; batch size] -- tensor with predicted tokens
        :param target: [target seq len; batch size] -- tensor with ground truth tokens
        :return:
        """
        batch_size = target.shape[1]
        if predicted.shape[1] != batch_size:
            raise ValueError(f"Wrong batch size for prediction (expected: {batch_size}, actual: {predicted.shape[1]})")

        for batch_idx in range(batch_size):
            target_seq = [token.item() for token in target[:, batch_idx]]
            predicted_seq = [token.item() for token in predicted[:, batch_idx]]

            target_str = " ".join(decode(target_seq, self.__id2label, self.__ignore_indexes))
            predicted_str = " ".join(decode(predicted_seq, self.__id2label, self.__ignore_indexes))

            if target_str == "":
                # Empty target string mean that the original string encoded only with <UNK> token
                continue

            self.chrf += self.__chrf.sentence_score(predicted_str, [target_str]).score
            self.count += 1

    def compute(self) -> torch.Tensor:
        return self.chrf / self.count
