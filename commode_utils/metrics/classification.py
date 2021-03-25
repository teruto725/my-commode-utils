from dataclasses import dataclass

from torch import Tensor


@dataclass
class ClassificationMetrics:
    f1_score: Tensor
    precision: Tensor
    recall: Tensor
