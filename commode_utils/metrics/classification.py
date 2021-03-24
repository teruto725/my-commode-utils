from dataclasses import dataclass


@dataclass
class ClassificationMetrics:
    f1_score: float
    precision: float
    recall: float
