from typing import Optional

import torch


def sequence_cross_entropy_loss(
    logits: torch.Tensor, labels: torch.Tensor, pad_idx: Optional[int] = None, reduction: Optional[str] = "mean"
) -> torch.Tensor:
    """Calculate cross entropy for predicted sequence with ignoring PAD index
    :param logits: [seq length; batch size; vocab size]
    :param labels: [seq length; batch size]
    :param pad_idx: index of pad token
    :param reduction: reduction rule for batch, None mean no reduction
    :return: calculated loss
    """
    # [batch size; vocab size; seq length]
    _logits = logits.permute(1, 2, 0)
    # [batch size; seq length]
    _labels = labels.permute(1, 0)
    # [batch size; seq length]
    loss = torch.nn.functional.cross_entropy(_logits, _labels, reduction="none")
    if pad_idx is not None:
        # [batch size; seq length]
        mask = _labels != pad_idx
        # [batch size; seq length]
        loss = loss * mask
    # [1]
    if reduction is None:
        return loss
    elif reduction == "mean":
        batch_size = labels.shape[-1]
        return loss.sum() / batch_size
    else:
        raise NotImplementedError(f"Unknown reduction rule: {reduction}")
