from typing import Optional

from torch import nn, Tensor


class SequenceCrossEntropyLoss(nn.CrossEntropyLoss):
    """Calculate cross entropy loss with ignoring PAD index.

    :param pad_idx: an optional index of padding in target sequence.
    :param reduction: rule to reduce computed loss over batch.
    """

    __known_reductions = {"mean": lambda loss: loss.mean(), "sum": lambda loss: loss.sum()}

    def __init__(self, pad_idx: Optional[int] = None, reduction: Optional[str] = "mean"):
        super().__init__()
        self.__pad_idx = pad_idx
        if reduction is not None and reduction not in self.__known_reductions.keys():
            raise ValueError(f"Unknown reduction: {reduction}")
        self.__reduction = reduction

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """Calculated loss for given logits and target.

        :param logits: tensor with logits with shape [seq len; batch size; vocab size]
        :param target: tensor with target classes with shape [seq len; batch size]
        :return:
        """
        # [batch size; vocab size; seq length]
        _logits = logits.permute(1, 2, 0)
        # [batch size; seq length]
        _labels = target.permute(1, 0)
        # [batch size; seq length]
        loss = nn.functional.cross_entropy(_logits, _labels, reduction="none")
        if self.__pad_idx is not None:
            # [batch size; seq length]
            mask = _labels != self.__pad_idx
            seq_len = mask.sum(-1)
            # [batch size]
            example_loss = (loss * mask).sum(-1) / seq_len
        else:
            # [batch size]
            example_loss = loss.mean(-1)

        if self.__reduction is None:
            return example_loss
        else:
            return self.__known_reductions[self.__reduction](example_loss)
