import unittest

import torch

from commode_utils.losses import SequenceCrossEntropyLoss


class TestSequenceCrossEntropyLoss(unittest.TestCase):
    __pad_idx = 0

    def setUp(self):
        self.__loss = SequenceCrossEntropyLoss(self.__pad_idx, None)

    def test_no_padding(self):
        seq_len, batch_size, vocab_size = 3, 1, 5
        logits = torch.ones((seq_len, batch_size, vocab_size))
        targets = torch.ones((seq_len, batch_size), dtype=torch.long)
        expected_loss = -torch.log(torch.tensor(1.0 / vocab_size))
        actual_loss = self.__loss(logits, targets)[0]
        torch.testing.assert_close(actual_loss, expected_loss)

    def test_padded_sequences(self):
        seq_len, batch_size, vocab_size = 3, 1, 5
        sequence_lengths = list(range(1, seq_len + 1))

        logits = torch.ones((seq_len, batch_size, vocab_size))
        targets = torch.ones((seq_len, batch_size), dtype=torch.long)
        for i, start_pad in enumerate(sequence_lengths):
            targets[i, start_pad:] = self.__pad_idx

        expected_losses = -torch.log(torch.tensor(1.0 / vocab_size)).repeat(batch_size)
        actual_loss = self.__loss(logits, targets)
        torch.testing.assert_close(actual_loss, expected_losses)
