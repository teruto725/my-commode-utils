from abc import abstractmethod
from typing import Union, Tuple

import torch
from torch import nn

DecoderState = Union[torch.Tensor, Tuple[torch.Tensor, ...]]


class BaseDecoderStep(nn.Module):
    @abstractmethod
    def get_initial_state(self, encoder_output: torch.Tensor, attention_mask: torch.Tensor) -> DecoderState:
        """Use this function to initialize decoder state, e.g. h and c for LSTM decoder.

        :param encoder_output: [batch size; max seq len; encoder size] -- encoder output
        :param attention_mask: [batch size; max seq len] -- mask with zeros on non pad elements
        :return: decoder state: single tensor of tuple of tensors
        """
        raise NotImplementedError()

    @abstractmethod
    def forward(
        self,
        input_token: torch.Tensor,
        encoder_output: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_state: DecoderState,
    ) -> Tuple[torch.Tensor, DecoderState]:
        """Perform one decoder step based on input token and encoder output.

        :param input_token: [batch size] -- tokens from previous step or from target sequence
        :param encoder_output: [batch size; max seq len; encoder size] -- encoder output
        :param attention_mask: [batch size; max seq len] -- mask with zeros on non pad elements
        :param decoder_state: decoder state from previous or initial step
        :return: [batch size; vocab size] -- logits and new decoder step
        """
        raise NotImplementedError()
