from typing import Tuple

import torch
from torch import nn

from commode_utils.modules.base_decoder_step import BaseDecoderStep
from commode_utils.training import cut_into_segments

import numpy as np

class Decoder(nn.Module):

    _negative_value = -1e9

    def __init__(self, decoder_step: BaseDecoderStep, output_size: int, sos_token: int, teacher_forcing: float = 0.0):
        super().__init__()
        self._decoder_step = decoder_step
        self._teacher_forcing = teacher_forcing
        self._out_size = output_size #最終的な単語ベクトル
        self._sos_token = sos_token

    def forward(
        self,
        encoder_output: torch.Tensor,
        infer_vec: torch.Tensor,
        segment_sizes: torch.LongTensor,
        output_size: int,
        target_sequence: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate output sequence based on encoder output

        :param encoder_output: [n sequences; encoder size] -- stacked encoder output
        :param segment_sizes: [batch size] -- size of each segment in encoder output
        :param output_size: size of output sequence
        :param target_sequence: [batch size; max seq len] -- if passed can be used for teacher forcing
        :return:
            [output size; batch size; vocab size] -- sequence with logits for each position
            [output size; batch size; encoder seq length] -- sequence with attention weights for each position
        """
        batch_size = segment_sizes.shape[0]
        # encoder output -- [batch size; max context len; units]
        # attention mask -- [batch size; max context len]
        #print("encoder output: ", encoder_output.shape)
        #print("infer_vec:", infer_vec.shape)
        #print("segment_sizes",np.shape( segment_sizes))
        
        batched_encoder_output, attention_mask = cut_into_segments(encoder_output, infer_vec, segment_sizes, self._negative_value)
        #print("batched_encoder_output: ", batched_encoder_output.shape)
        #print("attention_mask: ", attention_mask.shape)
        decoder_state = self._decoder_step.get_initial_state(batched_encoder_output, attention_mask)
        #print("decoder_state: ", decoder_state)
        # [output size; batch size; vocab size]
        output = batched_encoder_output.new_zeros((output_size, batch_size, self._out_size))
        #print("output: ", output.shape)
        output[0, :, self._sos_token] = 1

        # [output size; batch size; encoder seq size]
        attentions = batched_encoder_output.new_zeros((output_size, batch_size, attention_mask.shape[1]))

        # [batch size]
        current_input = batched_encoder_output.new_full((batch_size,), self._sos_token, dtype=torch.long)
        for step in range(1, output_size):
            current_output, current_attention, decoder_state = self._decoder_step(
                current_input, batched_encoder_output, attention_mask, decoder_state
            ) # 学習部分
            output[step] = current_output
            attentions[step] = current_attention
            if self.training and target_sequence is not None and torch.rand(1) <= self._teacher_forcing:
                current_input = target_sequence[step]
            else:
                current_input = output[step].argmax(dim=-1)
        #print("output_final: ",np.shape( output))
        #print("attention:" ,np.shape( attentions ))
        return output, attentions
