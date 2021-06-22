import torch
from omegaconf import DictConfig
from torch import nn

from commode_utils.modules import LocalAttention
from commode_utils.training import cut_into_segments


class Classifier(nn.Module):

    _negative_value = -1e9
    _activations = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}

    def _get_activation(self, activation_name: str) -> nn.Module:
        if activation_name in self._activations:
            return self._activations[activation_name]()
        raise KeyError(f"Activation {activation_name} is not supported")

    def __init__(self, config: DictConfig, output_size: int):
        super().__init__()

        self._attention = LocalAttention(config.classifier_size)
        hidden_layers = []
        for _ in range(config.classifier_layers):
            hidden_layers += [
                nn.Linear(config.hidden_size, config.hidden_size),
                self._get_activation(config.activation),
            ]
        hidden_layers.append(nn.Linear(config.hidden_size, output_size))
        self._layers = nn.Sequential(*hidden_layers)

    def forward(self, encoder_output: torch.Tensor, segment_sizes: torch.LongTensor) -> torch.Tensor:
        """Classify encoder output using local attention

        :param encoder_output: [n sequences; encoder size] -- stacked encoder output
        :param segment_sizes: [batch size] -- size of each segment
        :return: [batch size; logit size] -- predicted probabilities
        """
        # [batch size; max seq len; classifier size], [batch size; max seq len]
        batched_encoder_output, attention_mask = cut_into_segments(encoder_output, segment_sizes, self._negative_value)

        # [batch size; max context size; 1]
        attn_weights = self._attention(batched_encoder_output, attention_mask)

        # [batch size; classifier size]
        context = torch.bmm(attn_weights.transpose(1, 2), batched_encoder_output).squeeze(1)

        # [batch size; num classes]
        output = self._layers(context)
        output = torch.softmax(output, dim=-1)
        return output
