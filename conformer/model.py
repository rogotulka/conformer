import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

class Conformer(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        outputs = torch.randn(50, 3,  10).log_softmax(2).detach().requires_grad_().to("cuda")
        encoder_output_lengths = torch.LongTensor([9,9,9]).to("cuda")
        return outputs, encoder_output_lengths