import math
import torch as th
import torch.nn as nn
from torch import Tensor
from typing import List
from models.components.quantizable_module import QuantizableModule
from utils.misc import POSSIBLE_Q_STEP_SYN_NN

class SynthesisLayer(nn.Module):
    def __init__(
        self,
        input_ft: int,
        output_ft: int,
        kernel_size: int,
        non_linearity: nn.Module = nn.Identity()
    ):
        """Instantiate a synthesis layer.

        Args:
            input_ft (int): Input feature
            output_ft (int): Output feature
            kernel_size (int): Kernel size
            non_linearity (nn.Module): Non linear function applied at the very end
                of the forward. Defaults to nn.Identity()
        """
        super().__init__()

        self.pad = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        self.conv_layer = nn.Conv2d(
            input_ft,
            output_ft,
            kernel_size
        )

        self.non_linearity = non_linearity

        # More stable if initialized as a zero-bias layer with smaller variance
        # for the weights.
        with th.no_grad():
            # nn.init.kaiming_uniform_(self.conv_layer.weight.data, a=math.sqrt(5))
            self.conv_layer.weight.data = self.conv_layer.weight.data / output_ft ** 2
            self.conv_layer.bias.data = self.conv_layer.bias.data * 0.

    def forward(self, x: Tensor) -> Tensor:
        return self.non_linearity(self.conv_layer(self.pad(x)))

class SynthesisResidualLayer(nn.Module):
    def __init__(
        self,
        input_ft: int,
        output_ft: int,
        kernel_size: int,
        non_linearity: nn.Module = nn.Identity()
    ):
        """Instantiate a synthesis residual layer.

        Args:
            input_ft (int): Input feature
            output_ft (int): Output feature
            kernel_size (int): Kernel size
            non_linearity (nn.Module): Non linear function applied at the very end
                of the forward. Defaults to nn.Identity()
        """
        super().__init__()

        assert input_ft == output_ft,\
            f'Residual layer in/out dim must match. Input = {input_ft}, output = {output_ft}'

        self.pad = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        self.conv_layer = nn.Conv2d(
            input_ft,
            output_ft,
            kernel_size
        )

        self.non_linearity = non_linearity

        # More stable if a residual is initialized with all-zero parameters.
        # This avoids increasing the output dynamic at the initialization
        with th.no_grad():
            # nn.init.kaiming_uniform_(self.conv_layer.weight.data, a=math.sqrt(5))
            # self.conv_layer.weight.data = self.conv_layer.weight.data / output_ft ** 2
            self.conv_layer.weight.data = self.conv_layer.weight.data * 0.
            self.conv_layer.bias.data = self.conv_layer.bias.data * 0.

    def forward(self, x: Tensor) -> Tensor:
        return self.non_linearity(self.conv_layer(self.pad(x)) + x)
# =================== Conv layer layers for the Synthesis =================== #

class Synthesis(QuantizableModule):
    possible_non_linearity = {
        'none': nn.Identity,
        'relu': nn.ReLU,
        'leakyrelu': nn.LeakyReLU,
        'gelu': nn.GELU
    }
    possible_mode = {
        'linear': SynthesisLayer,
        'residual': SynthesisResidualLayer,
    }

    def __init__(self, input_ft: int, layers_dim: List[str]):
        super().__init__(possible_q_steps=POSSIBLE_Q_STEP_SYN_NN)
        layers_list = nn.ModuleList()

        # Construct the hidden layer(s)
        for layers in layers_dim:
            out_ft, k_size, mode, non_linearity = layers.split('-')
            out_ft = int(out_ft)
            k_size = int(k_size)

            # Check that mode and non linearity is correct
            assert mode in Synthesis.possible_mode,\
                f'Unknown mode. Found {mode}. Should be in {Synthesis.possible_mode.keys()}'

            assert non_linearity in Synthesis.possible_non_linearity,\
                f'Unknown non linearity. Found {non_linearity}. '\
                f'Should be in {Synthesis.possible_non_linearity.keys()}'

            # Instantiate them
            layers_list.append(
                Synthesis.possible_mode[mode](
                    input_ft,
                    out_ft,
                    k_size,
                    non_linearity=Synthesis.possible_non_linearity[non_linearity](),
                )
            )

            input_ft = out_ft

        self.layers = nn.Sequential(*layers_list)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)