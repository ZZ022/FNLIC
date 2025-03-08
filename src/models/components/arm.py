import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List
from models.components.quantizable_module import QuantizableModule
from utils.misc import ARMINT, POSSIBLE_Q_STEP_ARM_NN

# ===================== Linear (MLP) layers for the ARM ===================== #
class CustomLinear(nn.Module):
    def __init__(self, in_ft: int, out_ft: int, scale: int = 0):
        super().__init__()
        self.weight = nn.Parameter(th.randn(out_ft, in_ft, requires_grad=True) / out_ft ** 2)
        self.bias = nn.Parameter(th.zeros((out_ft), requires_grad=True))
        self.scale = scale
        self.armint = ARMINT
        if self.armint:
            self.qw = th.zeros_like(self.weight).to(th.int32)
            self.qb = th.zeros_like(self.bias).to(th.int32)
        else:
            self.qw = th.zeros_like(self.weight).to(th.int32).to(th.float)
            self.qb = th.zeros_like(self.bias).to(th.int32).to(th.float)

    def forward(self, x: Tensor) -> Tensor:
        if self.scale == 0:
            return F.linear(x, self.weight, bias=self.bias)
        if self.armint:
            return (F.linear(x, self.qw, bias=self.qb) + self.scale//2)//self.scale
        else:
            return th.floor((F.linear(x, self.qw, bias=self.qb) + self.scale//2)/self.scale).to(th.int32).to(th.float)


class CustomLinearResBlock(nn.Module):
    def __init__(self, in_ft: int, out_ft: int, scale: int = 0):
        super().__init__()
        self.weight = nn.Parameter(th.zeros((out_ft, in_ft), requires_grad=True))
        self.bias = nn.Parameter(th.zeros((out_ft), requires_grad=True))
        self.scale = scale
        self.armint = ARMINT
        if self.armint:
            self.qw = th.zeros_like(self.weight).to(th.int32)
            self.qb = th.zeros_like(self.bias).to(th.int32)
        else:
            self.qw = th.zeros_like(self.weight).to(th.int32).to(th.float)
            self.qb = th.zeros_like(self.bias).to(th.int32).to(th.float)

    def forward(self, x: Tensor) -> Tensor:
        if self.scale == 0:
            return F.linear(x, self.weight, bias=self.bias) + x
        if self.armint:
            return (F.linear(x, self.qw, bias=self.qb) + x*self.scale + self.scale//2)//self.scale
        else:
            return th.floor((F.linear(x, self.qw, bias=self.qb) + x*self.scale + self.scale//2)/self.scale).to(th.int32).to(th.float)
# ===================== Linear (MLP) layers for the ARM ===================== #

class Arm(QuantizableModule):
    def __init__(self, input_ft: int, layers_dim: List[int], fpfm: int = 0):
        super().__init__(possible_q_steps=POSSIBLE_Q_STEP_ARM_NN)

        self.FPFM = fpfm # FIXED_POINT_FRACTIONAL_MULT # added for decode_network with torchscript.
        self.ARMINT = ARMINT
        self.qw = None
        self.qb = None
        # ======================== Construct the MLP ======================== #
        layers_list = nn.ModuleList()

        # Construct the hidden layer(s)
        for out_ft in layers_dim:
            if input_ft == out_ft:
                layers_list.append(CustomLinearResBlock(input_ft, out_ft, self.FPFM))
            else:
                layers_list.append(CustomLinear(input_ft, out_ft, self.FPFM))
            layers_list.append(nn.ReLU())
            input_ft = out_ft

        # Construct the output layer. It always has 2 outputs (mu and scale)
        layers_list.append(CustomLinear(input_ft, 2, self.FPFM))
        self.mlp = nn.Sequential(*layers_list)
        # ======================== Construct the MLP ======================== #

    def set_quant(self, fpfm: int = 0):
        # Non-zero fpfm implies we're in fixed point mode. weights and biases are integers.
        # ARMINT False => the integers are stored in floats.
        self.FPFM = fpfm
        self.cnt = 0
        # Convert from float to fixed point int.
        for l in self.mlp.children():
            if isinstance(l, CustomLinearResBlock) or isinstance(l, CustomLinear) \
                or (hasattr(l, "original_name") and (l.original_name == "CustomLinearResBlock" or l.original_name == "CustomLinear")):
                l.scale = self.FPFM
                # shadow fixed point weights and biases.
                if self.ARMINT:
                    l.qw = th.round(l.weight*l.scale).to(th.int32)
                    l.qb = th.round(l.bias*l.scale*l.scale).to(th.int32)
                else:
                    l.qw = th.round(l.weight*l.scale).to(th.int32).to(th.float)
                    l.qb = th.round(l.bias*l.scale*l.scale).to(th.int32).to(th.float)
    
    def forward(self, x: Tensor) -> Tensor:
        if self.FPFM == 0:
            # Pure float processing.
            return self.mlp(x)

        # integer mode.
        xint = x.clone().detach()
        if self.ARMINT:
            xint = xint.to(th.int32)*self.FPFM
        else:
            xint = (xint.to(th.int32)*self.FPFM).to(th.float)

        for l in self.mlp.children():
            xint = l(xint)

        # float the result.
        x = xint / self.FPFM
        self.cnt += 1
        return x