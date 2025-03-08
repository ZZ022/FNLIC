import torch as th
import numpy as np
from typing import List, OrderedDict, Tuple
from dataclasses import dataclass, fields
from torch import Tensor, nn
from models.components.quantizer import NoiseQuantizer, STEQuantizer
from models.components.synthesis import Synthesis
from models.components.upsampling import Upsampling
from models.components.arm import Arm
from utils.misc import DescriptorOverfitter
from utils.distribution import weak_colorar_rate, get_latent_rate
from utils.helpers import get_flat_latent_and_context, space2depth, depth2space

@dataclass
class OverfitterParameter():
    """Dataclass to store the parameters of Overfitter"""

    img_shape: Tuple[int, int]

    # ----- Architecture options
    layers_synthesis: List[str]         # Synthesis architecture (e.g. '12-1-linear-relu', '12-1-residual-relu', '3-1-linear-relu', '3-3-residual-none')
    layers_arm: List[int]               # Output dim. of each hidden layer for the ARM (Empty for linear MLP)
    n_latents:int         # Number of latents for each resolution.
    upsampling_kernel_size: int = 8     # Kernel size for the upsampler ≥8. if set to zero the kernel is not optimised, the bicubic upsampler is used
    img_bitdepth: int = 8               # Bitdepth of the input image
    latent_bitdepth: int = 6            # Bitdepth of the latent variables
    freq_precision: int = 15            # Precision of the frequency
    latent_freq_precision: int = 12
    # ==================== Not set by the init function ===================== #

    def pretty_string(self) -> str:
        """Return a pretty string formatting the data within the class"""
        ATTRIBUTE_WIDTH = 25
        VALUE_WIDTH = 80

        s = 'OverfitterParameter value:\n'
        s += '-------------------------------\n'
        for k in fields(self):
            s += f'{k.name:<{ATTRIBUTE_WIDTH}}: {str(getattr(self, k.name)):<{VALUE_WIDTH}}\n'
        s += '\n'
        return s

class OverFitter(nn.Module):

    def __init__(self, param:OverfitterParameter, alpha_init:float=1.) -> None:
        super().__init__()
        self.param = param

        # regarding image configuration
        self.img_shape = param.img_shape
        h,w = self.img_shape      
        self.img_size = np.prod(param.img_shape)*3
        self.bitdepth = param.img_bitdepth
        self.freq_precision = param.freq_precision

        # regarding the latent variables
        self.latent_bitdepth = param.latent_bitdepth
        self.latent_max_val = 2**(param.latent_bitdepth - 1) - 1
        self.n_latents = param.n_latents
        self.latents = nn.ParameterList()
        self.noise_quantizer = NoiseQuantizer()
        self.ste_quantizer = STEQuantizer()
        self.encoder_gains = th.ones(param.n_latents,) * 64
        padding = 1 << (self.n_latents - 1)
        h = (h//padding) * padding + padding * (h % padding > 0)
        w = (w//padding) * padding + padding * (w % padding > 0)
        for i in range(param.n_latents):
            self.latents.append(nn.Parameter(th.zeros(1, 1, h//(2**i), w//(2**i))))
        self.upsampling = Upsampling(param.upsampling_kernel_size)

        # regarding the ARM and the synthesis
        self.arm = Arm(2, param.layers_arm)
        non_zero_pixel_ctx_index = [1, 3]
        self.non_zero_pixel_ctx_index = th.tensor(non_zero_pixel_ctx_index)
        self.synthesis = Synthesis(param.n_latents, param.layers_synthesis)

        # regarding the scale of prior parameters
        self.coefficients = nn.ParameterList([nn.Parameter(alpha_init*th.ones(1,9,1,1).float()) for _ in range(4)])
        self.modules_to_send = [tmp.name for tmp in fields(DescriptorOverfitter)]

    def modify_prior(self, prior:th.Tensor):
        priors = th.chunk(space2depth(prior), 4, dim=1)
        priors = [prior * coeff for prior, coeff in zip(priors, self.coefficients)]
        return depth2space(th.cat(priors, dim=1))

    def get_quantized_latent(self, use_ste_quant: bool=False) -> List[Tensor]:
        """
        Args:
            use_ste_quant (bool, optional): True to use the straight-through estimator for
                quantization. Defaults to True.

        Returns:
            List[Tensor]: List of [1, C, H', W'] latent variable with H' and W' depending
                on the particular resolution of each latent.
        """
        scaled_latent = [
            cur_latent * self.encoder_gains[i] for i, cur_latent in enumerate(self.latents)
        ]

        if self.training:
            if use_ste_quant:
                sent_latent = [self.ste_quantizer(cur_latent) for cur_latent in scaled_latent]
            else:
                sent_latent = [self.noise_quantizer(cur_latent) for cur_latent in scaled_latent]
        else:
            sent_latent = [th.round(cur_latent) for cur_latent in scaled_latent]

        # Clamp latent if we need to write a bitstream
        sent_latent = [
            th.clamp(cur_latent, -self.latent_max_val, self.latent_max_val) for cur_latent in sent_latent
        ]

        return sent_latent

    def get_network_rate(self) -> DescriptorOverfitter:
        """Return the rate (in bits) associated to the parameters (weights and biases)
        of the different modules

        Returns:
            DescriptorOverfitter: The rate (in bits) associated with the weights and biases of each module
        """
        rate_per_module: DescriptorOverfitter = {
            module_name: {'weight': 0., 'bias': 0.} for module_name in self.modules_to_send
        }

        for module_name in self.modules_to_send:
            rate_per_module[module_name] = getattr(self, module_name).measure_laplace_rate()

        return rate_per_module

    def get_network_quantization_step(self) -> DescriptorOverfitter:
        """Return the quantization step associated to the parameters (weights and biases)
        of the different modules

        Returns:
            DescriptorOverfitter: The quantization step associated with the weights and biases of each module
        """
        q_step_per_module: DescriptorOverfitter = {
            module_name: {'weight': 0., 'bias': 0.} for module_name in self.modules_to_send
        }

        for module_name in self.modules_to_send:
            q_step_per_module[module_name] = getattr(self, module_name).get_q_step()

        return q_step_per_module

    def forward(self, 
                img_t:Tensor, 
                prior:Tensor,
                use_ste_quant:bool=False)->OrderedDict:
        prior = self.modify_prior(prior)
        latents = self.get_quantized_latent(use_ste_quant)
        latent_flat, context_flat = get_flat_latent_and_context(latents, 3, self.non_zero_pixel_ctx_index)
        latent_flat = latent_flat.unsqueeze(1)
        params = self.arm(context_flat)
        latent_rate  = get_latent_rate(latent_flat, params, self.latent_bitdepth, self.param.latent_freq_precision).sum()
        latent_prior = self.upsampling(latents)

        params = self.synthesis(latent_prior) + prior
        latent_bpd = latent_rate/self.img_size
        img_rates = weak_colorar_rate(params, img_t, self.bitdepth, self.freq_precision)
        img_bpd = img_rates.sum()/self.img_size
      
        return OrderedDict(
            latent_bpd=latent_bpd,
            img_bpd=img_bpd,
            loss = latent_bpd + img_bpd
        )
    
    @th.no_grad()
    def get_delta(self)->Tensor:
        latents = self.get_quantized_latent()
        synthesis_input = self.upsampling(latents)
        return self.synthesis(synthesis_input)

    def to_device(self, device:th.device)->None:
        self.to(device)
        self.non_zero_pixel_ctx_index = self.non_zero_pixel_ctx_index.to(device)
        self.encoder_gains = self.encoder_gains.to(device)

        for idx_layer, layer in enumerate(self.arm.mlp):
            if hasattr(layer, 'qw'):
                if layer.qw is not None:
                    self.arm.mlp[idx_layer].qw = layer.qw.to(device)

            if hasattr(layer, 'qb'):
                if layer.qb is not None:
                    self.arm.mlp[idx_layer].qb = layer.qb.to(device)
                    
    @th.no_grad()
    def inference_for_decode(self, x:Tensor, prior:Tensor, max_latent_v:int)->None:
        symbols = th.arange(-max_latent_v, max_latent_v+1, 1.0).to(x.device)
        symbols_2d = th.cartesian_prod(symbols, symbols).float()
        latent_params = self.arm(symbols_2d)
        latents = self.get_quantized_latent()
        latent_prior = self.upsampling(latents)
        params = self.synthesis(latent_prior) + self.modify_prior(prior)
    
    def save(self, path:str)->None:
        self.eval()
        with th.no_grad():
            quant_latents_list = self.get_quantized_latent()
        quant_latents_uint8 = [(latent + self.latent_max_val).to(th.uint8).cpu() for latent in quant_latents_list]
        save_dict = {
            'quant_latents_uint8': quant_latents_uint8,
            'arm': self.arm.cpu(),
            'upsampling': self.upsampling.cpu(),
            'synthesis': self.synthesis.cpu(),
            'coefficients': self.coefficients.cpu(),
        }
        th.save(save_dict, path)
    
    def load(self, path:str)->None:
        load_dict = th.load(path)
        self.arm = load_dict['arm']
        self.upsampling = load_dict['upsampling']
        self.synthesis = load_dict['synthesis']
        self.coefficients = load_dict['coefficients']

        quant_latents_uint8 = load_dict['quant_latents_uint8']
        self.latents = nn.ParameterList([nn.Parameter((quant_latents_uint8[i].float() - self.latent_max_val)/self.encoder_gains[i]) for i in range(self.n_latents)])