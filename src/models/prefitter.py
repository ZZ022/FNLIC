import torchac
import torch as th
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass, fields
from typing import List, OrderedDict
from utils.helpers import space2depth, depth2space
from utils.distribution import *

class ResidualBlock(nn.Module):
    def __init__(self, width:int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width, width, 3, 1, 1)
    
    def forward(self, x:Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out

@dataclass
class PrefitterParameter():
    img_bitdepth: int = 8               # Bitdepth of the input image
    prior_arm_width: int = 0
    prior_arm_depth: int = 0
    freq_precision: int = 16

    def pretty_string(self) -> str:
        """Return a pretty string formatting the data within the class"""
        ATTRIBUTE_WIDTH = 25
        VALUE_WIDTH = 80

        s = 'PrefitterParameter value:\n'
        s += '-------------------------------\n'
        for k in fields(self):
            s += f'{k.name:<{ATTRIBUTE_WIDTH}}: {str(getattr(self, k.name)):<{VALUE_WIDTH}}\n'
        s += '\n'
        return s

class Prefitter(nn.Module):

    def __init__(self, param:PrefitterParameter)->None:
        super().__init__()
        self.param = param
        self.bit_depth = param.img_bitdepth

        self.param0 = nn.Parameter(th.zeros(1, 9, 1, 1))
        self.prior_ar1 = self.ar_model(3, param.prior_arm_width, param.prior_arm_depth)
        self.prior_ar2 = self.ar_model(6, param.prior_arm_width, param.prior_arm_depth)
        self.prior_ar3 = self.ar_model(9, param.prior_arm_width, param.prior_arm_depth)
        

    def ar_model(self, in_ft:int, width:int, depth:int)->nn.Module:
        if depth > 1:
            prior_ar_layers = [nn.Conv2d(in_ft, width, 3, 1, 1), nn.ReLU(True)]
            for _ in range(depth-2):
                prior_ar_layers+=[ResidualBlock(width)]
            prior_ar_layers+=[nn.Conv2d(width, 9, 3, 1, 1)]
            prior_ar = nn.Sequential(*prior_ar_layers)
        else:
            prior_ar = nn.Conv2d(in_ft, 9, 3, 1, 1)
        return prior_ar
    
    def forward(self, x:Tensor)->OrderedDict:
        b, _, h, w = x.size()
        h = h//2
        w = w//2

        x_reshape = space2depth(x)
        
        params1 = self.param0.repeat(b,1,h,w)
        target1 = x_reshape[:,:3,...].clone()
        
        params2 = self.prior_ar1(x_reshape[:,:3,...])
        target2 = x_reshape[:,3:6,...].clone()
        
        params3 = self.prior_ar2(x_reshape[:,:6,...])
        target3 = x_reshape[:,6:9,...].clone() 
        
        params4 = self.prior_ar3(x_reshape[:,:9,...])
        target4 = x_reshape[:,9:12,...].clone()

        img_size = x.numel()
        img_rate1 = weak_colorar_rate(params1, target1, self.param.img_bitdepth, self.param.freq_precision).sum()
        img_rate2 = weak_colorar_rate(params2, target2, self.param.img_bitdepth, self.param.freq_precision).sum()
        img_rate3 = weak_colorar_rate(params3, target3, self.param.img_bitdepth, self.param.freq_precision).sum()
        img_rate4 = weak_colorar_rate(params4, target4, self.param.img_bitdepth, self.param.freq_precision).sum()
        img_bpd = (img_rate1 + img_rate2 + img_rate3 + img_rate4)/img_size
        
        return img_bpd

    @th.no_grad()
    def get_prior(self, x:Tensor):
        x_reshape = space2depth(x)
        _, _, h, w = x_reshape.size()
        params1 = self.param0.repeat(1,1,h,w)
        params2 = self.prior_ar1(x_reshape[:,:3,...])
        params3 = self.prior_ar2(x_reshape[:,:6,...])
        params4 = self.prior_ar3(x_reshape[:,:9,...])
        return depth2space(th.cat([params1, params2, params3, params4], dim=1))

    def to_device(self, device:th.device)->None:
        self.to(device)
    
    @th.no_grad()
    def encode(self, x:Tensor, delta_params:Tensor, coeffs:List[Tensor]):
        x_maxv = (1 << self.bit_depth) - 1
        x = th.round(x_maxv*x).to(th.int16).float()/x_maxv
        priors = self.get_prior(x)
        priors = th.chunk(space2depth(priors), 4, dim=1)
        priors = [prior * coeff for prior, coeff in zip(priors, coeffs)]
        priors = depth2space(th.cat(priors, dim=1))
        params = priors + delta_params
        mu, scale = get_mu_and_scale_linear_color(params, x)
        mu_reshape = space2depth(mu)
        x_reshape = th.round(space2depth(x) * ((1 << self.bit_depth) - 1)).to(th.int16).cpu()

        scale_reshape = space2depth(scale) 
        byte_strings = []
        for i in range(12):
            symbols = x_reshape[:,i:i+1,...]
            cur_cdfs = compute_logistic_cdfs(mu_reshape[:,i:i+1,...], scale_reshape[:,i:i+1,...], self.bit_depth).cpu()
            byte_strings.append(torchac.encode_int16_normalized_cdf(cur_cdfs, symbols))
        return byte_strings
    
    @th.no_grad()
    def decode(self, byte_strings:bytes, delta_params:Tensor, device:str, coeffs:List[Tensor]):
        assert len(byte_strings) == 12
        x_maxv = (1 << self.bit_depth) - 1
        params:Tensor = space2depth(delta_params)
        
        _, _, h, w = params.size()
        params[:,:9,...] += self.param0.repeat(1,1,h,w) * coeffs[0]
        x_rec = th.zeros(1, 12, h, w, device=device)
        nets = [self.prior_ar1, self.prior_ar2, self.prior_ar3]
        for i in range(4):
            cur_param = params[:,9*i:9*i+9,...]
            _mu, log_scale, pp = th.chunk(cur_param, 3, dim=1)
            mu = th.zeros_like(_mu)
            mu[:,:1,...] = _mu[:,:1,...]
            alpha, beta, gamma = th.chunk(pp, 3, dim=1)
            scale = get_scale(log_scale)

            cur_cdfs_r = compute_logistic_cdfs(mu[:,:1,...], scale[:,:1,...], self.bit_depth).cpu()
            symbols_r = torchac.decode_int16_normalized_cdf(cur_cdfs_r, byte_strings[3*i])

            x_r = symbols_r.reshape(1, 1, h, w).float().to(device) / x_maxv
            x_rec[:,3*i:3*i+1,...] = x_r
            mu[:,1:2,...] = _mu[:,1:2,...] + x_r * alpha
            cur_cdfs_g = compute_logistic_cdfs(mu[:,1:2,...], scale[:,1:2,...], self.bit_depth).cpu()
            symbols_g = torchac.decode_int16_normalized_cdf(cur_cdfs_g, byte_strings[3*i+1])
            x_g = symbols_g.reshape(1, 1, h, w).float().to(device) / x_maxv
            x_rec[:,3*i+1:3*i+2,...] = x_g
            mu[:,2:3,...] = _mu[:,2:3,...] + x_r * beta + x_g * gamma
            cur_cdfs_b = compute_logistic_cdfs(mu[:,2:3,...], scale[:,2:3,...], self.bit_depth).cpu()
            symbols_b = torchac.decode_int16_normalized_cdf(cur_cdfs_b, byte_strings[3*i+2])
            x_b = symbols_b.reshape(1, 1, h, w).float().to(device) / x_maxv
            x_rec[:,3*i+2:3*i+3,...] = x_b
            if i<3:
                params[:,9*i+9:9*i+18,...] += nets[i](x_rec[:,:3*i+3,...]) * coeffs[i+1]
        return depth2space(x_rec)