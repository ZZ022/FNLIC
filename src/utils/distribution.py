import torch as th
import numpy as np
from torch import Tensor
from typing import Tuple
rans_freq_precision:int = 16

def get_scale(logscale:Tensor)->Tensor:
    '''
        function to calculate the scale from the log-scale
    '''
    return th.exp(-0.5 * th.clamp(logscale, min=-10., max=13.8155))

def get_mu_scale(params:Tensor)->Tuple[Tensor, Tensor]:
    '''
        function to calculate the mean and scale from the parameters
    '''
    mu, logscale = th.chunk(params, 2, dim=1)
    scale = get_scale(logscale)
    return mu, scale

def modify_prob(prob:Tensor, precision:int)->Tensor:
    '''
        function to modify the frequencies of the symbols that all possible value has at lease 1 frequency
        by adding 1 to the last symbol
        freqs: ndarray (ndist, symbols)
    '''
    a = 2**(-precision)
    prob = prob.clamp_min(a)
    return prob

def modify_regular_prob(probs:Tensor)->Tensor:
    '''
        function to modify the frequencies of the symbols that all possible value has at lease 1 frequency and the sum of the frequencies is 2**rans_freq_precision
    '''
    a = 2 ** (-rans_freq_precision)
    n = 256
    probs = probs*(1-n*a)+a
    return probs

def compute_logistic_cdfs(mu:Tensor, scale:Tensor, bitdepth:Tensor)->Tensor:
    '''
        function to calculate the cdfs of the Logistic(mu, scale) distribution
        used for encoder
    '''
    mu = mu
    scale = scale
    n,c,h,w=mu.shape
    max_v = float((1<<bitdepth)-1)
    size = 1 << bitdepth
    interval = 1. / max_v
    endpoints = th.arange(-1.0+interval,1.0, 2*interval, device=mu.device).repeat((n,c,h,w,1)) # n c w h maxv
    mu = mu.unsqueeze(-1).repeat((1,1,1,1,size-1)) # n c w h maxv
    scale = scale.unsqueeze(-1).repeat((1,1,1,1,size-1)) # n c w h maxv
    invscale = 1. / scale
    endpoints_rescaled = (endpoints-mu)*invscale
    cdfs = th.zeros(n,c,h,w,size+1, device=mu.device)
    cdfs[...,1:-1] = th.sigmoid(endpoints_rescaled)
    cdfs[...,-1] = 1.0
    probs = cdfs[...,1:] - cdfs[...,:-1]
    probs = modify_regular_prob(probs)
    th.use_deterministic_algorithms(False)
    cdfs[...,1:] = th.cumsum(probs, dim=-1)
    th.use_deterministic_algorithms(True)
    cdfs[...,-1] = 1.0
    cdfs_q = th.round(cdfs * float(1<<rans_freq_precision)).to(th.int16)
    return cdfs_q

def discretized_logistic_prob(mu:Tensor, scale:Tensor, x:Tensor, bitdepth:int=8)->Tensor:
    '''
        function to calculate the log-probability of x under a discretized Logistic(mu, scale) distribution
        heavily based on discretized_mix_logistic_loss() in https://github.com/openai/pixel-cnn
        x in [0, 1]
    '''
    # [0,255] -> [-1.1] (this means bin sizes of 2./255.)
    max_v = float((1<<bitdepth)-1)
    
    x_rescaled = x * 2.0 - 1
    # a, b = x_rescaled.min(), x_rescaled.max()
    invscale = 1. / scale
    thre = 1 - 1/max_v/2
    x_centered = x_rescaled - mu

    plus_in = invscale * (x_centered + 1. / max_v)
    cdf_plus = th.sigmoid(plus_in)
    min_in = invscale * (x_centered - 1. / max_v)
    cdf_min = th.sigmoid(min_in)

    diff = cdf_plus - cdf_min
    cond1 = th.where(x_rescaled < -thre, cdf_plus, diff)
    prob = th.where(x_rescaled > thre, th.ones_like(cdf_min)-cdf_min, cond1)
    return prob


def discretized_logistic_logp(mu:Tensor, scale:Tensor, x:Tensor, bitdepth:int=8, freq_precision:int=16)->Tensor:
    '''
        function to calculate the log-probability of x under a discretized Logistic(mu, scale) distribution
        heavily based on discretized_mix_logistic_loss() in https://github.com/openai/pixel-cnn
        x in [0, 1]
    '''
    # [0,255] -> [-1.1] (this means bin sizes of 2./255.)
    n = 1 << bitdepth
    a = 2**(-freq_precision)
    prob = discretized_logistic_prob(mu, scale, x, bitdepth)
    a = float(2**(-freq_precision))
    prob = (1-n*a)*prob + a
    logp = th.log2(prob)
    return logp

def get_mu_and_scale_linear_color(params:Tensor, x:Tensor)->Tuple[Tensor, Tensor]:
    '''
        function to calculate the mean and scale from the parameters
    '''
    _mu, log_scale, pp = th.chunk(params, 3, dim=1)
    alpha, beta, gamma = th.chunk(pp, 3, dim=1)
    mu = th.zeros_like(_mu)
    mu[:,0:1,...] = _mu[:,0:1,...]
    mu[:,1:2,...] = _mu[:,1:2,...] + alpha*x[:,0:1,...]
    mu[:,2:3,...] = _mu[:,2:3,...] + beta*x[:,0:1,...] + gamma*x[:,1:2,...]
    scale = get_scale(log_scale)
    return mu, scale

def weak_colorar_rate(params:Tensor, x:Tensor, bitdepth:int, freq_precision:int, log_nfreq:int=10)->Tuple[Tensor]:
    '''
       params N 9 H W, x normalized to [0,1]
    '''
    mu, scale = get_mu_and_scale_linear_color(params, x)
    logp = discretized_logistic_logp(mu, scale, x, bitdepth, freq_precision)
    return -logp
        
def laplace_cdf(x: Tensor, loc: Tensor, scale: Tensor) -> Tensor:
    """Compute the laplace cumulative evaluated in x. All parameters
    must have the same dimension.

    Args:
        x (Tensor): Where the cumulative if evaluated.
        loc (Tensor): Expectation.
        scale (Tensor): Scale

    Returns:
        Tensor: CDF(x, mu, scale)
    """
    return 0.5 - 0.5 * (x - loc).sign() * th.expm1(-(x - loc).abs() / scale)

def get_latent_rate(x: Tensor, params:Tensor, bitdepth:int, freq_precision:int) -> Tensor:
    """Compute the laplace log-probability evaluated in x. All parameters
    must have the same dimension.

    Args:
        x (Tensor): Where the log-probability if evaluated.
        loc (Tensor): Expectation.
        scale (Tensor): Scale

    Returns:
        Tensor: log(P(x | mu, scale))
    """
    n = 1 << bitdepth
    mu, scale = get_mu_scale(params)
    prob = laplace_cdf(x + 0.5, mu, scale) - laplace_cdf(x - 0.5, mu, scale)
    a = float(2**(-freq_precision))
    prob = (1-n*a)*prob + a
    logp = th.log2(prob)
    return -logp

def laplace_freqs(scale:float, num_symbols:int, freq_precision:int)->np.ndarray:
    '''
        function to calculate the frequencies of the symbols
    '''
    centers = th.arange(1, num_symbols) - num_symbols/2
    proba = laplace_cdf(centers + 0.5, 0, scale) - laplace_cdf(centers - 0.5, 0, scale)
    prob = modify_prob(proba, freq_precision)
    freqs = (prob*(1<<freq_precision)).round().int().unsqueeze(0).numpy()
    return freqs

@th.no_grad()
def fsar_freqs(arm:th.nn.Module, max_val:int, freq_precision:int, device:str)->np.ndarray:
    '''
        function to calculate the pmfs of the symbols in order 2
        symbols, 1 dim tensor of symbols
    '''
    scale = 1 << freq_precision
    num_symbols = 2*max_val+1
    symbols = th.arange(-max_val, max_val+1, 1.0).to(device)
    symbols_2d = th.cartesian_prod(symbols, symbols).float()
    params = arm(symbols_2d)
    mu, scale = get_mu_scale(params)
    mu = mu.repeat(1,num_symbols)
    scale = scale.repeat(1,num_symbols)
    symbols = symbols.repeat(num_symbols**2,1)
    prob = laplace_cdf(symbols + 0.5, mu, scale) - laplace_cdf(symbols - 0.5, mu, scale)
    prob = modify_prob(prob, freq_precision)
    freqs = (prob*(1<<freq_precision)).round().int().cpu().numpy()
    return freqs 