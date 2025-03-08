import torch as th
import numpy as np
import torch.nn.functional as F
from torch import Tensor, index_select
from einops import rearrange
from typing import List, Tuple

def space2depth(x:Tensor)->Tensor:
    return th.cat(
            [
                x[...,0::2,0::2],
                x[...,1::2,1::2],
                x[...,0::2,1::2],
                x[...,1::2,0::2]
            ]
            ,dim=1
        )

def space2depth_np(x:np.ndarray)->np.ndarray:
    return np.concatenate(
            [
                x[...,0::2,0::2],
                x[...,1::2,1::2],
                x[...,0::2,1::2],
                x[...,1::2,0::2]
            ]
            ,axis=1
        )

def depth2space(x:Tensor)->Tensor:
    n, c, w, h = x.shape
    cc = c//4
    ret = th.zeros(n, cc, w*2, h*2, device=x.device)
    ret[...,0::2,0::2] = x[:,0:cc,...]
    ret[...,1::2,1::2] = x[:,cc:2*cc,...]
    ret[...,0::2,1::2] = x[:,2*cc:3*cc,...]
    ret[...,1::2,0::2] = x[:,3*cc:4*cc,...]
    return ret

def get_unfold(x: Tensor, mask_size: int)->Tensor:
    pad = int((mask_size - 1) / 2)
    x_pad = F.pad(x, (pad, pad, pad, pad), mode='constant', value=0.)
    x_pad = F.pad(x, (pad, pad, pad, pad), mode='constant', value=0.)
    # Shape of x_unfold is [B, C, H, W, mask_size, mask_size]
    x_unfold = x_pad.unfold(2, mask_size, step=1).unfold(3, mask_size, step=1)
    return x_unfold

def get_neighbor(x: Tensor, mask_size: int, non_zero_pixel_ctx_idx: Tensor) -> Tensor:
    """Use the unfold function to extract the neighbors of each pixel in x.

    Args:
        x (Tensor): [b, 3, H, W] feature map from which we wish to extract the
            neighbors
        mask_size (int): Virtual size of the kernel around the current coded latent.
            mask_size = 2 * n_ctx_rowcol - 1
        non_zero_pixel_ctx_idx (Tensor): [N] 1D tensor containing the indices
            of the non zero context pixels 

    Returns:
        torch.tensor: [(b H W) (c n)] the spatial neighbors
    """
    pad = int((mask_size - 1) / 2)
    x_pad = F.pad(x, (pad, pad, pad, pad), mode='constant', value=0.)
    # Shape of x_unfold is [B, C, H, W, mask_size, mask_size]
    x_unfold = x_pad.unfold(2, mask_size, step=1).unfold(3, mask_size, step=1)

    # Convert x_unfold to a 2D tensor: [Number of pixels, all neighbors]
    x_unfold = rearrange(
        x_unfold, 'b c h w mask_h mask_w -> (b c h w) (mask_h mask_w)'
    )
    # Select the pixels for which the mask is not zero
    # For a N x N mask, select only the first (N x N - 1) / 2 pixels
    # (those which aren't null)
    # ! index_select is way faster than normal indexing when using torch script
    neighbor = index_select(x_unfold, dim=-1, index=non_zero_pixel_ctx_idx)

    return neighbor

def get_flat_latent_and_context(
    sent_latent: List[Tensor],
    mask_size: int,
    non_zero_pixel_ctx_idx: Tensor,
) -> Tuple[Tensor, Tensor]:
    """From a list of C tensors [1, 1, H_i, W_i], where H_i = H / 2 ** i,
    extract all the visible neighbors based on spatial_ctx_mask.

    Args:
        sent_latent (List[Tensor]): C tensors [1, 1, H_i, W_i], where H_i = H / 2 ** i
        mask_size (int): Virtual size of the kernel around the current coded latent.
            mask_size = 2 * n_ctx_rowcol - 1
        non_zero_pixel_ctx_idx (Tensor): [N] 1D tensor containing the indices
            of the non zero context pixels (i.e. floor(N ** 2 / 2) - 1).
            It looks like: [0, 1, ..., floor(N ** 2 / 2) - 1].
            This allows to use the index_select function, which is significantly
            faster than usual indexing.

    Returns:
        Tuple[Tensor, Tensor]:
            flat_latent [B], all the sent latent variables as a 1D tensor.
            flat_context [B, N_neighbors], the neighbors of each latent variable.
    """
    flat_latent_list: List[Tensor] = []
    flat_context_list: List[Tensor] = []

    # ============================= Context ============================= #
    # Get all the context as a single 2D vector of size [B, context size]
    for spatial_latent_i in sent_latent:
        # Nothing to do when we have an empty latent
        if spatial_latent_i.numel() == 0:
            continue

        flat_context_list.append(
            get_neighbor(spatial_latent_i, mask_size, non_zero_pixel_ctx_idx)
        )

        flat_latent_list.append(spatial_latent_i.view(-1))
    flat_context: Tensor = th.cat(flat_context_list, dim=0)

    # Get all the B latent variables as a single one dimensional vector
    flat_latent: Tensor = th.cat(flat_latent_list, dim=0)
    # ============================= Context ============================= #
    return flat_latent, flat_context

def pad_image(x:Tensor, scale:int)->Tensor:
    '''pad on the right and bottom, scale: log2 of the padding size'''
    scale = 1 << scale
    h, w = x.shape[-2:]
    pad_h = (scale - (h % scale)) % scale
    pad_w = (scale - (w % scale)) % scale
    return F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0.)