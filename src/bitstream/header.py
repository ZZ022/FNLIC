
"""
Bitstream structure
-------------------

Limitations:
    - MLPs hidden layers must be of constant size e.g. 24,24 or 16,16
    - One bitstream = one gop i.e. something starting with an intra frame

Header for the image:
------------------------------------
    ? ======================== IMAGE HEADER ======================== ?
    [Number of bytes used for the header]           2 bytes
    [Image height]                                  2 bytes
    [Image width]                                   2 bytes
    
    [Number of hidden layer ARM]                    1 byte  (e.g. 2 if archi='24,24')
    [Hidden layer dimension ARM]                    1 byte  (e.g. 24 if archi='24,24')
    [Upsampling layer kernel size]                  1 byte  (From 0 to 255)
    [Number of layer Synthesis]                     1 byte  (e.g. 2 if archi='24,24')
    [Synthesis layer 0 description ]                3 bytes  (Contains out_ft / k_size / layer_type / non-linearity)
                        ...
    [Synthesis layer N - 1 description ]            3 bytes  (Contains out_ft / k_size / layer_type / non-linearity)


    [Index of quantization step weight ARM]         1 bytes (From 0 to 255)
    [Index of quantization step bias ARM]           1 bytes (From 0 to 255)
    [Index of quantization step weight Upsampling]  1 bytes (From 0 to 255)
    [Index of quantization step bias Upsampling]    1 bytes (From 0 to 255)
    [Index of quantization step weight Synthesis]   1 bytes (From 0 to 255)
    [Index of quantization step bias Synthesis]     1 bytes (From 0 to 255)

    [Index of scale entropy coding weight ARM]      2 bytes (From 0 to 2 ** 16 - 1)
    [Index of scale entropy coding bias ARM]        2 bytes (From 0 to 2 ** 16 - 1)
    [Index of scale entropy coding weight Upsampling] 2 bytes (From 0 to 2 ** 16 - 1)
    [Index of scale entropy coding bias Upsampling]   2 bytes (From 0 to 2 ** 16 - 1)
    [Index of scale entropy coding weight Synthesis]2 bytes (From 0 to 2 ** 16 - 1)
    [Index of scale entropy coding bias Synthesis]  2 bytes (From 0 to 2 ** 16 - 1)

    [Number of bytes used for weight ARM]           2 bytes (less than 65535 bytes)
    [Number of bytes used for bias ARM]             2 bytes (less than 65535 bytes)
    [Number of bytes used for weight Upsampling]           2 bytes (less than 65535 bytes)
    [Number of bytes used for bias Upsampling]             2 bytes (less than 65535 bytes)
    [Number of bytes used for weight Synthesis]     2 bytes (less than 65535 bytes)
    [Number of bytes used for bias Synthesis]       2 bytes (less than 65535 bytes)

    ! Here, we split the 3D latent grids into several 2D grids. For instance,
    [Number of latents]                             1 byte
    [Number of bytes used for latent grid 0]        3 bytes (less than 16777215 bytes)
    [Number of bytes used for latent grid 1]        3 bytes (less than 16777215 bytes)
                        ...
    [Number of bytes used for latent grid N - 1]    3 bytes (less than 16777215 bytes)

    [Number of bytes used for subimage 0]           3 bytes (less than 16777215 bytes)
    [Number of bytes used for subimage 1]           3 bytes (less than 16777215 bytes)
    ...
    [Number of bytes used for subimage 11]          3 bytes (less than 16777215 bytes)
    ? ======================== FRAME HEADER ======================== ?

"""

import os
import numpy as np
from typing import List, Tuple, TypedDict
from models.components.synthesis import Synthesis
from models.fnlic import FNLIC
from utils.misc import MAX_AC_MAX_VAL, DescriptorOverfitter

_POSSIBLE_SYNTHESIS_MODE = [k for k in Synthesis.possible_mode]
_POSSIBLE_SYNTHESIS_NON_LINEARITY = [k for k in Synthesis.possible_non_linearity]

class FrameHeader(TypedDict):
    """Define the dictionary containing the header information as a type."""
    n_bytes_header: int                 # Number of bytes for the headerme
    n_latents: int           # Number of different latent resolutions
    n_bytes_per_latent: List[int]       # Number of bytes for each 2D latent
    img_shape: Tuple[int, int]           # Format: (height, width)
    layers_arm: List[int]               # Dimension of each hidden layer in the ARM
    upsampling_kernel_size: int         # Upsampling model kernel size
    layers_synthesis: List[str]         # Dimension of each hidden layer in the Synthesis
    q_step_index_nn: DescriptorOverfitter # Index of the quantization step used for the weight & bias of the NNs
    scale_index_nn: DescriptorOverfitter  # Index of the scale used for entropy code weight & bias of the NNs
    n_bytes_nn: DescriptorOverfitter      # Number of bytes for weight and bias of the NNs
    n_bytes_img:List[int]
    ac_max_val_nn: int                  # The range coder AC_MAX_VAL parameters for entropy coding the NNs
    ac_max_val_latent: int              # The range coder AC_MAX_VAL parameters for entropy coding the latents
    coeffs: List[np.ndarray]            # The coefficients

def write_header(
    fnlic: FNLIC,
    header_path: str,
    n_bytes_per_latent: List[int],
    q_step_index_nn: DescriptorOverfitter,
    scale_index_nn: DescriptorOverfitter,
    n_bytes_nn: DescriptorOverfitter,
    n_bytes_img:List[int],
    ac_max_val_nn: int,
    ac_max_val_latent: int,
    coeffs: List[np.ndarray]
):
    """Write a frame header to a a file located at <header_path>.
    The structure of the header is described above.

    Args:
        model (fnlic): Model from which info located in the header will be retrieved.
        header_path (str): Path of the file where the header is written.
        n_bytes_per_latent (List[int]): Indicates the number of bytes for each 2D
            latent grid.
        q_step_index_nn (DescriptorOverfitter): Dictionary containing the index of the
            quantization step for the weight and bias of each network.
        scale_index_nn (DescriptorOverfitter): Dictionary containing the index of the
            scale parameter used during the entropy coding of the weight and bias
            of each network.
        n_bytes_nn (DescriptorOverfitter): Dictionary containing the number of bytes
            used for the weights and biases of each network
        ac_max_val_nn (int): The range coder AC_MAX_VAL parameters for entropy coding the NNs
        ac_max_val_latent (int): The range coder AC_MAX_VAL parameters for entropy coding the latents
    """

    n_bytes_header = 0
    n_bytes_header += 2     # Number of bytes header
    n_bytes_header += 2     # Width of the image
    n_bytes_header += 2     # Height of the image

    n_bytes_header += 1     # Number hidden layer ARM
    n_bytes_header += 1     # Hidden layer dimension ARM
    n_bytes_header += 1     # Upsampling kernel size
    n_bytes_header += 1     # Number hidden layer Synthesis
    # Hidden Synthesis layer out#, kernelsz, mode+nonlinearity
    n_bytes_header += 3 * len(fnlic.encoder_param.layers_synthesis)

    n_bytes_header += 2     # AC_MAX_VAL for neural networks
    n_bytes_header += 1     # AC_MAX_VAL for the latent variables

    n_bytes_header += 1     # Index of the quantization step weight ARM
    n_bytes_header += 1     # Index of the quantization step bias ARM
    n_bytes_header += 1     # !! -1 => no bias # Index of the quantization step weight Upsampling
    n_bytes_header += 1     # Index of the quantization step bias Upsampling
    n_bytes_header += 1     # Index of the quantization step weight Synthesis
    n_bytes_header += 1     # Index of the quantization step bias Synthesis

    n_bytes_header += 2     # Index of scale entropy coding weight ARM
    n_bytes_header += 2     # Index of scale entropy coding bias ARM
    n_bytes_header += 2     # Index of scale entropy coding weight Upsampling
    n_bytes_header += 2     # Index of scale entropy coding weight Synthesis
    n_bytes_header += 2     # Index of scale entropy coding bias Synthesis

    n_bytes_header += 2     # Number of bytes for weight ARM
    n_bytes_header += 2     # Number of bytes for bias ARM
    n_bytes_header += 2     # Number of bytes for weight Upsampling
    n_bytes_header += 2     # Number of bytes for weight Synthesis
    n_bytes_header += 2     # Number of bytes for bias Synthesis

    n_bytes_header += 1     # Number of latents
    n_bytes_header += 3 * fnlic.encoder.n_latents  # Number of bytes for each 2D latent grid

    n_bytes_header += 3 * 12 # Number of bytes for each subimage
    n_bytes_header += 36 * 4 # Number of bytes for coefficients 

    byte_to_write = b''
    byte_to_write += n_bytes_header.to_bytes(2, byteorder='big', signed=False)
    byte_to_write += fnlic.encoder.img_shape[0].to_bytes(2, byteorder='big', signed=False)
    byte_to_write += fnlic.encoder.img_shape[1].to_bytes(2, byteorder='big', signed=False)

    byte_to_write += len(fnlic.encoder_param.layers_arm).to_bytes(1, byteorder='big', signed=False)
    # If no hidden layers in the ARM, model.param.layers_arm is an empty list. So write 0
    if len(fnlic.encoder_param.layers_arm) == 0:
        byte_to_write += int(0).to_bytes(1, byteorder='big', signed=False)
    else:
        byte_to_write += fnlic.encoder_param.layers_arm[0].to_bytes(1, byteorder='big', signed=False)

    byte_to_write += fnlic.encoder_param.upsampling_kernel_size.to_bytes(1, byteorder='big', signed=False)

    byte_to_write += len(fnlic.encoder_param.layers_synthesis).to_bytes(1, byteorder='big', signed=False)
    # If no hidden layers in the Synthesis, fnlic.encoder_param.layers_synthesis is an empty list. So write 0
    for layer_spec in fnlic.encoder_param.layers_synthesis:
        out_ft, k_size, mode, non_linearity = layer_spec.split('-')
        byte_to_write += int(out_ft).to_bytes(1, byteorder='big', signed=False)
        byte_to_write += int(k_size).to_bytes(1, byteorder='big', signed=False)
        byte_to_write += (_POSSIBLE_SYNTHESIS_MODE.index(mode)*16+_POSSIBLE_SYNTHESIS_NON_LINEARITY.index(non_linearity)).to_bytes(1, byteorder='big', signed=False)

    if ac_max_val_nn > MAX_AC_MAX_VAL:
        print(f'AC_MAX_VAL NN is too big!')
        print(f'Found {ac_max_val_nn}, should be smaller than {MAX_AC_MAX_VAL}')
        print(f'Exiting!')
        return
    if ac_max_val_latent > MAX_AC_MAX_VAL:
        print(f'AC_MAX_VAL latent is too big!')
        print(f'Found {ac_max_val_latent}, should be smaller than {MAX_AC_MAX_VAL}')
        print(f'Exiting!')
        return

    byte_to_write += ac_max_val_nn.to_bytes(2, byteorder='big', signed=False)
    byte_to_write += ac_max_val_latent.to_bytes(1, byteorder='big', signed=False)

    for nn_name in ['arm', 'upsampling', 'synthesis']:
        for nn_param in ['weight', 'bias']:
            cur_q_step_index = q_step_index_nn.get(nn_name).get(nn_param)
            if cur_q_step_index < 0:
                # hack -- ensure -1 translated to 255.
                byte_to_write += int(255).to_bytes(1, byteorder='big', signed=False)
            else:
                byte_to_write += cur_q_step_index.to_bytes(1, byteorder='big', signed=False)

    for nn_name in ['arm', 'upsampling', 'synthesis']:
        for nn_param in ['weight', 'bias']:
            cur_q_step_index = q_step_index_nn.get(nn_name).get(nn_param)
            if cur_q_step_index < 0:
                # no bias.
                continue
            cur_scale_index = scale_index_nn.get(nn_name).get(nn_param)
            byte_to_write += cur_scale_index.to_bytes(2, byteorder='big', signed=False)

    for nn_name in ['arm', 'upsampling', 'synthesis']:
        for nn_param in ['weight', 'bias']:
            cur_q_step_index = q_step_index_nn.get(nn_name).get(nn_param)
            if cur_q_step_index < 0:
                # no bias
                continue
            cur_n_bytes = n_bytes_nn.get(nn_name).get(nn_param)
            if cur_n_bytes > MAX_AC_MAX_VAL:
                print(f'Number of bytes for {nn_name} {nn_param} is too big!')
                print(f'Found {cur_n_bytes}, should be smaller than {MAX_AC_MAX_VAL}')
                print(f'Exiting!')
                return
            byte_to_write += cur_n_bytes.to_bytes(2, byteorder='big', signed=False)

    byte_to_write += fnlic.encoder_param.n_latents.to_bytes(1, byteorder='big', signed=False)

    for i, v in enumerate(n_bytes_per_latent):
        if v > 2 ** 24 - 1:
            print(f'Number of bytes for latent {i} is too big!')
            print(f'Found {v}, should be smaller than {2 ** 24 - 1}')
            print(f'Exiting!')
            return
    # for tmp in n_bytes_per_latent:
        byte_to_write += v.to_bytes(3, byteorder='big', signed=False)

    for i, v in enumerate(n_bytes_img):
        if v > 2 ** 24 - 1:
            print(f'Number of bytes for subimage {i} is too big!')
            print(f'Found {v}, should be smaller than {2 ** 24 - 1}')
            print(f'Exiting!')
            return
        byte_to_write += v.to_bytes(3, byteorder='big', signed=False)
    
    for coeff in coeffs:
        byte_to_write += coeff.tobytes()

    with open(header_path, 'wb') as fout:
        fout.write(byte_to_write)

    # print(n_bytes_header)
    if n_bytes_header != os.path.getsize(header_path):
        print('Invalid number of bytes in header!')
        print("expected", n_bytes_header)
        print("got", os.path.getsize(header_path))
        exit(1)

def read_header(bitstream: bytes) -> FrameHeader:
    """Read the first few bytes of a bitstream file located at
    <bitstream_path> and parse the different information.

    Args:
        bitstream_path (str): Path where the bitstream is located.

    Returns:
        FrameHeader: The parsed info from the bitstream.
    """

    ptr = 0
    n_bytes_header = int.from_bytes(bitstream[ptr: ptr + 2], byteorder='big', signed=False)
    ptr += 2

    img_height = int.from_bytes(bitstream[ptr: ptr + 2], byteorder='big', signed=False)
    ptr += 2
    img_width = int.from_bytes(bitstream[ptr: ptr + 2], byteorder='big', signed=False)
    ptr += 2


    n_hidden_dim_arm = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
    ptr += 1
    hidden_size_arm = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
    ptr += 1

    upsampling_kernel_size = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
    ptr += 1

    n_hidden_dim_synthesis = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
    ptr += 1
    layers_synthesis = []
    for i in range(n_hidden_dim_synthesis):
        out_ft = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
        ptr += 1
        kernel_size = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
        ptr += 1
        mode_non_linearity = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
        ptr += 1
        mode = _POSSIBLE_SYNTHESIS_MODE[mode_non_linearity // 16]
        non_linearity = _POSSIBLE_SYNTHESIS_NON_LINEARITY[mode_non_linearity % 16]
        layers_synthesis.append(f'{out_ft}-{kernel_size}-{mode}-{non_linearity}')

    ac_max_val_nn = int.from_bytes(bitstream[ptr: ptr + 2], byteorder='big', signed=False)
    ptr += 2
    ac_max_val_latent = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
    ptr += 1

    q_step_index_nn: DescriptorOverfitter = {}
    for nn_name in ['arm', 'upsampling', 'synthesis']:
        q_step_index_nn[nn_name] = {}
        for nn_param in ['weight', 'bias']:
            q_step_index_nn[nn_name][nn_param] = int.from_bytes(
                bitstream[ptr: ptr + 1], byteorder='big', signed=False
            )
            # Hack -- 255 -> -1 for upsampling bias.  Indiciating no bias.
            if q_step_index_nn[nn_name][nn_param] == 255:
                q_step_index_nn[nn_name][nn_param] = -1
                # print("got -1:", nn_name, nn_param)
            ptr += 1

    scale_index_nn: DescriptorOverfitter = {}
    for nn_name in ['arm', 'upsampling', 'synthesis']:
        scale_index_nn[nn_name] = {}
        for nn_param in ['weight', 'bias']:
            if q_step_index_nn[nn_name][nn_param] < 0:
                scale_index_nn[nn_name][nn_param] = -1
            else:
                scale_index_nn[nn_name][nn_param] = int.from_bytes(
                    bitstream[ptr: ptr + 2], byteorder='big', signed=False
                )
                ptr += 2

    n_bytes_nn: DescriptorOverfitter = {}
    for nn_name in ['arm', 'upsampling', 'synthesis']:
        n_bytes_nn[nn_name] = {}
        for nn_param in ['weight', 'bias']:
            if q_step_index_nn[nn_name][nn_param] < 0:
                n_bytes_nn[nn_name][nn_param] = -1
            else:
                n_bytes_nn[nn_name][nn_param] = int.from_bytes(
                    bitstream[ptr: ptr + 2], byteorder='big', signed=False
                )
                ptr += 2

    n_latents = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
    ptr += 1

    n_bytes_per_latent = []
    for _ in range(n_latents):
        n_bytes_per_latent.append(
            int.from_bytes(bitstream[ptr: ptr + 3], byteorder='big', signed=False)
        )
        ptr += 3

    n_bytes_img = []
    for _ in range(12):
        n_bytes_img.append(
            int.from_bytes(bitstream[ptr: ptr + 3], byteorder='big', signed=False)
        )
        ptr += 3

    coeffs = []

    for _ in range(4):
        coeffs.append(np.frombuffer(bitstream[ptr: ptr + 36], dtype=np.float32))
        ptr += 36

    header_info: FrameHeader = {
        'n_bytes_header': n_bytes_header,
        'n_latents': n_latents,
        'n_bytes_per_latent': n_bytes_per_latent,
        'img_shape': (img_height, img_width),
        'layers_arm': [hidden_size_arm for _ in range(n_hidden_dim_arm)],
        'upsampling_kernel_size': upsampling_kernel_size,
        'layers_synthesis': layers_synthesis,
        'q_step_index_nn': q_step_index_nn,
        'scale_index_nn': scale_index_nn,
        'n_bytes_nn': n_bytes_nn,
        'n_bytes_img':n_bytes_img,
        'ac_max_val_nn': ac_max_val_nn,
        'ac_max_val_latent': ac_max_val_latent,
        'coeffs': coeffs
    }

    # print('\nContent of the frame header:')
    # print('------------------------------')
    # for k, v in header_info.items():
    #     print(f'{k:>20}: {v}')
    # print('         ------------------------')

    return header_info