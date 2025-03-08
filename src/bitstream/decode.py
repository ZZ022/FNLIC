
import subprocess
import numpy as np
import torch

from torch import nn, Tensor
from bitstream.header import read_header
from bitstream.range_coder import RangeCoder
from cbench.ans import TansDecoder
from models.components.arm import Arm
from utils.misc import DescriptorNN, POSSIBLE_Q_STEP_ARM_NN, POSSIBLE_Q_STEP_SYN_NN, POSSIBLE_SCALE_NN, FIXED_POINT_FRACTIONAL_MULT
from models.components.upsampling import Upsampling
from models.components.synthesis import Synthesis
from utils.distribution import fsar_freqs
from models.prefitter import Prefitter

def decode_network(
    empty_module: nn.Module,
    bitstream_path: DescriptorNN,
    q_step_nn: DescriptorNN,
    scale_nn: DescriptorNN,
    ac_max_val: int,
) -> nn.Module:
    """Decode a neural network from a bitstream. The idea is to iterate
    on all the parameters of <empty_module>, filling it with values read
    from the bitstream.

    Args:
        empty_module (nn.Module): An empty (i.e. randomly initialized) instance
            of the network to load.
        bitstream_path (str): Weight and bias will be found at
            <bitstream_path>_weight and <bitstream_path>_arm
        q_step_nn (DescriptorNN): Describe the quantization steps used
            for the weight and bias of the network.
        scale_nn (DescriptorNN): Describe the scale parameters used
            for entropy coding of the weight and bias of the network.
        ac_max_val (int): Data are in [-ac_max_val, ac_max_val - 1]

    Returns:
        nn.Module: The decoded module
    """
    have_bias = q_step_nn.bias > 0

    # Instantiate two range coder objects to decode simultaneously weight and bias
    range_coder_nn_weight = RangeCoder(
        0,      # 0 because we don't have a ctx_row_col
        AC_MAX_VAL = ac_max_val
    )
    range_coder_nn_weight.load_bitstream(bitstream_path.weight)

    if have_bias:
        range_coder_nn_bias = RangeCoder(
            0,      # 0 because we don't have a ctx_row_col
            AC_MAX_VAL = ac_max_val
        )
        range_coder_nn_bias.load_bitstream(bitstream_path.bias)

    loaded_param = {}
    for k, v in empty_module.named_parameters():
        if k.endswith('.w') or k.endswith('.weight'):
            cur_scale = scale_nn.weight
            cur_q_step = q_step_nn.weight
            cur_param = range_coder_nn_weight.decode(
                torch.zeros_like(v.flatten()),
                torch.ones_like(v.flatten()) * cur_scale
            )
        elif k.endswith('.b') or k.endswith('.bias'):
            cur_scale = scale_nn.bias
            cur_q_step = q_step_nn.bias
            cur_param = range_coder_nn_bias.decode(
                torch.zeros_like(v.flatten()),
                torch.ones_like(v.flatten()) * cur_scale
            )
        else:
            # Ignore network parameters whose name does not end with '.w', '.b', '.weight', '.bias'
            continue

        # Don't forget inverse quantization!
        loaded_param[k] = cur_param.reshape_as(v)  * cur_q_step

    empty_module.load_state_dict(loaded_param)
    return empty_module

@torch.no_grad()
def fnlic_decode(bitstream_path: str, prefitter:Prefitter, device:str, freq_pre:int=12) -> Tensor:
    """Decode a bitstream located at <bitstream_path> and save the decoded image
    at <output_path>

    Args:
        bitstream_path (str): Absolute path of the bitstream. We keep that to output some temporary file
            at the same location than the bitstream itself
        bitstream (bytes): The bytes composing the bitstream
        img_size (Tuple[int, int]): Height and width of the frame

    Returns:
        Tensor: The decoded image with shape [1, 3, H, W] in [0., 1.]. Normalization, and conversion
            into 4:2:0 does **not** happen in this function
        bytes: The remaining bytes of the bitstream (i.e. for the following frames).
    """
    torch.use_deterministic_algorithms(True)
    bitstream = open(bitstream_path, 'rb').read()
    # ========================== Parse the header =========================== #
    header_info = read_header(bitstream)
    # ========================== Parse the header =========================== #

    # =================== Split the intermediate bitstream ================== #
    # The constriction range coder only works when we decode each latent grid
    # from a dedicated bitstream file.
    # The idea here is to split the single bitstream file to N files
    # bitstream_i, one for each latent grid
    # Read all the bits and discard the header
    bitstream = bitstream[header_info.get('n_bytes_header'): ]
    
    # For each module and each parameter type, keep the first bytes, write them
    # in a standalone file and keep the remainder of the bitstream in bitstream
    # Next parameter of next module will start at the first element of the bitstream.
    for cur_module_name in ['arm', 'upsampling', 'synthesis']:
        for parameter_type in ['weight', 'bias']:
            cur_n_bytes = header_info.get('n_bytes_nn')[cur_module_name][parameter_type]

            # Write the first byte in a dedicated file
            if cur_n_bytes > 0:
                bytes_for_current_parameters = bitstream[:cur_n_bytes]
                current_bitstream_file = f'{bitstream_path}_{cur_module_name}_{parameter_type}'
                with open(current_bitstream_file, 'wb') as f_out:
                    f_out.write(bytes_for_current_parameters)
                # Keep the rest of the bitstream for next loop
                bitstream = bitstream[cur_n_bytes: ]

    # For each latent grid: keep the first bytes to decode the current latent
    # and store the rest in bitstream. Latent i + 1 will start at the first
    # element of bitstream
    byte_string_latents = []
    for i in range(header_info.get('n_latents')):
        cur_n_bytes = header_info.get('n_bytes_per_latent')[i]

        # Write the first bytes in a dedicated file
        bytes_for_current_latent = bitstream[: cur_n_bytes]
        byte_string_latents.append(bytes_for_current_latent)
        # Keep the rest of the bitstream for next loop
        bitstream = bitstream[cur_n_bytes: ]
    # =================== Split the intermediate bitstream ================== #

    # ========== Reconstruct some information from the header data ========== #

    # =========================== Decode the NNs ============================ #
    # To decode each NN (arm, upsampling and synthesis):
    #   1. Instantiate an empty Module
    #   2. Populate it with the weights and biases decoded from the bitstream
    #   3. Send it to the requested device.
    arm = decode_network(
        Arm(2, header_info.get('layers_arm')),  # Empty module
        DescriptorNN(
            weight = f'{bitstream_path}_arm_weight',
            bias = f'{bitstream_path}_arm_bias',
        ),
        DescriptorNN (
            weight = POSSIBLE_Q_STEP_ARM_NN[header_info['q_step_index_nn']['arm']['weight']],
            bias = POSSIBLE_Q_STEP_ARM_NN[header_info['q_step_index_nn']['arm']['bias']],
        ),
        DescriptorNN (
            weight = POSSIBLE_SCALE_NN[header_info['scale_index_nn']['arm']['weight']],
            bias = POSSIBLE_SCALE_NN[header_info['scale_index_nn']['arm']['bias']],
        ),
        header_info.get('ac_max_val_nn'),
    )
    # Set the desired quantization accuracy for the ARM
    arm.set_quant(FIXED_POINT_FRACTIONAL_MULT)

    have_bias = header_info['q_step_index_nn']['upsampling']['bias'] >= 0
    if have_bias:
        # For the moment we do not expect this!  No bias for upsampling.
        print("WHAT")
        exit(1)
    upsampling = decode_network(
        Upsampling(
            header_info.get('upsampling_kernel_size')
        ),  # Empty module
        DescriptorNN(
            weight = f'{bitstream_path}_upsampling_weight',
            bias = "",
        ),
        DescriptorNN (
            weight = POSSIBLE_Q_STEP_SYN_NN[header_info['q_step_index_nn']['upsampling']['weight']],
            bias = 0,
        ),
        DescriptorNN (
            weight = POSSIBLE_SCALE_NN[header_info['scale_index_nn']['upsampling']['weight']],
            bias = 0,
        ),
        header_info.get('ac_max_val_nn'),
    )

    synthesis = decode_network(
        Synthesis(
            header_info.get('n_latents'), header_info.get('layers_synthesis')
        ),  # Empty module
        DescriptorNN(
            weight = f'{bitstream_path}_synthesis_weight',
            bias = f'{bitstream_path}_synthesis_bias',
        ),
        DescriptorNN (
            weight = POSSIBLE_Q_STEP_SYN_NN[header_info['q_step_index_nn']['synthesis']['weight']],
            bias = POSSIBLE_Q_STEP_SYN_NN[header_info['q_step_index_nn']['synthesis']['bias']],
        ),
        DescriptorNN (
            weight = POSSIBLE_SCALE_NN[header_info['scale_index_nn']['synthesis']['weight']],
            bias = POSSIBLE_SCALE_NN[header_info['scale_index_nn']['synthesis']['bias']],
        ),
        header_info.get('ac_max_val_nn'),
    )
    # =========================== Decode the NNs ============================ #

    # =========================== Decode the latents ========================= #


    # =========================== Decode the latents ========================= #

    # ================= Clean up the intermediate bitstream ================= #
    for cur_module_name in ['arm', 'synthesis', 'upsampling']:
        for parameter_type in ['weight', 'bias']:
            cur_bitstream = f'{bitstream_path}_{cur_module_name}_{parameter_type}'
            subprocess.call(f'rm -f {cur_bitstream}', shell=True)

    # ================= Clean up the intermediate bitstream ================= #
    ac_max_val_latent = header_info.get('ac_max_val_latent')
    max_latent_value = 2*ac_max_val_latent + 1
    latent_decoder = TansDecoder(table_log=freq_pre, max_symbol_value=max_latent_value-1)
    arm.to(device)
    for idx_layer, layer in enumerate(arm.mlp):
        if hasattr(layer, 'qw'):
            if layer.qw is not None:
                arm.mlp[idx_layer].qw = layer.qw.to(device)

        if hasattr(layer, 'qb'):
            if layer.qb is not None:
                arm.mlp[idx_layer].qb = layer.qb.to(device)
    freqs = fsar_freqs(arm, ac_max_val_latent, freq_pre, device)
    n_state = freqs.shape[0]
    n_symbols = np.ones(n_state, dtype=int) * max_latent_value
    offsets_latent = np.zeros(n_state, dtype=int)
    latent_decoder.init_params(freqs, n_symbols, offsets_latent)
    latents = []
    h_ori, w_ori = header_info.get('img_shape')
    n_latents = header_info.get('n_latents')
    padding = 1 << (n_latents - 1)
    h_pad = (h_ori//padding) * padding + padding * (h_ori % padding > 0)
    w_pad = (w_ori//padding) * padding + padding * (w_ori % padding > 0)
    for i in range(header_info.get('n_latents')):
        byte_string = byte_string_latents[i]
        h = h_pad//(2**i)
        w = w_pad//(2**i)
        latent = latent_decoder.decode_fsar(byte_string, ac_max_val_latent, max_latent_value, h, w)[1:,1:] - ac_max_val_latent
        latents.append(torch.tensor(latent).unsqueeze(0).unsqueeze(0).float().to(device))
    upsampling.to(device)
    synthesis_input = upsampling(latents)
    synthesis.to(device)
    synthesis_output = synthesis(synthesis_input)
    byte_strings = []
    n_bytes_img = header_info.get('n_bytes_img')
    for i in range(12):
        byte_strings.append(bitstream[:n_bytes_img[i]])
        bitstream = bitstream[n_bytes_img[i]:]
    coeffs = header_info.get('coeffs')
    coeffs = [torch.Tensor(coeff.copy()).view(1,-1,1,1).to(device) for coeff in coeffs]
    img_rec = prefitter.decode(byte_strings, synthesis_output, device, coeffs)
    return img_rec[...,:h_ori,:w_ori]
    