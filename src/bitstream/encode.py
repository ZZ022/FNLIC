import math
import os
import subprocess
import torch
import numpy as np

from bitstream.decode import decode_network
from bitstream.header import write_header
from cbench.ans import TansEncoder
from bitstream.range_coder import RangeCoder
from models.components.arm import Arm
from models.components.upsampling import Upsampling
from models.components.synthesis import Synthesis
from models.fnlic import FNLIC
from utils.misc import POSSIBLE_Q_STEP_ARM_NN, POSSIBLE_Q_STEP_SYN_NN, POSSIBLE_Q_STEP_UPS_NN, POSSIBLE_SCALE_NN, FIXED_POINT_FRACTIONAL_MULT, DescriptorNN, DescriptorOverfitter
from utils.distribution import fsar_freqs
from utils.helpers import get_neighbor


def get_ac_max_val_nn(fnlic: FNLIC) -> int:
    """Look within the neural networks of a frame encoder. Return the maximum
    amplitude of the quantized model (i.e. weight / q_step).
    This allows to get the AC_MAX_VAL constant, used by the entropy coder. All
    symbols sent by the entropy coder will be in [-AC_MAX_VAL, AC_MAX_VAL - 1].

    Args:
        fnlic (FNLIC): Model to quantize.

    Returns:
        int: The AC_MAX_VAL parameter.
    """
    model_param_quant = []

    # Loop on all the modules to be sent, and find the biggest quantized value
    for cur_module_name in fnlic.encoder.modules_to_send:
        module_to_encode = getattr(fnlic.encoder, cur_module_name)

        # Retrieve all the weights and biases for the ARM MLP
        for k, v in module_to_encode.named_parameters():
            if cur_module_name == 'arm':
                Q_STEPS = POSSIBLE_Q_STEP_ARM_NN
            else:
                Q_STEPS = POSSIBLE_Q_STEP_SYN_NN

            if k.endswith('.w') or k.endswith('.weight'):
                # Find the index of the closest quantization step in the list of
                # the possible quantization step.
                cur_q_step_index = int(torch.argmin(
                    (Q_STEPS - module_to_encode._q_step.get('weight')).abs()
                ).item())

                # Quantize the weight with the actual quantization step and add it
                # to the list of (quantized) weights
                model_param_quant.append(torch.round(v / Q_STEPS[cur_q_step_index]).flatten())

            elif k.endswith('.b') or k.endswith('.bias'):
                # Find the index of the closest quantization step in the list of
                # the possible quantization step.
                cur_q_step_index = int(torch.argmin(
                    (Q_STEPS - module_to_encode._q_step.get('bias')).abs()
                ).item())

                # Quantize the bias with the actual quantization step and add it
                # to the list of (quantized) bias
                model_param_quant.append(torch.round(v / Q_STEPS[cur_q_step_index]).flatten())

    # Gather them
    model_param_quant = torch.cat(model_param_quant).flatten()

    # Compute AC_MAX_VAL.
    AC_MAX_VAL = int(torch.ceil(model_param_quant.abs().max() + 2).item())
    return AC_MAX_VAL

def get_ac_max_val_latent(fnlic: FNLIC) -> int:
    """Look within the latent variables of a frame encoder. Return the maximum
    amplitude of the quantized latent variables.
    This allows to get the AC_MAX_VAL constant, used by the entropy coder. All
    symbols sent by the entropy coder will be in [-AC_MAX_VAL, AC_MAX_VAL - 1].

    Args:
        fnlic (FNLIC): Model storing the latent.

    Returns:
        int: The AC_MAX_VAL parameter.
    """
    latent = fnlic.encoder.get_quantized_latent()
    latent = torch.cat([cur_latent.flatten() for cur_latent in latent])
    # Compute AC_MAC_VAL
    AC_MAX_VAL = int(torch.ceil(latent.abs().max()).item())
    return AC_MAX_VAL

@torch.no_grad()
def fnlic_encode(fnlic: FNLIC, bitstream_path: str, device:str, freq_pre:int=12):
    """Convert a model to a bitstream located at <bitstream_path>.

    Args:
        model (FNLIC): A trained and quantized model
        bitstream_path (str): Where to save the bitstream
    """
    abs_path = lambda x: os.path.join(x)
    torch.use_deterministic_algorithms(True)
    fnlic.set_to_eval()
    fnlic.to_device(device)
    prefitter = fnlic.prefitter

    subprocess.call(f'rm -f {bitstream_path}', shell=True)
    # ================= Encode the MLP into a bitstream file ================ #
    ac_max_val_nn = get_ac_max_val_nn(fnlic)
    range_coder_nn = RangeCoder(
        0,
        ac_max_val_nn,
    )

    scale_index_nn: DescriptorOverfitter = {}
    q_step_index_nn: DescriptorOverfitter = {}
    n_bytes_nn: DescriptorOverfitter = {}
    for cur_module_name in fnlic.encoder.modules_to_send:
        # Prepare to store values dedicated to the current modules
        scale_index_nn[cur_module_name] = {}
        q_step_index_nn[cur_module_name] = {}
        n_bytes_nn[cur_module_name] = {}

        module_to_encode = getattr(fnlic.encoder, cur_module_name)

        weights, bias = [], []
        # Retrieve all the weights and biases for the ARM MLP
        q_step_index_nn[cur_module_name]['weight'] = -1
        q_step_index_nn[cur_module_name]['bias'] = -1
        for k, v in module_to_encode.named_parameters():
            assert cur_module_name in ['arm', 'synthesis', 'upsampling'], f'Unknow module name {cur_module_name}. '\
                'Module name should be in ["arm", "synthesis", "upsampling"].'

            if cur_module_name == 'arm':
                Q_STEPS = POSSIBLE_Q_STEP_ARM_NN
            elif cur_module_name == 'synthesis':
                Q_STEPS = POSSIBLE_Q_STEP_SYN_NN
            elif cur_module_name == 'upsampling':
                Q_STEPS = POSSIBLE_Q_STEP_UPS_NN

            if k.endswith('.w') or k.endswith('.weight'):
                # Find the index of the closest quantization step in the list of
                # the possible quantization step.
                cur_q_step_index = int(torch.argmin(
                    (Q_STEPS - module_to_encode._q_step.get('weight')).abs()
                ).item())

                # Store it into q_step_index_nn. It is overwritten for each
                # loop but it does not matter
                q_step_index_nn[cur_module_name]['weight'] = cur_q_step_index

                # Quantize the weight with the actual quantization step and add it
                # to the list of (quantized) weights
                weights.append(torch.round(v / Q_STEPS[cur_q_step_index]).flatten())

            elif k.endswith('.b') or k.endswith('.bias'):
                # Find the index of the closest quantization step in the list of
                # the Q_STEPS quantization step.
                cur_q_step_index = int(torch.argmin(
                    (Q_STEPS - module_to_encode._q_step.get('bias')).abs()
                ).item())

                # Store it into q_step_index_nn. It is overwritten for each
                # loop but it does not matter
                q_step_index_nn[cur_module_name]['bias'] = cur_q_step_index

                # Quantize the bias with the actual quantization step and add it
                # to the list of (quantized) bias
                bias.append(torch.round(v / Q_STEPS[cur_q_step_index]).flatten())

        # Gather them
        weights = torch.cat(weights).flatten()
        have_bias = len(bias) != 0
        if have_bias:
            bias = torch.cat(bias).flatten()

        floating_point_scale_weight = weights.std().item() / math.sqrt(2)
        if have_bias:
            floating_point_scale_bias = bias.std().item() / math.sqrt(2)

        # Find the closest element to the actual scale in the POSSIBLE_SCALE_NN list
        scale_index_weight = int(
            torch.argmin((POSSIBLE_SCALE_NN - floating_point_scale_weight).abs()).item()
        )
        if have_bias:
            scale_index_bias = int(
                torch.argmin((POSSIBLE_SCALE_NN - floating_point_scale_bias).abs()).item()
            )
        # Store this information for the header
        scale_index_nn[cur_module_name]['weight'] = scale_index_weight
        scale_index_nn[cur_module_name]['bias'] = scale_index_bias if have_bias else -1

        scale_weight = POSSIBLE_SCALE_NN[scale_index_weight]
        if scale_index_bias >= 0:
            scale_bias = POSSIBLE_SCALE_NN[scale_index_bias]

        # ----------------- Actual entropy coding
        # It happens on cpu
        weights = weights.cpu()
        if have_bias:
            bias = bias.cpu()

        cur_bitstream_path = abs_path(f'{bitstream_path}_{cur_module_name}_weight')
        range_coder_nn.encode(
            cur_bitstream_path,
            weights,
            torch.zeros_like(weights),
            scale_weight * torch.ones_like(weights),
            CHW = None,     # No wavefront coding for the weights
        )
        n_bytes_nn[cur_module_name]['weight'] = os.path.getsize(cur_bitstream_path)

        if have_bias:
            cur_bitstream_path = abs_path(f'{bitstream_path}_{cur_module_name}_bias')
            range_coder_nn.encode(
                cur_bitstream_path,
                bias,
                torch.zeros_like(bias),
                scale_bias * torch.ones_like(bias),
                CHW = None,     # No wavefront coding for the bias
            )
            n_bytes_nn[cur_module_name]['bias'] = os.path.getsize(cur_bitstream_path)
        else:
            n_bytes_nn[cur_module_name]['bias'] = 0
    # ================= Encode the MLP into a bitstream file ================ #

    # =============== Encode the latent into a bitstream file =============== #
    # To ensure perfect reproducibility between the encoder and the decoder,
    # we load the the different sub-networks from the bitstream here.
    for module_name in fnlic.encoder.modules_to_send:
        assert module_name in ['arm', 'synthesis', 'upsampling'], f'Unknow module name {module_name}. '\
            'Module name should be in ["arm", "synthesis", "upsampling"].'

        if module_name == 'arm':
            empty_module = Arm(
                2,
                fnlic.encoder.param.layers_arm
            )
            Q_STEPS = POSSIBLE_Q_STEP_ARM_NN
        elif module_name == 'synthesis':
            empty_module =  Synthesis(
                fnlic.encoder.n_latents,
                fnlic.encoder.param.layers_synthesis
            )
            Q_STEPS = POSSIBLE_Q_STEP_SYN_NN
        elif module_name == 'upsampling':
            empty_module = Upsampling(fnlic.encoder.param.upsampling_kernel_size)
            Q_STEPS = POSSIBLE_Q_STEP_SYN_NN

        have_bias = q_step_index_nn[module_name].get('bias') >= 0
        loaded_module = decode_network(
            empty_module,
            DescriptorNN(
                weight = abs_path(f'{bitstream_path}_{module_name}_weight'),
                bias = abs_path(f'{bitstream_path}_{module_name}_bias') if have_bias else "",
            ),
            DescriptorNN (
                weight = Q_STEPS[q_step_index_nn[module_name]['weight']],
                bias = Q_STEPS[q_step_index_nn[module_name]['bias']] if have_bias else 0,
            ),
            DescriptorNN (
                weight = POSSIBLE_SCALE_NN[scale_index_nn[module_name]['weight']],
                bias = POSSIBLE_SCALE_NN[scale_index_nn[module_name]['bias']] if have_bias else 0,
            ),
            ac_max_val_nn
        )
        setattr(fnlic.encoder, module_name, loaded_module)
    fnlic.encoder.arm.set_quant(FIXED_POINT_FRACTIONAL_MULT)
    fnlic.encoder.to_device(device)

    ac_max_val_latent = get_ac_max_val_latent(fnlic)
    max_latent_value = 2*ac_max_val_latent + 1
    latent_encoder = TansEncoder(table_log=freq_pre, max_symbol_value=max_latent_value-1)
    
    # get freqs
    arm = fnlic.encoder.arm
    freqs = fsar_freqs(arm, ac_max_val_latent, freq_pre, device)
    n_state = freqs.shape[0]
    n_symbols = np.ones(n_state, dtype=int) * max_latent_value
    offsets_latent = np.zeros(n_state, dtype=int)
    latent_encoder.init_params(freqs, n_symbols, offsets_latent)

    # Encode the different 2d latent grids one after the other
    n_bytes_per_latent = []
    # Loop on the different resolutions

    latents = fnlic.encoder.get_quantized_latent()
    for i in range(len(latents)):
        latent = torch.round(latents[i]).int()
        neighbors = get_neighbor(latent, 3, fnlic.encoder.non_zero_pixel_ctx_index) + ac_max_val_latent
        indexes = neighbors[:, 0] * max_latent_value + neighbors[:, 1]
        symbols = (latent + ac_max_val_latent).flatten().cpu().numpy()
        indexes = indexes.flatten().cpu().numpy()
        cur_latent_bitstream = abs_path(bitstream_path+f'latent{i}')
        byte_string = latent_encoder.encode_with_indexes(symbols, indexes)
        with open(cur_latent_bitstream, 'wb') as f:
            f.write(byte_string)
        n_bytes_per_latent.append(os.path.getsize(cur_latent_bitstream))


    # prefitter encode x

    byte_strings = prefitter.encode(fnlic.img_t, fnlic.encoder.get_delta(), fnlic.encoder.coefficients)
    n_bytes_img = [len(byte_string) for byte_string in byte_strings]
    # write to file
    for i, byte_string in enumerate(byte_strings):
        with open(abs_path(f'{bitstream_path}_prefitter{i}'), 'wb') as f:
            f.write(byte_string)

    # Write the header
    header_path = abs_path(f'{bitstream_path}_header')
    coeffs = [x.flatten().detach().cpu().numpy() for x in fnlic.encoder.coefficients]
    write_header(
        fnlic,
        header_path,
        n_bytes_per_latent,
        q_step_index_nn,
        scale_index_nn,
        n_bytes_nn,
        n_bytes_img,
        ac_max_val_nn,
        ac_max_val_latent,
        coeffs
    )

    # Concatenate everything inside a single file
    subprocess.call(f'rm -f {bitstream_path}', shell=True)
    subprocess.call(f'cat {header_path} >> {bitstream_path}', shell=True)
    subprocess.call(f'rm -f {header_path}', shell=True)

    for cur_module_name in ['arm', 'upsampling', 'synthesis']:
        for parameter_type in ['weight', 'bias']:
            cur_bitstream = abs_path(f'{bitstream_path}_{cur_module_name}_{parameter_type}')
            if os.path.exists(cur_bitstream):
                subprocess.call(f'cat {cur_bitstream} >> {bitstream_path}', shell=True)
                subprocess.call(f'rm -f {cur_bitstream}', shell=True)

    for i in range(len(latents)):
        cur_latent_bitstream = abs_path(bitstream_path+f'latent{i}')
        subprocess.call(f'cat {cur_latent_bitstream} >> {bitstream_path}', shell=True)
        subprocess.call(f'rm -f {cur_latent_bitstream}', shell=True)
    
    for i in range(len(byte_strings)):
        cur_img_bitstream = abs_path(f'{bitstream_path}_prefitter{i}')
        subprocess.call(f'cat {cur_img_bitstream} >> {bitstream_path}', shell=True)
        subprocess.call(f'rm -f {cur_img_bitstream}', shell=True)

    # Encoding's done, we no longer need deterministic algorithms
    torch.use_deterministic_algorithms(False)
    return os.path.getsize(bitstream_path)*8/fnlic.encoder.img_size
