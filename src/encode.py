import os
import sys
import yaml
import subprocess
import argparse
import logging
import shutil
import torch as th

from PIL import Image
from glob import glob
from torchvision.transforms.functional import to_tensor

from utils.misc import get_best_device
from models.prefitter import PrefitterParameter, Prefitter
from models.overfitter import OverfitterParameter
from models.fnlic import EncoderManager, FNLIC, load_fnlic
from bitstream.encode import fnlic_encode

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def set_logger(dst:str):
    level = getattr(logging, 'INFO', None)
    handler = logging.FileHandler(dst)
    formatter = logging.Formatter('')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(level)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            logger.removeHandler(handler)
    

if __name__ == '__main__':
    # =========================== Parse arguments =========================== #
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,help='Path to the input image.png (RGB444)', type=str)
    parser.add_argument('-f', '--fnlic', default='', help='Path to overfitted model, overfit one if not set', type=str)
    parser.add_argument('-o', '--output', type=str, default="", help='Output bitstream path' )
    parser.add_argument('--workdir', help='Path to the working directory', type=str, default='../workspace')
    parser.add_argument('--prefitter_ckpt', help='Path to the prefitter checkpoint', type=str, default='../weight/prefitter.pth')
    parser.add_argument('--remove_workdir', help='Set to remove the working directory', action='store_true')

    parser.add_argument('--model_config', help='Path to the model configuration file', type=str, default='config/model_cfg.yaml')
    parser.add_argument('--training_config', help='Path to the overfit configuration file', type=str, default='config/training_cfg.yaml')
    
    args = parser.parse_args()

    with open(args.model_config, 'r') as f:
        model_cfg = yaml.safe_load(f)
    with open(args.training_config, 'r') as f:
        training_cfg = yaml.safe_load(f)
    assert len(training_cfg['alpha_inits']) > 0, 'Please provide at least one alpha_init'
    
    # =========================== Parse arguments =========================== #

    # ====================== Torchscript JIT parameters ===================== #
    # From https://github.com/pytorch/pytorch/issues/52286
    # This is no longer the case with the with torch.jit.fuser
    # ! This gives a significant (+25 %) speed up
    th._C._jit_set_profiling_executor(False)
    th._C._jit_set_texpr_fuser_enabled(False)
    th._C._jit_set_profiling_mode(False)
    # ====================== Torchscript JIT parameters ===================== #

    # =========================== Parse arguments =========================== #
    workdir = f'{args.workdir.rstrip("/")}/'
    if not os.path.exists(workdir):
        os.mkdir(workdir)
    shutil.copyfile(args.model_config, os.path.join(args.workdir, 'model_cfg.yaml'))
    shutil.copyfile(args.training_config, os.path.join(args.workdir, 'training_cfg.yaml'))

    # Dump raw parameters into a text file to keep track
    with open(f'{workdir}param.txt', 'w') as f_out:
        f_out.write(str(sys.argv))

    # Parse arguments
    layers_synthesis = [x for x in model_cfg['layers_synthesis'].split(',') if x != '']
    layers_arm = [int(x) for x in model_cfg['layers_arm'].split(',') if x != '']

    # Automatic device detection
    device = get_best_device()
    logging.info(f'{"Device":<20}: {device}')
    # =========================== Parse arguments =========================== #

    # ====================== Torchscript JIT parameters ===================== #
    # From https://github.com/pytorch/pytorch/issues/52286
    # This is no longer the case with the with torch.jit.fuser
    # ! This gives a significant (+25 %) speed up
    th._C._jit_set_profiling_executor(False)
    th._C._jit_set_texpr_fuser_enabled(False)
    th._C._jit_set_profiling_mode(False)

    if device == 'cpu':
        # the number of cores is adjusted wrt to the slurm variable if exists
        n_cores=os.getenv('SLURM_JOB_CPUS_PER_NODE')
        # otherwise use the machine cpu count
        if n_cores is None:
            n_cores = os.cpu_count()

        n_cores=int(n_cores)
        logging.info(f'{"CPU cores":<20}: {n_cores}')

        th.set_flush_denormal(True)
        # This is ignored due to the torch._C.jit instructions above
        # torch.jit.enable_onednn_fusion(True)
        th.set_num_interop_threads(n_cores) # Inter-op parallelism
        th.set_num_threads(n_cores) # Intra-op parallelism

        subprocess.call('export OMP_PROC_BIND=spread', shell=True)  # ! VERY IMPORTANT
        subprocess.call('export OMP_PLACES=threads', shell=True)
        subprocess.call('export OMP_SCHEDULE=static', shell=True)   # ! VERY IMPORTANT

        subprocess.call(f'export OMP_NUM_THREADS={n_cores}', shell=True)
        subprocess.call('export KMP_HW_SUBSET=1T', shell=True)
    # ====================== Torchscript JIT parameters ===================== #
    need_to_overfit = False
    alpha_inits = []
    for alpha_init in training_cfg['alpha_inits']:
        if not os.path.exists(os.path.join(workdir, f'fnlic_{alpha_init}.pth')):
            need_to_overfit = True
            alpha_inits.append(alpha_init)
    
    img_t = to_tensor(Image.open(args.input)).unsqueeze(0)

    overfitter_parameter = OverfitterParameter(
                img_shape=img_t.shape[-2:],
                layers_synthesis=layers_synthesis,
                layers_arm=layers_arm,
                n_latents=model_cfg['n_latents'],
                upsampling_kernel_size=model_cfg['upsampling_kernel_size'],
                img_bitdepth=model_cfg['img_bitdepth'],
                latent_bitdepth=model_cfg['latent_bitdepth'],
                freq_precision=16
    )

    prefitter_parameter = PrefitterParameter(
        img_bitdepth=model_cfg['img_bitdepth'],
        prior_arm_width=model_cfg['prefitter_width'],
        prior_arm_depth=model_cfg['prefitter_depth']
    )
    prefitter = Prefitter(prefitter_parameter)
    if os.path.exists(args.prefitter_ckpt):
        try:
            prefitter.load_state_dict(th.load(args.prefitter_ckpt, map_location='cpu'), strict=False)
        except:
            print('Could not load the prefitter checkpoint')
            exit(1)
    else:
        print('Could not find the prefitter checkpoint')
        exit(1)

    if need_to_overfit and args.fnlic == '':
        for alpha_init in alpha_inits:
            logging_path = os.path.join(workdir, f'fnlic_{alpha_init}.log')
            set_logger(logging_path)
            logging.info(args)

            
            
            encoder_manager = EncoderManager(
                preset_name='fnlic',
                start_lr=training_cfg['start_lr'],
                n_loops=training_cfg['n_overfit_loops'],
                n_itr=training_cfg['n_itr'],
            )

            encoder = FNLIC(
                encoder_param=overfitter_parameter,
                encoder_manager=encoder_manager,
                img_t=img_t,
                prefitter=prefitter
            )

            encoder.overfit(device, workdir, alpha_init)
            encoder.save(os.path.join(workdir, f'fnlic_{alpha_init}.pth'))
    
    if args.output != "":
        if args.fnlic == "":
            encoder_paths = glob(os.path.join(workdir, 'fnlic_*.pth'))
            bpd_min = 1e9
            bpd_min_path = ''
            for path in encoder_paths:
                encoder = load_fnlic(os.path.join(workdir, f'fnlic_{alpha_init}.pth'), overfitter_parameter, img_t, prefitter)
                encoder.to_device(device)
                bitstream_path = path.replace('.pth', '.fnlic')
                bpd = fnlic_encode(encoder, bitstream_path, device=device)
                alpha_init = path.split('_')[-1].replace('.pth', '')
                logging_path = os.path.join(workdir, f'fnlic_{alpha_init}.log')
                set_logger(logging_path)
                logging.info(f'BPD: {bpd}')
                if bpd < bpd_min:
                    bpd_min = bpd
                    bpd_min_path = bitstream_path
            shutil.copyfile(bpd_min_path, args.output)
        else:
            encoder = load_fnlic(args.fnlic, overfitter_parameter, img_t, prefitter)
            encoder.to_device(device)
            bpd = fnlic_encode(encoder, args.output, device=device)
    if args.remove_workdir:
        subprocess.call(f'rm -r {workdir}', shell=True)