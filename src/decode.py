
import os
import torch as th
import numpy as np
import subprocess
import argparse
import sys
import logging

from PIL import Image
from models.prefitter import PrefitterParameter, Prefitter
from bitstream.decode import fnlic_decode
from utils.misc import get_best_device

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

if __name__ == '__main__':
    # =========================== Parse arguments =========================== #
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Path to the bitstream', type=str)
    parser.add_argument('-o', '--output', type=str, help='Output image path.')
    parser.add_argument('--prefitter_width', required=True, help='prefitter width', type=int, default=32)
    parser.add_argument('--prefitter_depth', help='prefitter depth', type=int, default=3)
    parser.add_argument('--prefitter_ckpt', help='Path to prefitter checkpoint', type=str, default='../weight/prefitter.pth')
    # =========================== Parse arguments =========================== #
    args = parser.parse_args(sys.argv[1:])

    device = get_best_device()

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

    param = PrefitterParameter(
        prior_arm_depth=args.prefitter_depth,
        prior_arm_width=args.prefitter_width
    )
    
    prefitter = Prefitter(param)
    prefitter.load_state_dict(th.load(args.prefitter_ckpt, map_location='cpu'), strict=False)
    prefitter.to_device(device)
    prefitter.eval()

    img_rec = fnlic_decode(args.input, prefitter, device=device)
    img_rec = fnlic_decode(args.input, prefitter, device=device)
    img = th.round(img_rec * 255.0).squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    Image.fromarray(img).save(args.output)