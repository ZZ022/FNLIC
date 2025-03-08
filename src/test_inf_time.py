import os
import yaml
import argparse
import torch as th
import numpy as np

from PIL import Image
from glob import glob
from torchvision.transforms.functional import to_tensor

from models.prefitter import PrefitterParameter, Prefitter
from models.overfitter import OverfitterParameter
from models.fnlic import load_fnlic

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Run FNLIC Inference Time Test')
    parser.add_argument('--input_model_dir', type=str, required=True, help='Path to the overfitted model directory')
    parser.add_argument('--input_image_dir', required=True, help='Path to the input image. png (RGB444)', type=str)
    parser.add_argument('--prefitter_ckpt', help='Path to the prefitter checkpoint', type=str, default='../weight/prefitter.pth')

    parser.add_argument('--model_config', help='Path to the model configuration file', type=str, default='config/model_cfg.yaml')

    parser.add_argument('--log_dst', type=str, default="", help='Path to the output log file, if not provided, no log will be saved')

    parser.add_argument('--num_tests', type=int, default=5, help='Number of tests to run')
    
    args = parser.parse_args()

    with open(args.model_config, 'r') as f:
        model_cfg = yaml.safe_load(f)
    
    image_paths = glob(os.path.join(args.input_image_dir, '*.png'))

    assert len(image_paths) > 0, 'No images found in the input image directory'

    layers_synthesis = [x for x in model_cfg['layers_synthesis'].split(',') if x != '']
    layers_arm = [int(x) for x in model_cfg['layers_arm'].split(',') if x != '']

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

    device = 'cuda' if th.cuda.is_available() else 'cpu'

    results = []
    # the first loop is for the warm-up
    for i in range(args.num_tests+1):
        for image_path in image_paths:
            img_t = to_tensor(Image.open(image_path)).unsqueeze(0)
            img_basename = os.path.basename(image_path).split('.')[0]
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
            encoder = load_fnlic(os.path.join(args.input_model_dir, f'{img_basename}_fnlic.pth'), overfitter_parameter, img_t, prefitter)
            encoder.to_device(device)
            encoder.set_to_eval()
            t = encoder.test_inference_time()
            if i > 0:
                results.append(t)
                if args.log_dst != "":
                    with open(args.log_dst, 'a') as f:
                        f.write(f'{img_basename},{t}\n')
    results_np = np.array(results)
    print(f'Average Inference Time: {results_np.mean()}')
    print(f'Std Inference Time: {results_np.std()}')
    if args.log_dst != "":
        with open(args.log_dst, 'a') as f:
            f.write(f'Average,{results_np.mean()}\n')
            f.write(f'Std,{results_np.std()}\n')
        


        