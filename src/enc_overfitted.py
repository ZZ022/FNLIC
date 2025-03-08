import argparse
import subprocess
import glob
import os

from concurrent.futures import ThreadPoolExecutor, as_completed

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

encoding_fmt = 'CUDA_VISIBLE_DEVICES={} ' \
          'python encode.py ' \
          '-i {} ' \
          '-o {} ' \
          '-f {} ' \


def run_command(cmd):
    """ Utility function to run a shell command. """
    process = subprocess.Popen(cmd, shell=True)
    process.wait()

def run_encode(gpu, img_paths, src, dst, max_concurrent_tasks):
    futures = []
    os.makedirs(dst, exist_ok=True)
    
    with ThreadPoolExecutor(max_workers=max_concurrent_tasks) as executor:
        for img_path in img_paths:
            img_name = os.path.basename(img_path).split('.')[0]
            encoding_cmd = encoding_fmt.format(gpu,
                                               img_path,
                                               os.path.join(dst, f'{img_name}.fnlic'),
                                               os.path.join(src, f'{img_name}_fnlic.pth'))
            future = executor.submit(run_command, encoding_cmd)
            futures.append(future)
    return futures

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run FNLIC decoding')
    parser.add_argument('--gpus', type=str, required=True, help='Comma-separated list of GPUs to use')
    parser.add_argument('--input_model_dir', type=str, required=True, help='Path to the overfitted model directory')
    parser.add_argument('--input_image_dir', type=str, default='', help='Path to the input image directory')
    parser.add_argument('--dst', type=str, required=True, help='Path to the output decoded image directory')
    parser.add_argument('--max_tasks_per_gpu', type=int, default=4, help='Maximum concurrent tasks per GPU')
    args = parser.parse_args()
    
    gpus = args.gpus.split(',')
    paths = glob.glob(os.path.join(args.input_image_dir, '*.png'))
    
    num_gpus = len(gpus)
    paths_per_gpu = [paths[i::num_gpus] for i in range(num_gpus)]
    
    all_encode_futures = []
    for i, gpu in enumerate(gpus):
        executor = ThreadPoolExecutor(max_workers=args.max_tasks_per_gpu)
        encode_futures = executor.submit(run_encode, gpu, paths_per_gpu[i], args.input_model_dir, args.dst, args.max_tasks_per_gpu)
        all_encode_futures.append(encode_futures)
    
    for future in as_completed(all_encode_futures):
        for task in future.result():
            task.result()