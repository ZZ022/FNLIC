import argparse
import subprocess
import glob
import os
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

from bitstream.header import read_header

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

encoding_fmt = 'CUDA_VISIBLE_DEVICES={} ' \
          'python encode.py ' \
          '--workdir {} ' \
          '--prefitter_ckpt {} ' \
          '-i {} ' \
          '-o {} ' \


decoding_fmt = 'CUDA_VISIBLE_DEVICES={} ' \
          'python decode.py ' \
          '-i {} ' \
          '-o {} ' \

def run_command(cmd):
    """ Utility function to run a shell command. """
    process = subprocess.Popen(cmd, shell=True)
    process.wait()

def run_encode(gpu, img_paths, args, max_concurrent_tasks):
    futures = []
    workdir_workspace = os.path.join(args.workdir, 'workspace')
    encoding_dir = os.path.join(args.workdir, 'encoded')
    os.makedirs(encoding_dir, exist_ok=True)
    os.makedirs(workdir_workspace, exist_ok=True)
    
    with ThreadPoolExecutor(max_workers=max_concurrent_tasks) as executor:
        for img_path in img_paths:
            img_name = os.path.basename(img_path).split('.')[0]
            workdir_log = os.path.join(workdir_workspace, img_name)
            os.makedirs(workdir_log, exist_ok=True)
            encoding_cmd = encoding_fmt.format(gpu,
                                               workdir_log,
                                               args.prefitter_ckpt,
                                               img_path,
                                               os.path.join(encoding_dir, f'{img_name}.fnlic'))
            future = executor.submit(run_command, encoding_cmd)
            futures.append(future)
    return futures

def run_decode(gpu, img_paths, args, max_concurrent_tasks):
    futures = []
    workdir_workspace = os.path.join(args.workdir, 'workspace')
    encoding_dir = os.path.join(args.workdir, 'encoded')
    os.makedirs(workdir_workspace, exist_ok=True)
    
    with ThreadPoolExecutor(max_workers=max_concurrent_tasks) as executor:
        for img_path in img_paths:
            img_name = os.path.basename(img_path).split('.')[0]
            workdir_decode = os.path.join(args.workdir, 'decoded')
            os.makedirs(workdir_decode, exist_ok=True)
            decoding_cmd = decoding_fmt.format(
                                    gpu,
                                    os.path.join(encoding_dir, f'{img_name}.fnlic'),
                                    os.path.join(workdir_decode, f'{img_name}_dec.png'))
            future = executor.submit(run_command, decoding_cmd)
            futures.append(future)
    return futures

def parse_workdir(src, workdir):

    results = []
    columns = None

    for img_name in os.listdir(src):
        if img_name.endswith(".png"):
            base_name = img_name[:-4]
            orig_img_path = os.path.join(src, img_name)
            rec_img_path = os.path.join(workdir, "decoded", base_name + "_dec.png")
            orig_img = Image.open(orig_img_path)
            rec_img = Image.open(rec_img_path)

            is_the_same = list(orig_img.getdata()) == list(rec_img.getdata())
            img_size = float(orig_img.size[0] * orig_img.size[1] * 3)

            real_bpd = os.path.getsize(os.path.join(workdir, 'encoded', f'{base_name}.fnlic'))*8/img_size
                
            bitstream_path = os.path.join(workdir, 'encoded', f'{base_name}.fnlic')
            header = read_header(open(bitstream_path, 'rb').read())
            n_bytes_nn_dict = header.get('n_bytes_nn')
            n_bytes_nn = 0
            for v in n_bytes_nn_dict.values():
                for _v in v.values():
                    if _v>0:
                        n_bytes_nn += _v
            n_bpd_nn = n_bytes_nn*8/img_size
            
            n_bytes_per_latents = header.get('n_bytes_per_latent')
            n_bpd_per_latents = [x*8/img_size for x in n_bytes_per_latents]
            n_bpd_latent = sum(n_bpd_per_latents)

            n_byte_subimages = header.get('n_bytes_img')
            n_bpd_subimages = [x*8/img_size for x in n_byte_subimages]
            n_bpd_img = sum(n_bpd_subimages)
            
            result = [base_name, real_bpd, is_the_same, n_bpd_nn, n_bpd_latent, n_bpd_img, *n_bpd_per_latents, *n_bpd_subimages]
            results.append(result)
            if columns is None:
                columns = ['Image', 'Real_BPD', 'Is_Same', 'NN_BPD', 'Latent_BPD', 'Img_BPD']
                columns += [f'Latent_{i}_BPD' for i in range(len(n_bpd_per_latents))]
                columns += [f'Img_{i}_BPD' for i in range(len(n_bpd_subimages))]
    df = pd.DataFrame(results, columns=columns)
    df = df.sort_values(by='Image')
    
    averages = df.mean(numeric_only=True)
    averages['Image'] = 'Average'
    df = df._append(averages, ignore_index=True)
    # set numeric precision to 3
    df = df.round(3)
    
    df.to_csv(os.path.join(workdir, 'results.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run FNLIC overfitting, encoding and decoding')
    parser.add_argument('--workdir', type=str, required=True, help='Work directory')
    parser.add_argument('--prefitter_ckpt', type=str, default='../weight/prefitter.pth', help='Path to the prefitter checkpoint')
    parser.add_argument('--gpus', type=str, required=True, help='Comma-separated list of GPUs to use')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input dataset')
    parser.add_argument('--max_tasks_per_gpu', type=int, default=4, help='Maximum concurrent tasks per GPU')
    parser.add_argument('--parse_results', type=int, default=1, help='whether to parse the results or not')
    args = parser.parse_args()

    if not os.path.exists(args.workdir):
        os.makedirs(args.workdir)
    
    gpus = args.gpus.split(',')
    img_paths = glob.glob(os.path.join(args.input_dir, '*.png'))
    
    num_gpus = len(gpus)
    img_paths_per_gpu = [img_paths[i::num_gpus] for i in range(num_gpus)]

    # Collect all futures from encoding and decoding for each GPU
    all_encode_futures = []
    for i, gpu in enumerate(gpus):
        executor = ThreadPoolExecutor(max_workers=args.max_tasks_per_gpu)
        encode_futures = executor.submit(run_encode, gpu, img_paths_per_gpu[i], args, args.max_tasks_per_gpu)
        all_encode_futures.append(encode_futures)
    
    for future in as_completed(all_encode_futures):
        for task in future.result():
            task.result()
    
    all_decode_futures = []
    for i, gpu in enumerate(gpus):
        executor = ThreadPoolExecutor(max_workers=args.max_tasks_per_gpu)
        decode_futures = executor.submit(run_decode, gpu, img_paths_per_gpu[i], args, args.max_tasks_per_gpu)
        all_decode_futures.append(decode_futures)
    
    for future in as_completed(all_decode_futures):
        for task in future.result():
            task.result()

    # Now that all tasks are done, run the parsing command
    # parse_result
    if args.parse_results:
        parse_workdir(args.input_dir, args.workdir)