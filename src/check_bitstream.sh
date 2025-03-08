#!/bin/bash

if [ $# -ne 3 ]; then
    echo "Usage: $0 <gpus> <dataset_name> <max_tasks_per_gpu>"
    echo "Example: bash $0 0,1 kodak 4"
    exit 1
fi

gpus=$1
dataset_name=$2
max_tasks_per_gpu=$3

mkdir -p ../test

python test_dec.py \
    --gpus "$gpus" \
    --input_bitstream_dir "../encoded_bitstreams/${dataset_name}" \
    --input_image_dir "../datasets/${dataset_name}" \
    --dec_dst "../test/test_dec_${dataset_name}" \
    --max_tasks_per_gpu "$max_tasks_per_gpu" \
    --log_dst "../test/${dataset_name}_bitstream_check.csv"