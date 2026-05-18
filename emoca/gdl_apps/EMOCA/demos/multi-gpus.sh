#!/bin/bash

for gpu_id in 
do
    CUDA_VISIBLE_DEVICES=$gpu_id python demos/test_emoca_on_video_queue_10_9.py --gpu_id $gpu_id &
done

wait
