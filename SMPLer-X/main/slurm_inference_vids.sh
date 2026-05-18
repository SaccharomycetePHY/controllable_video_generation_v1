#!/usr/bin/env bash
set -x

PARTITION=Zoetrope

INPUT_DIR=/amax/zyude/video_mimo/BV1U4421X7sU/
OUTPUT_DIR=/amax/zejian/DATA/video_mimo/

GPUS=4
JOB_NAME=inference
CKPT=smpler_x_h32

GPUS_PER_NODE=$((${GPUS}<8?${GPUS}:8))
CPUS_PER_TASK=4 # ${CPUS_PER_TASK:-2}
SRUN_ARGS=${SRUN_ARGS:-""}

# 遍历处理所有的 clip.mp4 文件
for clip in ${INPUT_DIR}/*/clip.mp4; do
    CLIP_NAME=$(basename $(dirname "$clip"))
    SAVE_DIR=${OUTPUT_DIR}/${CLIP_NAME}

    mkdir -p ${SAVE_DIR}/images
    ffmpeg -i "$clip" -f image2 -vf fps=30 -qscale 0 "${SAVE_DIR}/images/%06d.jpg"

    end_count=$(find "$SAVE_DIR/images" -type f | wc -l)
    echo "Processed $CLIP_NAME: $end_count frames"

    # inference
    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    python inference.py \
        --num_gpus ${GPUS_PER_NODE} \
        --exp_name output/demo_${JOB_NAME} \
        --pretrained_model ${CKPT} \
        --agora_benchmark agora_model \
        --img_path ${SAVE_DIR}/images \
        --start 1 \
        --end  $end_count \
        --output_folder ${SAVE_DIR} \
        --show_verts \
        --show_bbox \
        --save_mesh \
        --iou_thr 0.2 \
        --bbox_thr 20

    # images to video
    # ffmpeg -y -f image2 -r 30 -i ${SAVE_DIR}/%06d.jpg -c:v libx264 -strict -2 -pix_fmt yuv420p ${SAVE_DIR}/smpl.mp4
done