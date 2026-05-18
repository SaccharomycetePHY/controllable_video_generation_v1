#!/bin/bash

# 记录日志的目录
LOG_DIR="./logs"
mkdir -p $LOG_DIR  # 如果日志目录不存在，则创建

# Python 脚本路径
PYTHON_BIN="/home/zyude/anaconda3/envs/smplerx/bin/python"

# Python 脚本名称
SCRIPT="inference_video_nobkg.py"


ROOT_PATH="/amax/zyude/human_video_data/1643718_1"

# 启动的总任务数
TOTAL_TASKS=10


NUM_GPUS=7

for i in $(seq 0 $((TOTAL_TASKS - 1)))
do

    GPU_ID=$((i % NUM_GPUS))

  
    CUT_ID=$i

    
    LOG_FILE="$LOG_DIR/task_${CUT_ID}_gpu_${GPU_ID}.log"

 
    nohup $PYTHON_BIN $SCRIPT --root_path $ROOT_PATH --gpu $GPU_ID --cut $CUT_ID > $LOG_FILE 2>&1 &

    echo "Started task ${CUT_ID} on GPU ${GPU_ID}, log: ${LOG_FILE}"
done

