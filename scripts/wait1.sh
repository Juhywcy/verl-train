#!/bin/bash

GPU_IDS=(0 1 2 3)    
# GPU_IDS=(4 5 6 7)            # 要监控的GPU编号（从0开始）
THRESHOLD=1500           # 显存占用低于该值(MiB)时视为“空闲”
SLEEP_INTERVAL=100      # 每隔几秒检查一次
echo "监控 GPU $GPU_ID，显存小于 ${THRESHOLD} MiB 时自动启动程序..."

while true; do
    
    all_idle=true
    for gpu_id in "${GPU_IDS[@]}"; do
        # 获取当前GPU的显存使用量
        USED_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$((gpu_id + 1))p")
        
        # 检查当前GPU是否空闲
        if [ "$USED_MEM" -ge "$THRESHOLD" ]; then
            echo "GPU $gpu_id 当前显存 ${USED_MEM} MiB，大于等于阈值 ${THRESHOLD} MiB，不满足条件"
            all_idle=false
        else
            echo "GPU $gpu_id 当前显存 ${USED_MEM} MiB，满足空闲条件"
        fi
    done
    if [ "$all_idle" = true ]; then
        echo "所有监控的GPU都满足条件，启动程序..."

        

       
        # bash verl/recipe/l2s/qwen3_4B.sh   
        # bash scripts/eval0.sh /home/verl-train/ckpts/DAPO/DAPO-Qwen3-4B-stage2-16384/global_step_380/actor_hf
       bash verl/recipe/ig/grpo.sh
    else
        echo "GPU $GPU_ID 当前显存 ${USED_MEM} MiB，大于等于阈值 ${THRESHOLD} MiB，等待中..."
        sleep $SLEEP_INTERVAL
    fi
done
