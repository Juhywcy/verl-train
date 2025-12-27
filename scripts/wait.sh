#!/bin/bash

# GPU_IDS=(0 1 2 3)    
GPU_IDS=(4 5 6 7)            # 要监控的GPU编号（从0开始）
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

        

        # bash scripts/eval_lora.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-alpha32/global_step_400/actor_hf
        # bash scripts/eval_verl.sh /home/hrzhu/model/DeepSeek-R1-Distill-Qwen-7B
        # bash scripts/eval_verl.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-full/global_step_50/actor_hf
        # bash scripts/eval_verl.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-full/global_step_100/actor_hf
        # bash scripts/eval_verl.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-full/global_step_150/actor_hf
        # bash scripts/eval_verl.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-full/global_step_200/actor_hf
        # bash scripts/eval_verl.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-full/global_step_250/actor_hf
        # bash scripts/eval_verl.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-full/global_step_300/actor_hf
        # bash scripts/eval_verl.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-full/global_step_350/actor_hf
        # bash scripts/eval_verl.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-full/global_step_400/actor_hf
        # bash scripts/eval_verl.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-full/global_step_450/actor_hf
        # bash scripts/convert_ckpt.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-full/global_step_500/actor
        # bash verl/recipe/l2s/eval_aime24.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-full/global_step_400/actor_hf
        # bash verl/recipe/l2s/eval_aime24.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-full/global_step_450/actor_hf
        # bash verl/recipe/l2s/eval_aime24.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-full/global_step_500/actor_hf
        # bash verl/recipe/l2s/eval_aime24.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-full/global_step_600/actor_hf

        # bash scripts/eval_verl.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_50/actor_hf
        # bash scripts/eval_verl.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_100/actor_hf
        # bash scripts/eval_verl.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_150/actor_hf
        # bash scripts/eval_verl.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_200/actor_hf
        # bash scripts/eval_verl.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_250/actor_hf
        # bash scripts/eval_verl.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_300/actor_hf
        # bash scripts/eval_verl.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_350/actor_hf
        # bash scripts/eval_verl.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_400/actor_hf
        # bash scripts/eval_verl.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_450/actor_hf
        
        # bash scripts/eval.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_50/actor_hf 
        # bash scripts/eval.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_100/actor_hf 
        # bash scripts/eval.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_150/actor_hf 
        # bash scripts/eval.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_200/actor_hf 
        # bash scripts/eval.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_250/actor_hf 
        # bash scripts/eval.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_300/actor_hf 
        # bash scripts/eval.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_350/actor_hf 
        # bash scripts/eval.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_400/actor_hf 
        # bash scripts/eval.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_450/actor_hf
        # bash scripts/eval.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_500/actor_hf
        # bash verl/recipe/l2s/eval_amc.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_500/actor_hf
        # bash verl/recipe/l2s/eval_olympiad_bench.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_500/actor_hf
        # bash scripts/eval.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_550/actor_hf
        # bash verl/recipe/l2s/eval_amc.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_550/actor_hf
        # bash verl/recipe/l2s/eval_olympiad_bench.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_550/actor_hf
        # bash scripts/eval0.sh ckpts/Laser-DE-L4096-7B
        # bash verl/recipe/l2s/eval_amc.sh ckpts/Laser-DE-L4096-7B
        # bash verl/recipe/l2s/eval_olympiad_bench.sh ckpts/Laser-DE-L4096-7B
        # bash scripts/eval0.sh ckpts/Skywork-7B-AutoThink-Stage3
        # bash verl/recipe/l2s/eval_amc.sh ckpts/Skywork-7B-AutoThink-Stage3
        # bash verl/recipe/l2s/eval_olympiad_bench.sh ckpts/Skywork-7B-AutoThink-Stage3
        # bash scripts/eval0.sh ckpts/LCR1_7B
        # bash verl/recipe/l2s/eval_amc.sh ckpts/LCR1_7B
        # bash verl/recipe/l2s/eval_olympiad_bench.sh ckpts/LCR1_7B
        # bash scripts/eval0.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2-logic/global_step_400/actor_hf
        # bash verl/recipe/l2s/eval_amc.sh ckpts/Qwen2.5-7B-Instruct
        # bash verl/recipe/l2s/eval_olympiad_bench.sh ckpts/Qwen2.5-7B-Instruct
        # bash scripts/eval0.sh ckpts/Qwen2.5-Math-7B
        # bash verl/recipe/l2s/eval_amc.sh ckpts/Qwen2.5-Math-7B
        # bash verl/recipe/l2s/eval_olympiad_bench.sh ckpts/Qwen2.5-Math-7B
        # bash scripts/eval0.sh ckpts/Qwen2.5-Math-7B-Instruct
        # bash verl/recipe/l2s/eval_amc.sh ckpts/Qwen2.5-Math-7B-Instruct
        # bash verl/recipe/l2s/eval_olympiad_bench.sh ckpts/Qwen2.5-Math-7B-Instruct
        # bash scripts/eval.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_600/actor_hf
        # bash verl/recipe/l2s/eval_amc.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_600/actor_hf
        # bash verl/recipe/l2s/eval_olympiad_bench.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_600/actor_hf
        # bash scripts/eval.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_650/actor_hf
        # bash verl/recipe/l2s/eval_amc.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_650/actor_hf
        # bash verl/recipe/l2s/eval_olympiad_bench.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_650/actor_hf
        # bash verl/recipe/l2s/eval_olympiad_bench.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_650/actor_hf
        # bash verl/recipe/l2s/eval_amc.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_700/actor_hf
        # bash verl/recipe/l2s/eval_olympiad_bench.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_700/actor_hf
        # bash scripts/eval.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_700/actor_hf
        # bash verl/recipe/l2s/eval_amc.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_400/actor_hf 0.8
        # bash verl/recipe/l2s/eval_amc.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_400/actor_hf 0.7
        # bash verl/recipe/l2s/eval_amc.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_400/actor_hf 0.6
        # bash verl/recipe/l2s/eval_amc.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_450/actor_hf 0.8
        # bash verl/recipe/l2s/eval_amc.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_450/actor_hf 0.7
        # bash verl/recipe/l2s/eval_amc.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_450/actor_hf 0.6
        # bash verl/recipe/l2s/eval_amc.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_500/actor_hf 0.8
        # bash verl/recipe/l2s/eval_amc.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_500/actor_hf 0.7
        # bash verl/recipe/l2s/eval_amc.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_500/actor_hf 0.6
        # bash verl/recipe/l2s/eval_olympiad_bench.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_400/actor_hf 0.85
        # bash verl/recipe/l2s/eval_olympiad_bench.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_400/actor_hf 0.75
        # bash verl/recipe/l2s/eval_olympiad_bench.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_400/actor_hf 0.65
        # bash verl/recipe/l2s/eval_olympiad_bench.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_400/actor_hf 0.8
        # bash verl/recipe/l2s/eval_olympiad_bench.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_400/actor_hf 0.7
        # # bash verl/recipe/l2s/eval_olympiad_bench.sh ckpts/DAPO/DAPO-deepseek-Qwen7B-stage2/global_step_400/actor_hf 0.6
        # bash verl/recipe/l2s/deepseek_qwen1.5B_total.sh
        # bash verl/recipe/l2s/deepseek_qwen1.5B_stage1_exp.sh
        # bash verl/recipe/l2s/deepseek_qwen7B_stage2-logic.sh
        # bash verl/recipe/sfmx/dapo_grad_trunc.sh 
        # bash verl/recipe/sfmx/dapo.sh
        bash verl/recipe/sfmx/dapo_grad_damping_eos.sh        
        break
    else
        echo "GPU $GPU_ID 当前显存 ${USED_MEM} MiB，大于等于阈值 ${THRESHOLD} MiB，等待中..."
        sleep $SLEEP_INTERVAL
    fi
done
