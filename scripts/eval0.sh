# export CUDA_VISIBLE_DEVICES="4,5,6,7" 
export CUDA_VISIBLE_DEVICES="0,1,2,3"
# export CUDA_VISIBLE_DEVICES="4,5"
export VLLM_USE_V1=0
N_GPUS=4
DATA_PARALLEL_SIZE=${N_GPUS}
GPU_MEMORY_UTILIZATION=0.95
MODEL_PATH="${1:-ckpts/DAPO/DAPO-deepseek-Qwen1.5B/global_step_550/actor_hf}"
last_dir=$(basename "${MODEL_PATH}")

# 检查最后一个文件夹是否为"actor"
if [ "${last_dir}" = "actor" ]; then
    echo "检测到以actor结尾的路径，执行转换脚本..."
    # 运行转换脚本
    bash scripts/convert_ckpt.sh "${MODEL_PATH}"
    
    # 将MODEL_PATH更新为以actor_hf结尾的路径
    MODEL_PATH="${MODEL_PATH}_hf"
    
    echo "已更新模型路径为: ${MODEL_PATH}"
fi

output_path="${MODEL_PATH}/eval"
for seed in 0
do 
    lm_eval --model vllm --model_args pretrained=${MODEL_PATH},tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},data_parallel_size=${DATA_PARALLEL_SIZE},seed=${seed} --tasks math_500,aime,gsm8k_reasoning,minerva_algebra --batch_size auto --output_path ${output_path} --include_path eval_configs/deepseek --seed ${seed} --log_samples --apply_chat_template
    # lm_eval --model vllm --model_args pretrained=${MODEL_PATH},tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},data_parallel_size=${DATA_PARALLEL_SIZE},seed=${seed} --tasks aime --batch_size auto --output_path ${output_path} --include_path eval_configs/deepseek --seed ${seed} --log_samples --apply_chat_template
    # lm_eval --model vllm --model_args pretrained=${MODEL_PATH},tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},data_parallel_size=${DATA_PARALLEL_SIZE},seed=${seed} --tasks math_500,gsm8k_reasoning,minerva_algebra --batch_size auto --output_path ${output_path} --include_path eval_configs/deepseek --seed ${seed} --log_samples --apply_chat_template
done 

bash scripts/eval/eval_all.sh ${MODEL_PATH} ${output_path}
exit 0