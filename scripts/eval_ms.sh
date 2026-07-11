export CUDA_VISIBLE_DEVICES="0,1,2,3"

# swift eval \
#     --model Qwen/Qwen2.5-1.5B-Instruct \
#     --eval_backend OpenCompass \
#     --infer_backend vllm \
#     --eval_limit 100 \
#     --eval_dataset gsm8k
#     --adapters swift/test_lora \
#     --model Qwen/Qwen2.5-7B-Instruct \


swift eval \
    --model /home/hrzhu/model/DeepSeek-R1-Distill-Qwen-7B \
    --eval_backend Native \
    --infer_backend vllm \
    --eval_dataset gsm8k \
    --adapters /home/hrzhu/code/long2short/ckpts/DAPO/DAPO-deepseek-Qwen7B/global_step_50/actor/lora_adapter