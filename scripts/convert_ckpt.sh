local_dir="${1:-ckpts/DAPO/DAPO-deepseek-Qwen1.5B-stage2/global_step_600/actor}"
target_dir="${local_dir%/actor}/actor_hf"

echo $target_dir
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $local_dir \
    --target_dir $target_dir
