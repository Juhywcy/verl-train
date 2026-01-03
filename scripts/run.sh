#!/bin/bash

# bash verl/recipe/sfmx/dapo_grad_damping.sh 
# bash verl/recipe/sfmx/dapo.sh

# while true; do
#     bash scripts/wait.sh
# done

while true; do
    val=$(cat ckpts/DAPO/DAPO-deepseek-Qwen1.5B-100step-grad-damping-eos/latest_checkpointed_iteration.txt)
    if [ "$val" -eq 300 ]; then
        echo "Value is 300. Stopping."
        break
    fi
    bash scripts/wait.sh
done
