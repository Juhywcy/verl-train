
#!/bin/bash

# 输出的 Excel 文件名，你可以根据需要修改
OUTPUT_EXCEL="stage1_results.xlsx"

# 构建不同 step 下的 model path 数组
PATHS=()
for ((step=20; step<=200; step+=20)); do
    PATHS+=("/home/verl-train/ckpts/DAPO/DAPO-Qwen3-4B-stage1-4096/global_step_${step}/actor_hf")
done

# 运行 eval_excel_paths.py 把各个路径传过去
python scripts/eval/eval_excel_paths.py "$OUTPUT_EXCEL" "${PATHS[@]}"