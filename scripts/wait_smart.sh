#!/bin/bash

# 智能 GPU 选择脚本
# 自动识别空闲 GPU 并选择指定数量的卡运行程序

# ==================== 配置参数 ====================
THRESHOLD=1500           # 显存占用低于该值(MiB)时视为"空闲"
SLEEP_INTERVAL=100       # 检查间隔（秒）
REQUIRED_GPUS=4          # 需要多少个空闲 GPU 才启动程序（可调整）
COMMAND="bash scripts/eval.sh /home/verl-train/ckpts/DAPO/DAPO-Qwen3-4B-stage1-4096/global_step_20/actor"
# ===================================================

echo "=========================================="
echo "智能 GPU 监控脚本"
echo "显存阈值: ${THRESHOLD} MiB"
echo "检查间隔: ${SLEEP_INTERVAL} 秒"
echo "需要 GPU 数量: ${REQUIRED_GPUS}"
echo "=========================================="

# 获取 GPU 总数
TOTAL_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "检测到 ${TOTAL_GPUS} 个 GPU"

while true; do
    # 获取所有 GPU 的显存使用情况
    # nvidia-smi 输出第一行是标题，从第二行开始是数据
    # 使用 mapfile 读取所有 GPU 的显存信息到数组
    mapfile -t USED_MEMS < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

    echo ""
    echo "【$(date '+%Y-%m-%d %H:%M:%S')】检查 GPU 状态..."

    # 找出空闲的 GPU（显存 < 阈值）
    IDLE_GPUS=()
    for gpu_id in "${!USED_MEMS[@]}"; do
        used_mem="${USED_MEMS[$gpu_id]}"
        if [ "$used_mem" -lt "$THRESHOLD" ]; then
            echo "  ✓ GPU ${gpu_id}: ${used_mem} MiB (空闲)"
            IDLE_GPUS+=("$gpu_id")
        else
            echo "  ✗ GPU ${gpu_id}: ${used_mem} MiB (占用)"
        fi
    done

    IDLE_COUNT=${#IDLE_GPUS[@]}
    echo "空闲 GPU 数量: ${IDLE_COUNT}/${TOTAL_GPUS}"

    # 判断是否有足够空闲 GPU
    if [ "$IDLE_COUNT" -ge "$REQUIRED_GPUS" ]; then
        # 选择前 REQUIRED_GPUS 个空闲 GPU
        SELECTED_GPUS=("${IDLE_GPUS[@]:0:$REQUIRED_GPUS}")
        SELECTED_GPUS_STR=$(IFS=,; echo "${SELECTED_GPUS[*]}")

        echo "=========================================="
        echo "✓ 找到 ${IDLE_COUNT} 个空闲 GPU，选择前 ${REQUIRED_GPUS} 个: ${SELECTED_GPUS_STR}"
        echo "设置 CUDA_VISIBLE_DEVICES=${SELECTED_GPUS_STR}"
        echo "启动程序..."
        echo "=========================================="

        # 设置 CUDA_VISIBLE_DEVICES 并执行命令
        CUDA_VISIBLE_DEVICES="${SELECTED_GPUS_STR}" bash -c "$COMMAND"

        echo "程序执行完成"
        break
    else
        echo "⚠ 空闲 GPU 不足 ${REQUIRED_GPUS} 个，继续等待..."
        echo "将在 ${SLEEP_INTERVAL} 秒后再次检查..."
        sleep $SLEEP_INTERVAL
    fi
done
