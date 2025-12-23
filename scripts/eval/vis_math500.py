import json
import re
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# 你的 math_eval 脚本路径
sys.path.insert(0, os.path.abspath("scripts/eval/math_eval"))
from math_eval import grade_answer

import transformers


# ========================
#   工具函数
# ========================

def extract_all_boxed_content(text):
    """提取 \\boxed{} 中的内容，用于解析数学答案"""
    results = []
    start = 0

    while True:
        start = text.find(r"\boxed{", start)
        if start == -1:
            break

        brace_count = 0
        result = []
        i = start

        while i < len(text):
            ch = text[i]
            result.append(ch)

            if ch == "{":
                brace_count += 1
            elif ch == "}":
                brace_count -= 1

            if brace_count == 0 and result[-1] == "}":
                break
            i += 1

        results.append("".join(result))
        start = i + 1

    return results


def count_reasoning_steps(text: str) -> int:
    """以两个换行符 \\n\\n 划分推理步骤"""
    if not text:
        return 0
    parts = text.split("\n\n")
    parts = [p for p in parts if p.strip() != ""]
    return len(parts)


# ========================
#   主函数：评估 Math-500
# ========================

def evaluate_math500(tokenizer_path, jsonl_file_path):
    """
    输入 tokenizer_path 和 jsonl 文件路径
    输出一个 list，每个元素结构为：

    {
        "correct": bool,
        "tokens": int,
        "steps": int,
        "is_long": bool
    }
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

    results = []

    # 单文件 or 文件夹
    if os.path.isdir(jsonl_file_path):
        json_files = [
            os.path.join(jsonl_file_path, f)
            for f in os.listdir(jsonl_file_path)
            if f.endswith(".jsonl") and "samples_math_500" in f
        ]
    else:
        json_files = [jsonl_file_path]

    # ========================
    #   遍历文件
    # ========================
    for jf in json_files:
        with open(jf, "r") as file:
            for line in file:
                entry = json.loads(line.strip())

                solution = entry["resps"][0][0]
                expected_answer = entry.get("target", "")
                prompt = entry["arguments"]["gen_args_0"]["arg_0"]

                # 是否包含长思维
                is_long = ("<think>" in solution) or ("<think>" in prompt)

                # 统计 steps
                steps = count_reasoning_steps(solution)

                # 统计 tokens（系统提示 + 用户 + assistant 回复）
                conv = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": solution},
                ]
                tokens = tokenizer.apply_chat_template(conv, return_tensors="pt").shape[1]

                # ========================
                #   解析最终预测结果
                # ========================
                prediction = None

                boxed_matches = extract_all_boxed_content(str(solution))
                if boxed_matches:
                    last = boxed_matches[-1]
                    if "\\boxed" in last:
                        prediction = last.replace("\\boxed{", "")[:-1]
                else:
                    # 回退方案
                    patterns = [
                        r"<answer>(.*?)</answer>",
                        r"</answer>(.*?)</answer>",
                        r"<answer>(.*?)<answer>",
                        r"\*\*Answer:\*\* ([\d\.]+)",
                        r"[-+]?\d*\.\d+|\d+",
                    ]
                    for p in patterns:
                        m = re.findall(p, str(solution))
                        if m:
                            prediction = m[-1]
                            break

                # ========================
                #   判断是否回答正确
                # ========================
                correct = False

                if prediction is not None:
                    # 主判断
                    if grade_answer(prediction, expected_answer):
                        correct = True
                    else:
                        # 仅比较数字
                        pn = re.findall(r"[-+]?\d*\.\d+|\d+", prediction)
                        en = re.findall(r"[-+]?\d*\.\d+|\d+", expected_answer)
                        if pn and en and float(pn[0]) == float(en[0]):
                            correct = True

                # 保存记录
                results.append({
                    "correct": correct,
                    "tokens": tokens,
                    "steps": steps,
                    "is_long": is_long
                })

    return results

def visualize_compression_effects(compressed_results, original_results, save_dir="."):
    """
    可视化压缩模型 vs 原始模型的 Token 和 Step 变化，基于四种正确性分类。
    """
    
    # 1. 数据对齐与构建 DataFrame
    data = []
    
    for i, (comp, orig) in enumerate(zip(compressed_results, original_results)):
        # 定义四种组别
        if orig['correct'] and comp['correct']:
            group = "Preserved (✓ -> ✓)" # 原本正确，压缩后也正确
        elif orig['correct'] and not comp['correct']:
            group = "Lost (✓ -> ✗)"      # 原本正确，压缩后错误 (性能损失)
        elif not orig['correct'] and comp['correct']:
            group = "Gained (✗ -> ✓)"    # 原本错误，压缩后正确 (意外提升)
        else:
            group = "Failed (✗ -> ✗)"    # 原本错误，压缩后也错误
            
        data.append({
            "ID": i,
            "Group": group,
            "Orig_Tokens": orig['tokens'],
            "Comp_Tokens": comp['tokens'],
            "Orig_Steps": orig['steps'],
            "Comp_Steps": comp['steps'],
            "Token_Reduction": 1 - (comp['tokens'] / orig['tokens'] if orig['tokens'] > 0 else 0)
        })
        
    df = pd.DataFrame(data)
    
    # 设置绘图风格
    sns.set_theme(style="whitegrid")
    
    # 定义颜色映射
    palette = {
        "Preserved (✓ -> ✓)": "#2ca02c", # Green
        "Lost (✓ -> ✗)": "#d62728",      # Red
        "Gained (✗ -> ✓)": "#1f77b4",    # Blue
        "Failed (✗ -> ✗)": "#7f7f7f"     # Gray
    }

    # ==========================================
    # 图表 1: 散点图对比 (Scatter Plot)
    # ==========================================
    # 使用解包赋值，避免 axes,[object Object], 索引写法
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # 子图 1: Tokens 对比
    sns.scatterplot(data=df, x="Orig_Tokens", y="Comp_Tokens", hue="Group", palette=palette, alpha=0.6, ax=ax1)
    # 画对角线
    max_val = max(df["Orig_Tokens"].max(), df["Comp_Tokens"].max())
    ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label="No Change")
    ax1.set_title("Token Count: Original vs Compressed")
    ax1.set_xlabel("Original Model Tokens")
    ax1.set_ylabel("Compressed Model Tokens")
    
    # 子图 2: Steps 对比
    sns.scatterplot(data=df, x="Orig_Steps", y="Comp_Steps", hue="Group", palette=palette, alpha=0.6, ax=ax2)
    max_step = max(df["Orig_Steps"].max(), df["Comp_Steps"].max())
    ax2.plot([0, max_step], [0, max_step], 'k--', alpha=0.5, label="No Change")
    ax2.set_title("Reasoning Steps: Original vs Compressed")
    ax2.set_xlabel("Original Model Steps")
    ax2.set_ylabel("Compressed Model Steps")
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/scatter_comparison.pdf", dpi=300)
    print(f"Saved scatter plot to {save_dir}/scatter_comparison.png")
    plt.show()

    # ==========================================
    # 图表 2: 箱线图分布 (Box Plot)
    # ==========================================
    
    # 转换数据格式
    df_melted_tokens = df.melt(id_vars=["ID", "Group"], value_vars=["Orig_Tokens", "Comp_Tokens"], 
                               var_name="Model", value_name="Count")
    df_melted_steps = df.melt(id_vars=["ID", "Group"], value_vars=["Orig_Steps", "Comp_Steps"], 
                              var_name="Model", value_name="Count")
    
    # 使用解包赋值，ax_top 对应第一个图，ax_bottom 对应第二个图
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Token 分布
    sns.boxplot(data=df_melted_tokens, x="Group", y="Count", hue="Model", ax=ax_top, showfliers=False)
    ax_top.set_title("Distribution of Token Counts by Group")
    ax_top.set_ylabel("Token Count")
    
    # Step 分布
    sns.boxplot(data=df_melted_steps, x="Group", y="Count", hue="Model", ax=ax_bottom, showfliers=False)
    ax_bottom.set_title("Distribution of Reasoning Steps by Group")
    ax_bottom.set_ylabel("Step Count")
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/boxplot_distribution.pdf", dpi=300)
    print(f"Saved box plot to {save_dir}/boxplot_distribution.png")
    plt.show()
    
    # ==========================================
    # 统计摘要
    # ==========================================
    print("\n=== Compression Statistics by Group ===")
    summary = df.groupby("Group").agg({
        "ID": "count",
        "Orig_Tokens": "mean",
        "Comp_Tokens": "mean",
        "Token_Reduction": "mean"
    }).rename(columns={"ID": "Count", "Token_Reduction": "Avg Reduction Rate"})
    print(summary)

def visualize_compression_effects_(compressed_results, original_results, save_dir="."):
    """
    对比压缩模型和原始模型的推理长度（tokens）与推理步骤（steps）。
    
    输出图包括：
      1. Tokens vs Tokens 散点图（带对角线）
      2. Steps vs Steps 散点图
      3. Token Boxplot（单独保存）
      4. Step Boxplot（单独保存）
    """

    os.makedirs(save_dir, exist_ok=True)

    # -----------------------------
    # 1. 构造 DataFrame
    # -----------------------------
    data = []

    for i, (comp, orig) in enumerate(zip(compressed_results, original_results)):
        if orig['correct'] and comp['correct']:
            group = "(✓ → ✓)"
        elif orig['correct'] and not comp['correct']:
            group = "(✓ → ✗)"
        elif not orig['correct'] and comp['correct']:
            group = "(✗ → ✓)"
        else:
            group = "(✗ → ✗)"

        data.append({
            "ID": i,
            "Group": group,
            "Orig_Tokens": orig["tokens"],
            "Comp_Tokens": comp["tokens"],
            "Orig_Steps": orig["steps"],
            "Comp_Steps": comp["steps"],
            "Token_Reduction": 1 - (comp["tokens"] / orig["tokens"] if orig["tokens"] > 0 else 0),
        })

    df = pd.DataFrame(data)

    # -----------------------------
    # 2. 美观色板
    # -----------------------------
    palette = {
        "(✓ → ✓)": "#4CAF50",   # 绿色
        "(✓ → ✗)": "#E53935",        # 红色
        "(✗ → ✓)": "#1E88E5",      # 蓝色
        "(✗ → ✗)": "#8E8E8E",      # 灰紫色
    }

    sns.set_theme(style="whitegrid", font="DejaVu Sans")

    # -----------------------------
    # 3. 加上 (n=xxx)
    # -----------------------------
    counts = df.groupby("Group")["ID"].count().to_dict()
    df["Group_N"] = df["Group"].apply(lambda g: f"{g}\n(n={counts[g]})")


    # -----------------------------
    # 5. Token Boxplot（单独保存）
    # -----------------------------
    df_melt_tok = df.melt(
        id_vars=["ID", "Group_N"],
        value_vars=["Orig_Tokens", "Comp_Tokens"],
        var_name="Model",
        value_name="Value"
    )

    plt.figure(figsize=(12, 8))
    sns.boxplot(
        data=df_melt_tok,
        x="Group_N",
        y="Value",
        hue="Model",
        showfliers=False,
        palette=["#9E9E9E", "#4C9AFF"]
    )
    plt.ylabel("Token Count", fontweight="bold", fontsize=28)
    plt.xlabel("")
    plt.yticks(fontsize=24)
    plt.xticks(fontsize=24)
    plt.legend(fontsize=28, title_fontsize=24)
    # plt.title("Token Distribution by Group", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/token_distribution.png", dpi=300)
    plt.savefig(f"{save_dir}/token_distribution.pdf")
    plt.show()

    # -----------------------------
    # 6. Step Boxplot（单独保存）
    # -----------------------------
    df_melt_step = df.melt(
        id_vars=["ID", "Group_N"],
        value_vars=["Orig_Steps", "Comp_Steps"],
        var_name="Model",
        value_name="Value"
    )

    plt.figure(figsize=(12, 8))
    sns.boxplot(
        data=df_melt_step,
        x="Group_N",
        y="Value",
        hue="Model",
        showfliers=False,
        palette=["#9E9E9E", "#4C9AFF"]
    )
    plt.ylabel("Reasoning Steps", fontweight="bold", fontsize=28)
    plt.xlabel("")
    plt.yticks(fontsize=24)
    plt.xticks(fontsize=24)
    plt.legend(fontsize=28, title_fontsize=24)
    # plt.title("Reasoning Steps Distribution by Group", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/step_distribution.png", dpi=300)
    plt.savefig(f"{save_dir}/step_distribution.pdf")
    plt.show()

    # -----------------------------
    # 7. 打印统计
    # -----------------------------
    print("\n=== Compression Statistics by Group ===")
    stats = df.groupby("Group").agg({
        "ID": "count",
        "Orig_Tokens": "mean",
        "Comp_Tokens": "mean",
        "Token_Reduction": "mean"
    }).rename(columns={"ID": "Num_Samples"})
    print(stats)
# ========================
#   命令行用法（可选）
# ========================

if __name__ == "__main__":
    tokenizer_path = "ckpts/DAPO/deepseek-qwen1.5B-compressed"
    jsonl_file_path = "ckpts/DAPO/deepseek-qwen1.5B-compressed/eval/__home__hrzhu__code__l2s__ckpts__DAPO__DAPO-deepseek-Qwen1.5B-stage2__global_step_960__actor_hf/samples_math_500_2025-08-30T03-19-43.826854.jsonl"
    jsonl2_file_path = "/home/hrzhu/model/DeepSeek-R1-Distill-Qwen-1.5B/eval/__home__hrzhu__model__DeepSeek-R1-Distill-Qwen-1.5B/samples_math_500_2025-07-16T22-59-04.683043.jsonl"
    ours = evaluate_math500(tokenizer_path, jsonl_file_path)
    deepseekr1 = evaluate_math500(tokenizer_path, jsonl2_file_path)


    visualize_compression_effects_(ours, deepseekr1, "utils/")