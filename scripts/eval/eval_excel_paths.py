"""
用法：
    python scripts/eval/eval_excel_paths.py <output_excel> <model_path1> [model_path2] ...

示例：
    python scripts/eval/eval_excel_paths.py results.xlsx \
        ckpts/run1/global_step_100/actor_hf \
        ckpts/run2/global_step_200/actor_hf \
        ckpts/run3/global_step_300/actor_hf

每个 model_path 下需要存在 eval/ 子目录（由 eval.sh 生成的评测结果）。
输出的 Excel 文件路径由第一个参数指定。
"""

import sys
import os
from pathlib import Path

import pandas as pd

# 将脚本目录加入 sys.path，以便直接 import 同目录下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_aime import eval_aime_import
from eval_gsm8k import eval_gsm8k_import
from eval_math500 import eval_math500_import
from eval_minerva import eval_minerva_import


def eval_single_path(model_path: Path) -> dict | None:
    """对单个模型路径进行评测，返回指标字典；若 eval/ 目录不存在则返回 None。"""
    eval_path = model_path / "eval"
    if not eval_path.exists():
        print(f"[跳过] eval 目录不存在: {eval_path}")
        return None

    print(f"[评测] {model_path}")
    aime_acc,    aime_tokens    = eval_aime_import(model_path,    eval_path)
    gsm8k_acc,   gsm8k_tokens   = eval_gsm8k_import(model_path,   eval_path)
    math500_acc, math500_tokens = eval_math500_import(model_path, eval_path)
    minerva_acc, minerva_tokens = eval_minerva_import(model_path, eval_path)

    return {
        "model_path":     str(model_path),
        "aime_acc":       round(aime_acc    * 100, 2),
        "aime_tokens":    int(aime_tokens),
        "minerva_acc":    round(minerva_acc  * 100, 2),
        "minerva_tokens": int(minerva_tokens),
        "math500_acc":    round(math500_acc  * 100, 2),
        "math500_tokens": int(math500_tokens),
        "gsm8k_acc":      round(gsm8k_acc   * 100, 2),
        "gsm8k_tokens":   int(gsm8k_tokens),
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python eval_excel_paths.py <output_excel> <model_path1> [model_path2] ...")
        sys.exit(1)

    output_file  = Path(sys.argv[1])
    model_paths  = [Path(p) for p in sys.argv[2:]]

    data = []
    for mp in model_paths:
        result = eval_single_path(mp)
        if result is not None:
            data.append(result)

    if not data:
        print("没有找到任何有效的评测结果，Excel 文件未生成。")
        sys.exit(1)

    df = pd.DataFrame(data)
    df.to_excel(output_file, index=False)
    print(f"\n评估指标已保存到: {output_file}")
    print(df.to_string(index=False))
