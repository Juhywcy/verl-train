from eval_aime import eval_aime_import
from eval_gsm8k import eval_gsm8k_import
from eval_math500 import eval_math500_import
from eval_minerva import eval_minerva_import
import sys, os
from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    root_path = Path(sys.argv[1])
    idx_start = int(sys.argv[2])
    idx_end = int(sys.argv[3])
    data = []
    for idx in range(idx_start, idx_end + 1):
        eval_path = root_path / Path(f"global_step_{idx}")/Path(f"actor_hf")
        print(eval_path)
        if os.path.exists(eval_path/Path("eval")):
            aime_acc, aime_tokens = eval_aime_import(eval_path, eval_path/Path("eval"))
            gsm8k_acc, gsm8k_tokens = eval_gsm8k_import(eval_path, eval_path/Path("eval"))
            math500_acc, math500_tokens = eval_math500_import(eval_path, eval_path/Path("eval"))
            minerva_acc, minerva_tokens = eval_minerva_import(eval_path, eval_path/Path("eval"))
            data.append({
                "eval_path": str(eval_path),  # 转换为字符串便于Excel显示
                "aime_acc": round(aime_acc * 100, 2),
                "aime_tokens": int(aime_tokens),
                "minerva_acc": round(minerva_acc * 100, 2),
                "minerva_tokens": int(minerva_tokens),
                "math500_acc": round(math500_acc * 100, 2),
                "math500_tokens": int(math500_tokens),
                "gsm8k_acc": round(gsm8k_acc * 100, 2),
                "gsm8k_tokens": int(gsm8k_tokens)
            })
    print(data)
    df = pd.DataFrame(data)
    
    # 保存到Excel文件
    output_file = root_path / "evaluation_metrics.xlsx"
    df.to_excel(output_file, index=False)
    print(f"评估指标已保存到: {output_file}")



    