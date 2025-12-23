import os
import shutil
from pathlib import Path
def remove_actor_if_needed(root_path):
    """
    检查指定路径下的所有文件夹，当同时存在actor_hf和actor时，删除actor文件夹
    
    参数:
        root_path: 要检查的根路径
    """
    # 遍历根路径下的所有项目
    actor_hf_path = os.path.join(root_path, "actor_hf")
    actor_path = os.path.join(root_path, "actor")
    
    has_actor_hf = os.path.isdir(actor_hf_path)
    has_actor = os.path.isdir(actor_path)
    
    # 如果同时存在actor_hf和actor，则删除actor
    if has_actor_hf and has_actor:
        try:
            shutil.rmtree(actor_path)
            print(f"已删除: {actor_path}")
        except Exception as e:
            print(f"删除 {actor_path} 失败: {str(e)}")

if __name__ == "__main__":
    # 请替换为你要检查的路径
    target_path = Path("ckpts/DAPO-8192/DAPO-deepseek-Qwen1.5B-stage2-8192")
    for step_num in range(50,600,50):
        remove_actor_if_needed(os.path.join(target_path, Path(f"global_step_{step_num}")))

    
