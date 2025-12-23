import os
from huggingface_hub import HfApi

# 填入你的 Token (在 Settings -> Access Tokens 获取，需要 Write 权限)
token = os.getenv('HF_TOKEN')
repo_id = "Juhywcy/verl-whl" # 你的用户名/仓库名

api = HfApi()

# 上传单个文件
api.upload_file(
    path_or_fileobj="third_party/flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl", # 本地文件路径
    path_in_repo="flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl",                  # 在 HF 仓库中的文件名
    repo_id=repo_id,
    repo_type="model",                           # 或者是 "dataset"
    token=token
)

print("上传完成！")