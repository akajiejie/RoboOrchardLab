from huggingface_hub import snapshot_download
import os

# # 建议设置使用国内镜像，以提升下载稳定性 :cite[10]
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

snapshot_download(
    repo_id="HorizonRobotics/Real-World-Dataset",
    repo_type="dataset",
    local_dir="./Real-World-Dataset/",  # 建议指定一个清晰的本地目录
    allow_patterns=[
        "arrow_dataset_stack_bowls_three_2025_09_09/*",  # 匹配所有包含"2025_09_"的文件夹及其内部所有内容 :cite[2]
        "lmdb_dataset_stack_bowls_three_2025_09_09/*",  # 匹配所有包含"2025_09_"的文件夹及其内部所有内容 :cite[2]
        "urdf/*",
        # "*.md",  # 通常包含重要的数据集说明，建议保留
        # "*.txt",
        # "*.json"
    ],
    resume_download=True,  # 启用断点续传 :cite[3]
    local_dir_use_symlinks="auto"  # 让库自动决定是否使用符号链接 :cite[10]
)