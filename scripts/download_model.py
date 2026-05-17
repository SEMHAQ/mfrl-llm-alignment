"""模型下载脚本

支持从ModelScope（国内镜像，推荐）或HuggingFace下载模型。

用法：
    # 从ModelScope下载（推荐，国内最快）
    python scripts/download_model.py --source modelscope --model Qwen/Qwen2.5-1.5B-Instruct

    # 从HuggingFace下载（需设置HF_ENDPOINT镜像）
    python scripts/download_model.py --source huggingface --model Qwen/Qwen2.5-1.5B-Instruct
"""

import argparse
import os
from pathlib import Path


def download_from_modelscope(model_id: str, local_dir: str):
    """从ModelScope下载模型。"""
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("Installing modelscope...")
        os.system("pip install modelscope -q")
        from modelscope import snapshot_download

    # ModelScope的Qwen模型ID格式
    ms_model_map = {
        "Qwen/Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    }
    ms_model_id = ms_model_map.get(model_id, model_id)

    print(f"Downloading {ms_model_id} from ModelScope to {local_dir}...")
    snapshot_download(ms_model_id, cache_dir=local_dir)
    print(f"Done! Model saved to {local_dir}")
    return local_dir


def download_from_huggingface(model_id: str, local_dir: str):
    """从HuggingFace下载模型。"""
    from huggingface_hub import snapshot_download

    print(f"Downloading {model_id} from HuggingFace to {local_dir}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Done! Model saved to {local_dir}")
    return local_dir


def main():
    parser = argparse.ArgumentParser(description="Download model")
    parser.add_argument(
        "--source",
        type=str,
        default="modelscope",
        choices=["modelscope", "huggingface"],
        help="Download source (default: modelscope)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model ID",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Local output directory (default: models/<model_name>)",
    )
    args = parser.parse_args()

    if args.output is None:
        model_name = args.model.split("/")[-1]
        args.output = str(Path(__file__).parent.parent / "models" / model_name)

    os.makedirs(args.output, exist_ok=True)

    if args.source == "modelscope":
        download_from_modelscope(args.model, args.output)
    else:
        download_from_huggingface(args.model, args.output)

    print(f"\nUpdate configs/train.yaml with:")
    print(f'  model:\n    name: "{args.output}"')


if __name__ == "__main__":
    main()
