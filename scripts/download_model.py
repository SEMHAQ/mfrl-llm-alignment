"""模型下载脚本

支持从ModelScope或HuggingFace下载模型到本地目录。

用法：
    python scripts/download_model.py --source modelscope
    python scripts/download_model.py --source huggingface
"""

import argparse
import os
import subprocess
import shutil
from pathlib import Path


def download_from_modelscope(model_id: str, local_dir: str):
    """从ModelScope下载模型（使用CLI，最稳定）。"""
    # ModelScope CLI 直接下载，最可靠
    print(f"Downloading {model_id} from ModelScope to {local_dir}...")
    print("This may take several minutes for model weights (~3GB)...\n")

    cmd = [
        "modelscope", "download",
        "--model", model_id,
        "--local_dir", local_dir,
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("modelscope CLI failed, trying Python API...")
        _download_modelscope_python(model_id, local_dir)

    _flatten_nested_dir(model_id, local_dir)
    print(f"\nDone! Model saved to {local_dir}")
    return local_dir


def _download_modelscope_python(model_id: str, local_dir: str):
    """ModelScope Python API fallback。"""
    try:
        from modelscope import snapshot_download
    except ImportError:
        os.system("pip install modelscope -q")
        from modelscope import snapshot_download

    snapshot_download(model_id, local_dir=local_dir)


def _flatten_nested_dir(model_id: str, local_dir: str):
    """ModelScope会在local_dir下创建嵌套目录，需要展平。"""
    nested = os.path.join(local_dir, *model_id.split("/"))
    config_nested = os.path.join(nested, "config.json")
    config_root = os.path.join(local_dir, "config.json")

    if os.path.isdir(nested) and os.path.isfile(config_nested) and not os.path.isfile(config_root):
        print(f"Flattening nested directory: {nested} -> {local_dir}")
        for f in os.listdir(nested):
            src = os.path.join(nested, f)
            dst = os.path.join(local_dir, f)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        # 清理空目录
        for d in [nested, os.path.dirname(nested)]:
            if os.path.isdir(d) and not os.listdir(d):
                os.rmdir(d)


def download_from_huggingface(model_id: str, local_dir: str):
    """从HuggingFace下载模型。"""
    print(f"Downloading {model_id} from HuggingFace to {local_dir}...")

    cmd = [
        "huggingface-cli", "download",
        model_id,
        "--local-dir", local_dir,
    ]
    subprocess.run(cmd, capture_output=False)

    print(f"\nDone! Model saved to {local_dir}")
    return local_dir


def main():
    parser = argparse.ArgumentParser(description="Download model")
    parser.add_argument("--source", type=str, default="modelscope",
                        choices=["modelscope", "huggingface"])
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        model_name = args.model.split("/")[-1]
        args.output = str(Path(__file__).parent.parent / "models" / model_name)

    os.makedirs(args.output, exist_ok=True)

    if args.source == "modelscope":
        download_from_modelscope(args.model, args.output)
    else:
        download_from_huggingface(args.model, args.output)

    print(f"\nModel path: {args.output}")
    print(f"Verify: ls {args.output}")


if __name__ == "__main__":
    main()
