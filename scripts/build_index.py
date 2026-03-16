import argparse
import os
import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.index_service import IndexService


def main():
    parser = argparse.ArgumentParser(description="Build index for images using CLIP/VSE++/SCAN features")
    parser.add_argument("--image_root", default=os.path.join(os.getcwd(), "data", "images"))
    parser.add_argument("--index_dir", default=os.path.join(os.getcwd(), "data", "index"))
    parser.add_argument("--method", choices=["clip", "vse", "scan"], default="clip",
                       help="Matching method: clip, vse, or scan")
    parser.add_argument("--model_name", default="openai/clip-vit-base-patch32",
                       help="Dual-encoder model name for the clip method, e.g. openai/clip-vit-large-patch14 or google/siglip2-base-patch16-224")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--vse_checkpoint", help="Checkpoint path for a trained VSE++ model")
    parser.add_argument("--model_dtype", help="Optional model dtype for the clip method: fp16, bf16, or fp32")
    args = parser.parse_args()

    os.makedirs(args.index_dir, exist_ok=True)
    service = IndexService(
        image_root=args.image_root,
        index_dir=args.index_dir,
        method=args.method,
        model_name=args.model_name,
        vse_checkpoint=args.vse_checkpoint,
        model_dtype=args.model_dtype,
    )
    service.build_index(batch_size=args.batch_size)
    print(f"Index built using {args.method.upper()} method. Images: {len(service.meta)} -> {args.index_dir}")


if __name__ == "__main__":
    main()
