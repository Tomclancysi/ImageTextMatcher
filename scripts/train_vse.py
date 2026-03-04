import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.vse_service import VSEService
from app.training.vse_dataset import VSEPairDataset, VSESample, build_vse_samples, split_samples


def rank_loss(scores: torch.Tensor, margin: float = 0.2, max_violation: bool = True) -> torch.Tensor:
    diagonal = scores.diag().view(scores.size(0), 1)
    cost_s = (margin + scores - diagonal).clamp(min=0)
    cost_im = (margin + scores - diagonal.t()).clamp(min=0)

    mask = torch.eye(scores.size(0), device=scores.device, dtype=torch.bool)
    cost_s = cost_s.masked_fill(mask, 0)
    cost_im = cost_im.masked_fill(mask, 0)

    if max_violation:
        cost_s = cost_s.max(dim=1)[0]
        cost_im = cost_im.max(dim=0)[0]

    return cost_s.mean() + cost_im.mean()


def build_collate_fn(service: VSEService):
    def collate_fn(batch: List[VSESample]) -> Dict[str, object]:
        images = []
        texts = []
        for item in batch:
            try:
                image = Image.open(item.image_path).convert("RGB")
                images.append(service.image_transform(image))
            except Exception:
                images.append(torch.zeros(3, 224, 224))
            texts.append(item.text)

        image_tensor = torch.stack(images)
        tokenized = service.tokenize_texts(texts)
        return {
            "images": image_tensor,
            "tokenized": tokenized,
            "texts": texts,
        }

    return collate_fn


def run_epoch(
    service: VSEService,
    loader: DataLoader,
    optimizer: AdamW,
    margin: float,
    train: bool,
) -> float:
    if train:
        service.train()
    else:
        service.eval()

    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        images = batch["images"]
        tokenized = batch["tokenized"]

        if train:
            optimizer.zero_grad(set_to_none=True)

        context = torch.enable_grad() if train else torch.inference_mode()
        with context:
            image_features = service.encode_image_batch(images)
            text_features = service.encode_text_batch(tokenized)
            scores = image_features @ text_features.t()
            loss = rank_loss(scores, margin=margin)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_batches += 1

    if total_batches == 0:
        return math.nan
    return total_loss / total_batches


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a minimal VSE++ projection head on paired image-text data.")
    parser.add_argument("--csv", default="data/dataset_en.csv", help="CSV file containing url and cap_seg columns.")
    parser.add_argument("--image_root", default="data/images", help="Directory containing local images.")
    parser.add_argument("--output", default="data/checkpoints/vse_best.pt", help="Path to save the best checkpoint.")
    parser.add_argument("--embed_size", type=int, default=1024, help="Joint embedding dimension.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--margin", type=float, default=0.2, help="Ranking loss margin.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--max_samples", type=int, help="Cap the number of text-image pairs for quick experiments.")
    parser.add_argument("--captions_per_image", type=int, default=2, help="How many captions to keep per image.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", help="Override device, e.g. cpu or cuda.")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--finetune_image_backbone", action="store_true", help="Unfreeze the ResNet backbone.")
    parser.add_argument(
        "--unfreeze_text_backbone",
        action="store_true",
        help="Unfreeze the BERT backbone. This is much slower and uses more memory.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    samples = build_vse_samples(
        csv_path=args.csv,
        image_root=args.image_root,
        seed=args.seed,
        max_samples=args.max_samples,
        captions_per_image=args.captions_per_image,
    )
    if not samples:
        raise SystemExit("No usable image-text pairs found. Check that images exist and cap_seg is populated.")

    train_samples, val_samples = split_samples(samples, val_ratio=max(0.0, min(0.5, args.val_ratio)))
    if not train_samples:
        raise SystemExit("Training split is empty. Reduce --val_ratio or provide more samples.")
    if len(train_samples) < 2:
        raise SystemExit("At least two training pairs are required for ranking loss.")

    service = VSEService(
        embed_size=args.embed_size,
        use_bert=True,
        device=args.device,
        checkpoint_path=None,
        finetune_image_backbone=args.finetune_image_backbone,
        freeze_text_backbone=not args.unfreeze_text_backbone,
    )

    collate_fn = build_collate_fn(service)
    train_loader = DataLoader(
        VSEPairDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(0, args.workers),
        collate_fn=collate_fn,
        drop_last=len(train_samples) > args.batch_size,
    )
    val_loader = None
    if val_samples:
        val_loader = DataLoader(
            VSEPairDataset(val_samples),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=max(0, args.workers),
            collate_fn=collate_fn,
            drop_last=False,
        )

    trainable_params = [param for param in list(service.image_encoder.parameters()) + list(service.text_encoder.parameters()) if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    best_metric = float("inf")
    best_epoch = 0

    print(f"Loaded {len(samples)} total pairs")
    print(f"Train pairs: {len(train_samples)} | Val pairs: {len(val_samples)}")
    print(f"Saving best checkpoint to {args.output}")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(service, train_loader, optimizer, args.margin, train=True)
        if val_loader is not None:
            val_loss = run_epoch(service, val_loader, optimizer, args.margin, train=False)
            monitor = val_loss
            monitor_text = f"val_loss={val_loss:.4f}"
        else:
            val_loss = math.nan
            monitor = train_loss
            monitor_text = "val_loss=n/a"

        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f} {monitor_text}")

        if monitor < best_metric:
            best_metric = monitor
            best_epoch = epoch
            service.save_checkpoint(
                args.output,
                extra={
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "csv": os.path.abspath(args.csv),
                    "image_root": os.path.abspath(args.image_root),
                },
            )

    print(f"Best checkpoint saved from epoch {best_epoch} with monitored loss {best_metric:.4f}")


if __name__ == "__main__":
    main()
