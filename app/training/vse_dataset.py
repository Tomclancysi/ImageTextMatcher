import csv
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from torch.utils.data import Dataset


@dataclass
class VSESample:
    image_path: str
    text: str


def _parse_caption_blob(raw_value: str) -> Optional[Dict[str, object]]:
    if not raw_value:
        return None

    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        try:
            return json.loads(raw_value.replace('""', '"'))
        except json.JSONDecodeError:
            return None


def build_vse_samples(
    csv_path: str,
    image_root: str,
    seed: int = 42,
    max_samples: Optional[int] = None,
    captions_per_image: int = 2,
) -> List[VSESample]:
    image_root_path = Path(image_root)
    rng = random.Random(seed)
    samples: List[VSESample] = []

    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            url = (row.get("url") or "").strip()
            if not url:
                continue

            file_name = os.path.basename(url)
            image_path = image_root_path / file_name
            if not image_path.exists():
                continue

            description = _parse_caption_blob((row.get("cap_seg") or "").strip())
            if not description:
                continue

            texts: List[str] = []
            global_caption = str(description.get("global_caption") or "").strip()
            if global_caption:
                texts.append(global_caption)

            local_captions = description.get("local_caption") or []
            if isinstance(local_captions, list):
                for caption in local_captions:
                    text = str(caption or "").strip()
                    if text:
                        texts.append(text)

            if not texts:
                continue

            deduped: List[str] = []
            seen = set()
            for text in texts:
                if text not in seen:
                    seen.add(text)
                    deduped.append(text)

            rng.shuffle(deduped)
            for text in deduped[: max(1, captions_per_image)]:
                samples.append(VSESample(image_path=str(image_path), text=text))

    rng.shuffle(samples)
    if max_samples is not None:
        return samples[:max_samples]
    return samples


def split_samples(
    samples: Sequence[VSESample],
    val_ratio: float = 0.1,
) -> Tuple[List[VSESample], List[VSESample]]:
    if not samples:
        return [], []

    val_size = int(len(samples) * val_ratio)
    if val_size <= 0:
        return list(samples), []
    if val_size >= len(samples):
        return list(samples), []
    return list(samples[val_size:]), list(samples[:val_size])


class VSEPairDataset(Dataset):
    def __init__(self, samples: Sequence[VSESample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> VSESample:
        return self.samples[index]
