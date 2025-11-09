import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import faiss  # type: ignore
import numpy as np

from .clip_service import ClipService


@dataclass
class SearchResult:
    path: str
    score: float


class IndexService:
    def __init__(self, image_root: str, index_dir: str, model_name: str = "openai/clip-vit-base-patch32") -> None:
        self.image_root = os.path.abspath(image_root)
        self.index_dir = os.path.abspath(index_dir)
        self.clip = ClipService(model_name=model_name)
        self.index: faiss.Index | None = None
        self.meta: List[str] = []

    def _index_paths(self) -> Tuple[str, str]:
        return (
            os.path.join(self.index_dir, "image.index"),
            os.path.join(self.index_dir, "meta.json"),
        )

    def build_index(self, batch_size: int = 32, recursive: bool = True) -> None:
        paths = self._scan_images(self.image_root, recursive)
        os.makedirs(self.index_dir, exist_ok=True)
        if not paths:
            raise RuntimeError(f"No images found under {self.image_root}")

        emb = self.clip.encode_images(paths, batch_size=batch_size).numpy().astype(np.float32)
        index = faiss.IndexFlatIP(emb.shape[1])  # cosine on normalized vectors
        index.add(emb)

        index_path, meta_path = self._index_paths()
        faiss.write_index(index, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"image_paths": paths}, f, ensure_ascii=False, indent=2)

        self.index = index
        self.meta = paths

    def load_index(self) -> None:
        index_path, meta_path = self._index_paths()
        if not (os.path.exists(index_path) and os.path.exists(meta_path)):
            raise FileNotFoundError("Index files not found. Run the indexing script first.")
        self.index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            self.meta = meta.get("image_paths", [])

    def is_ready(self) -> bool:
        return self.index is not None and len(self.meta) > 0

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        if not self.is_ready():
            self.load_index()
        assert self.index is not None

        text_emb = self.clip.encode_texts([query])
        query_vec = text_emb.numpy().astype(np.float32)
        scores, indices = self.index.search(query_vec, min(top_k, len(self.meta)))
        results: List[SearchResult] = []
        for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            if idx == -1:
                continue
            results.append(SearchResult(path=self.meta[idx], score=float(score)))
        return results

    @staticmethod
    def _scan_images(root: str, recursive: bool) -> List[str]:
        valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
        paths: List[str] = []
        if recursive:
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    ext = os.path.splitext(fn)[1].lower()
                    if ext in valid_ext:
                        paths.append(os.path.join(dirpath, fn))
        else:
            for fn in os.listdir(root):
                full = os.path.join(root, fn)
                if os.path.isfile(full) and os.path.splitext(fn)[1].lower() in valid_ext:
                    paths.append(full)
        paths.sort()
        return paths
