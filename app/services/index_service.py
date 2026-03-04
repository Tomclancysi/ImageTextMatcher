import json
import os
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import faiss
import numpy as np
import torch

from .clip_service import ClipService
from .vse_service import VSEService
from .scan_service import SCANService


@dataclass
class SearchResult:
    path: str
    score: float
    description: Optional[Dict] = None
    vector_summary: Optional[List[float]] = None


class IndexService:
    def __init__(self, image_root: str, index_dir: str,
                 method: str = "clip", model_name: str = "openai/clip-vit-base-patch32",
                 dataset_csv: Optional[str] = None,
                 vse_checkpoint: Optional[str] = None) -> None:
        self.image_root = os.path.abspath(image_root)
        self.index_dir = os.path.abspath(index_dir)
        self.method = method.lower()
        vse_checkpoint = vse_checkpoint or os.environ.get("ITM_VSE_CHECKPOINT")
        
        if self.method == "clip":
            self.encoder = ClipService(model_name=model_name)
        elif self.method == "vse":
            self.encoder = VSEService(embed_size=1024, use_bert=True, checkpoint_path=vse_checkpoint)
        elif self.method == "scan":
            self.encoder = SCANService(embed_size=1024, use_bert=True)
        else:
            raise ValueError(f"Unknown method: {method}. Supported methods: clip, vse, scan")
        
        self.index: Optional[faiss.Index] = None
        self.meta: List[str] = []
        self.image_features: Optional[torch.Tensor] = None
        
        self.description_map: Dict[str, Dict] = {}
        if dataset_csv:
            self._load_dataset_descriptions(dataset_csv)

    def _index_paths(self) -> Tuple[str, str, str]:
        """返回索引文件路径"""
        method_prefix = self.method
        return (
            os.path.join(self.index_dir, f"{method_prefix}_image.index"),
            os.path.join(self.index_dir, f"{method_prefix}_meta.json"),
            os.path.join(self.index_dir, f"{method_prefix}_image_features.pt"),
        )

    def build_index(self, batch_size: int = 32, recursive: bool = True) -> None:
        paths = self._scan_images(self.image_root, recursive)
        os.makedirs(self.index_dir, exist_ok=True)
        if not paths:
            raise RuntimeError(f"No images found under {self.image_root}")

        if self.method == "scan":
            image_features = self.encoder.encode_images(paths, batch_size=batch_size)
            self.image_features = image_features
            
            index_path, meta_path, features_path = self._index_paths()
            torch.save(image_features, features_path)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"image_paths": paths, "method": self.method}, f, ensure_ascii=False, indent=2)
        else:
            emb = self.encoder.encode_images(paths, batch_size=batch_size).numpy().astype(np.float32)
            index = faiss.IndexFlatIP(emb.shape[1])
            index.add(emb)

            index_path, meta_path, features_path = self._index_paths()
            faiss.write_index(index, index_path)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"image_paths": paths, "method": self.method}, f, ensure_ascii=False, indent=2)

            self.index = index
        
        self.meta = paths

    def load_index(self) -> None:
        index_path, meta_path, features_path = self._index_paths()
        if not os.path.exists(meta_path):
            print(f"{meta_path} Index files not found. Run the indexing script first.")
            raise FileNotFoundError(f"{meta_path} Index files not found. Run the indexing script first.")
        
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            self.meta = meta.get("image_paths", [])
            stored_method = meta.get("method", "clip")
            if stored_method != self.method:
                raise ValueError(f"Index was built with method '{stored_method}', but current method is '{self.method}'")
        
        if self.method == "scan":
            if not os.path.exists(features_path):
                raise FileNotFoundError("Image features file not found.")
            self.image_features = torch.load(features_path, map_location="cpu", weights_only=True)
        else:
            if not os.path.exists(index_path):
                raise FileNotFoundError("Index file not found.")
            self.index = faiss.read_index(index_path)

    def is_ready(self) -> bool:
        if self.method == "scan":
            return self.image_features is not None and len(self.meta) > 0
        else:
            return self.index is not None and len(self.meta) > 0

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        if not self.is_ready():
            self.load_index()
        
        top_k = min(top_k, len(self.meta))
        
        if self.method == "scan":
            assert self.image_features is not None
            text_features = self.encoder.encode_texts([query])
            
            scores_list = []
            for i in range(len(self.meta)):
                img_feat = self.image_features[i:i+1]
                similarity = self.encoder.compute_similarity(img_feat, text_features)
                scores_list.append(float(similarity[0]))
            
            indexed_scores = [(i, score) for i, score in enumerate(scores_list)]
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            
            results: List[SearchResult] = []
            for idx, score in indexed_scores[:top_k]:
                description = self._get_image_description(self.meta[idx])
                vector_summary = self._get_vector_summary(idx)
                results.append(SearchResult(path=self.meta[idx], score=score, description=description, vector_summary=vector_summary))
        else:
            assert self.index is not None
            text_emb = self.encoder.encode_texts([query])
            query_vec = text_emb.numpy().astype(np.float32)
            scores, indices = self.index.search(query_vec, top_k)
            results: List[SearchResult] = []
            for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
                if idx == -1:
                    continue
                description = self._get_image_description(self.meta[idx])
                vector_summary = self._get_vector_summary(idx)
                results.append(SearchResult(path=self.meta[idx], score=float(score), description=description, vector_summary=vector_summary))
        
        return results

    def _load_dataset_descriptions(self, csv_path: str) -> None:
        """从CSV文件加载图片描述映射"""
        if not os.path.exists(csv_path):
            print(f"Warning: Dataset CSV file not found: {csv_path}")
            return
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    url = row.get('url', '')
                    cap_seg = row.get('cap_seg', '')
                    
                    if url:
                        filename = os.path.basename(url)
                        if filename and cap_seg:
                            try:
                                description = json.loads(cap_seg)
                            except json.JSONDecodeError:
                                try:
                                    fixed_str = cap_seg.replace('""', '"')
                                    description = json.loads(fixed_str)
                                except json.JSONDecodeError:
                                    continue
                            
                            self.description_map[filename] = description
        except Exception as e:
            print(f"Warning: Failed to load dataset descriptions: {e}")
    
    def _get_image_description(self, image_path: str) -> Optional[Dict]:
        """根据图片路径获取描述"""
        filename = os.path.basename(image_path)
        return self.description_map.get(filename)
    
    def _normalize_vector_for_visualization(self, vector: np.ndarray, max_dims: int = 64) -> List[float]:
        """将向量归一化并采样用于可视化"""
        v_min = vector.min()
        v_max = vector.max()
        v_range = v_max - v_min
        
        if v_range > 1e-6:
            vector_normalized = (vector - v_min) / v_range
        else:
            vector_normalized = np.full_like(vector, 0.5)
        
        dim = len(vector_normalized)
        if dim <= max_dims:
            sampled = vector_normalized.tolist()
        else:
            indices = np.linspace(0, dim - 1, max_dims, dtype=int)
            sampled = vector_normalized[indices].tolist()
        
        return sampled
    
    def _get_vector_summary(self, idx: int, max_dims: int = 64) -> Optional[List[float]]:
        """获取图片向量的摘要用于可视化（采样部分维度）"""
        try:
            if self.method == "scan":
                if self.image_features is not None:
                    region_features = self.image_features[idx]
                    vector = region_features.mean(dim=0).numpy()
                else:
                    return None
            else:
                if self.index is not None:
                    vector = self.index.reconstruct(int(idx))
                else:
                    return None
            
            return self._normalize_vector_for_visualization(vector, max_dims)
        except Exception as e:
            print(f"Warning: Failed to get vector summary: {e}")
            return None
    
    def _get_query_vector_summary(self, query: str, max_dims: int = 64) -> Optional[List[float]]:
        """获取查询文本向量的摘要用于可视化"""
        try:
            text_emb = self.encoder.encode_texts([query])
            
            if self.method == "scan":
                if text_emb.dim() == 3:
                    vector = text_emb[0].mean(dim=0).numpy()
                elif text_emb.dim() == 2:
                    vector = text_emb[0].numpy()
                else:
                    vector = text_emb.squeeze().numpy()
            else:
                if text_emb.dim() == 2:
                    vector = text_emb[0].numpy()
                else:
                    vector = text_emb.squeeze().numpy()
            
            if not isinstance(vector, np.ndarray):
                vector = np.array(vector)
            
            if vector.size == 0:
                print(f"Warning: Query vector is empty for query: {query}")
                return None
            
            result = self._normalize_vector_for_visualization(vector, max_dims)
            if not result or len(result) == 0:
                print(f"Warning: Normalized vector is empty for query: {query}")
                return None
            
            return result
        except Exception as e:
            print(f"Warning: Failed to get query vector summary: {e}")
            import traceback
            traceback.print_exc()
            return None
    
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
