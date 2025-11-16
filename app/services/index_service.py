import json
import os
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import faiss  # type: ignore
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


class IndexService:
    def __init__(self, image_root: str, index_dir: str, 
                 method: str = "clip", model_name: str = "openai/clip-vit-base-patch32",
                 dataset_csv: Optional[str] = None) -> None:
        self.image_root = os.path.abspath(image_root)
        self.index_dir = os.path.abspath(index_dir)
        self.method = method.lower()
        
        # 根据方法选择编码服务
        if self.method == "clip":
            self.encoder = ClipService(model_name=model_name)
        elif self.method == "vse":
            self.encoder = VSEService(embed_size=1024, use_bert=True)
        elif self.method == "scan":
            self.encoder = SCANService(embed_size=1024, use_bert=True)
        else:
            raise ValueError(f"Unknown method: {method}. Supported methods: clip, vse, scan")
        
        self.index: Optional[faiss.Index] = None
        self.meta: List[str] = []
        # SCAN方法需要存储图像特征（区域特征）
        self.image_features: Optional[torch.Tensor] = None
        
        # 加载数据集描述映射（图片文件名 -> 描述JSON）
        self.description_map: Dict[str, Dict] = {}
        if dataset_csv:
            self._load_dataset_descriptions(dataset_csv)

    def _index_paths(self) -> Tuple[str, str, str]:
        """返回索引文件路径"""
        method_prefix = self.method
        return (
            os.path.join(self.index_dir, f"{method_prefix}_image.index"),
            os.path.join(self.index_dir, f"{method_prefix}_meta.json"),
            os.path.join(self.index_dir, f"{method_prefix}_image_features.pt"),  # 用于SCAN
        )

    def build_index(self, batch_size: int = 32, recursive: bool = True) -> None:
        paths = self._scan_images(self.image_root, recursive)
        os.makedirs(self.index_dir, exist_ok=True)
        if not paths:
            raise RuntimeError(f"No images found under {self.image_root}")

        # 编码图像
        if self.method == "scan":
            # SCAN方法：存储区域特征
            image_features = self.encoder.encode_images(paths, batch_size=batch_size)
            self.image_features = image_features
            
            # 对于SCAN，不构建FAISS索引，直接存储特征
            index_path, meta_path, features_path = self._index_paths()
            torch.save(image_features, features_path)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"image_paths": paths, "method": self.method}, f, ensure_ascii=False, indent=2)
        else:
            # CLIP和VSE++方法：构建FAISS索引
            emb = self.encoder.encode_images(paths, batch_size=batch_size).numpy().astype(np.float32)
            index = faiss.IndexFlatIP(emb.shape[1])  # cosine on normalized vectors
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
            # 加载SCAN图像特征
            if not os.path.exists(features_path):
                raise FileNotFoundError("Image features file not found.")
            # 使用 weights_only=True 安全加载（仅用于加载特征张量，非模型权重）
            self.image_features = torch.load(features_path, map_location="cpu", weights_only=True)
        else:
            # 加载FAISS索引
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
            # SCAN方法：使用交叉注意力计算相似度
            assert self.image_features is not None
            text_features = self.encoder.encode_texts([query])  # [1, seq_len, embed_size]
            
            # 计算每个图像与查询文本的相似度
            scores_list = []
            for i in range(len(self.meta)):
                img_feat = self.image_features[i:i+1]  # [1, num_regions, embed_size]
                similarity = self.encoder.compute_similarity(img_feat, text_features)
                scores_list.append(float(similarity[0]))
            
            # 排序并取top_k
            indexed_scores = [(i, score) for i, score in enumerate(scores_list)]
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            
            results: List[SearchResult] = []
            for idx, score in indexed_scores[:top_k]:
                description = self._get_image_description(self.meta[idx])
                results.append(SearchResult(path=self.meta[idx], score=score, description=description))
        else:
            # CLIP和VSE++方法：使用FAISS索引
            assert self.index is not None
            text_emb = self.encoder.encode_texts([query])
            query_vec = text_emb.numpy().astype(np.float32)
            scores, indices = self.index.search(query_vec, top_k)
            results: List[SearchResult] = []
            for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
                if idx == -1:
                    continue
                description = self._get_image_description(self.meta[idx])
                results.append(SearchResult(path=self.meta[idx], score=float(score), description=description))
        
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
                    
                    # 从URL中提取图片文件名
                    if url:
                        filename = os.path.basename(url)
                        if filename and cap_seg:
                            # 解析JSON描述（处理CSV中的双引号转义）
                            try:
                                # 尝试直接解析
                                description = json.loads(cap_seg)
                            except json.JSONDecodeError:
                                # 如果失败，尝试修复双引号转义
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
