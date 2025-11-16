import os
from typing import List, Optional

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class ClipService:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: Optional[str] = None) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[CLIPModel] = None
        self._processor: Optional[CLIPProcessor] = None

    def _ensure_loaded(self) -> None:
        if self._model is None or self._processor is None:
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model = CLIPModel.from_pretrained(self.model_name, use_safetensors=True)
            self._model.eval()
            self._model.to(self.device)

    @torch.inference_mode()
    def encode_images(self, image_paths: List[str], batch_size: int = 32) -> torch.Tensor:
        """Encode images into L2-normalized feature vectors (CPU tensor)."""
        self._ensure_loaded()
        assert self._processor is not None and self._model is not None

        embeds: List[torch.Tensor] = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = self._processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            image_features = self._model.get_image_features(**inputs)
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
            embeds.append(image_features.detach().cpu())

        return torch.cat(embeds, dim=0) if embeds else torch.empty((0, self.get_feature_dim()))

    @torch.inference_mode()
    def encode_texts(self, texts: List[str], batch_size: int = 64) -> torch.Tensor:
        """Encode texts into L2-normalized feature vectors (CPU tensor)."""
        self._ensure_loaded()
        assert self._processor is not None and self._model is not None

        embeds: List[torch.Tensor] = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self._processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_features = self._model.get_text_features(**inputs)
            text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)
            embeds.append(text_features.detach().cpu())

        return torch.cat(embeds, dim=0) if embeds else torch.empty((0, self.get_feature_dim()))

    def get_feature_dim(self) -> int:
        self._ensure_loaded()
        assert self._model is not None
        return self._model.config.projection_dim
