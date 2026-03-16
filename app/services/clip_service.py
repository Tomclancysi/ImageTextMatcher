from typing import List, Optional

import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoProcessor


class ClipService:
    """Generic vision-text dual-encoder service.

    Despite the historical name, this service now supports CLIP, SigLIP, and
    SigLIP2-style Hugging Face models that expose `get_image_features` and
    `get_text_features`.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = self._resolve_dtype(torch_dtype)
        self._model = None
        self._processor = None
        self._config = None
        self._model_type = None

    def _resolve_dtype(self, torch_dtype: Optional[str]) -> Optional[torch.dtype]:
        if torch_dtype is None:
            return torch.float16 if self.device == "cuda" else None

        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        key = torch_dtype.strip().lower()
        if key not in mapping:
            raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
        return mapping[key]

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None and self._config is not None:
            return

        self._config = AutoConfig.from_pretrained(self.model_name)
        self._model_type = getattr(self._config, "model_type", "unknown")
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        model_kwargs = {"use_safetensors": True}
        if self.torch_dtype is not None:
            model_kwargs["torch_dtype"] = self.torch_dtype
        self._model = AutoModel.from_pretrained(self.model_name, **model_kwargs)

        if not hasattr(self._model, "get_image_features") or not hasattr(self._model, "get_text_features"):
            raise TypeError(
                f"Model '{self.model_name}' does not expose dual-encoder feature methods. "
                "Choose a CLIP/SigLIP/SigLIP2-style retrieval model."
            )

        self._model.eval()
        self._model.to(self.device)

    def _processor_text_kwargs(self) -> dict:
        self._ensure_loaded()
        kwargs = {"return_tensors": "pt", "truncation": True}

        if self._model_type in {"siglip", "siglip2"}:
            kwargs["padding"] = "max_length"
        else:
            kwargs["padding"] = True

        tokenizer = getattr(self._processor, "tokenizer", None)
        model_max_length = getattr(tokenizer, "model_max_length", None)
        if isinstance(model_max_length, int) and 0 < model_max_length < 10000:
            kwargs["max_length"] = model_max_length
        else:
            kwargs["max_length"] = 77

        return kwargs

    @torch.inference_mode()
    def encode_images(self, image_paths: List[str], batch_size: int = 32) -> torch.Tensor:
        self._ensure_loaded()

        embeds: List[torch.Tensor] = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = self._processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            image_features = self._model.get_image_features(**inputs)
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
            embeds.append(image_features.detach().cpu().float())

        return torch.cat(embeds, dim=0) if embeds else torch.empty((0, self.get_feature_dim()))

    @torch.inference_mode()
    def encode_texts(self, texts: List[str], batch_size: int = 64) -> torch.Tensor:
        self._ensure_loaded()

        embeds: List[torch.Tensor] = []
        text_kwargs = self._processor_text_kwargs()
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self._processor(text=batch_texts, **text_kwargs)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_features = self._model.get_text_features(**inputs)
            text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)
            embeds.append(text_features.detach().cpu().float())

        return torch.cat(embeds, dim=0) if embeds else torch.empty((0, self.get_feature_dim()))

    def get_feature_dim(self) -> int:
        self._ensure_loaded()

        if hasattr(self._model.config, "projection_dim"):
            return int(self._model.config.projection_dim)
        if hasattr(self._model.config, "text_config") and hasattr(self._model.config.text_config, "projection_size"):
            return int(self._model.config.text_config.projection_size)
        if hasattr(self._model.config, "vision_config") and hasattr(self._model.config.vision_config, "hidden_size"):
            return int(self._model.config.vision_config.hidden_size)

        raise AttributeError(f"Could not infer feature dimension for model '{self.model_name}'.")
