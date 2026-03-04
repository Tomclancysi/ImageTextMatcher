from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer


def build_image_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class ImageEncoder(nn.Module):
    def __init__(self, embed_size: int = 1024, finetune: bool = False):
        super().__init__()
        import torchvision.models as models

        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

        for param in self.resnet.parameters():
            param.requires_grad = finetune

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(self.training):
            features = self.resnet(images)
            features = features.view(features.size(0), -1)
        features = self.linear(features)
        features = self.bn(features)
        return features


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        word_dim: int = 300,
        embed_size: int = 1024,
        num_layers: int = 1,
        use_bert: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.use_bert = use_bert
        self.embed_size = embed_size

        if use_bert:
            self.bert = AutoModel.from_pretrained("bert-base-uncased", use_safetensors=True)
            if freeze_backbone:
                for param in self.bert.parameters():
                    param.requires_grad = False
            bert_dim = self.bert.config.hidden_size
            self.linear = nn.Linear(bert_dim, embed_size)
            self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        else:
            self.embed = nn.Embedding(vocab_size, word_dim)
            self.gru = nn.GRU(word_dim, embed_size, num_layers, batch_first=True)

    def forward(self, x: Dict[str, torch.Tensor], lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_bert:
            outputs = self.bert(x["input_ids"], attention_mask=x["attention_mask"])
            features = outputs.last_hidden_state[:, 0, :]
            features = self.linear(features)
            features = self.bn(features)
            return features

        embedded = self.embed(x)
        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
        _, hidden = self.gru(embedded)
        return hidden[-1]


class VSEService:
    def __init__(
        self,
        embed_size: int = 1024,
        use_bert: bool = True,
        device: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        finetune_image_backbone: bool = False,
        freeze_text_backbone: bool = True,
    ):
        self.embed_size = embed_size
        self.use_bert = use_bert
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"VSEService device: {self.device}")

        self.image_encoder = ImageEncoder(
            embed_size=embed_size,
            finetune=finetune_image_backbone,
        ).to(self.device)
        self.text_encoder = TextEncoder(
            vocab_size=30522,
            embed_size=embed_size,
            use_bert=use_bert,
            freeze_backbone=freeze_text_backbone,
        ).to(self.device)

        self.image_transform = build_image_transform()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") if use_bert else None

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        self.eval()

    def train(self) -> None:
        self.image_encoder.train()
        self.text_encoder.train()

    def eval(self) -> None:
        self.image_encoder.eval()
        self.text_encoder.eval()

    def encode_image_batch(self, images: torch.Tensor) -> torch.Tensor:
        features = self.image_encoder(images.to(self.device))
        return F.normalize(features, p=2, dim=1)

    def encode_text_batch(self, tokenized: Dict[str, torch.Tensor]) -> torch.Tensor:
        tokenized = {key: value.to(self.device) for key, value in tokenized.items()}
        features = self.text_encoder(tokenized)
        return F.normalize(features, p=2, dim=1)

    def tokenize_texts(self, texts: List[str], max_length: int = 77) -> Dict[str, torch.Tensor]:
        if not self.use_bert or self.tokenizer is None:
            raise NotImplementedError("Only the BERT text encoder is supported in this project.")
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

    @torch.inference_mode()
    def encode_images(self, image_paths: List[str], batch_size: int = 32) -> torch.Tensor:
        embeds: List[torch.Tensor] = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            images = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(self.image_transform(img))
                except Exception:
                    images.append(torch.zeros(3, 224, 224))

            images_tensor = torch.stack(images)
            features = self.encode_image_batch(images_tensor)
            embeds.append(features.detach().cpu())

        return torch.cat(embeds, dim=0) if embeds else torch.empty((0, self.embed_size))

    @torch.inference_mode()
    def encode_texts(self, texts: List[str], batch_size: int = 64) -> torch.Tensor:
        embeds: List[torch.Tensor] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            tokenized = self.tokenize_texts(batch_texts)
            features = self.encode_text_batch(tokenized)
            embeds.append(features.detach().cpu())

        return torch.cat(embeds, dim=0) if embeds else torch.empty((0, self.embed_size))

    def checkpoint_payload(self) -> Dict[str, object]:
        return {
            "embed_size": self.embed_size,
            "use_bert": self.use_bert,
            "image_encoder": self.image_encoder.state_dict(),
            "text_encoder": self.text_encoder.state_dict(),
        }

    def save_checkpoint(
        self,
        checkpoint_path: str,
        extra: Optional[Dict[str, object]] = None,
    ) -> None:
        payload = self.checkpoint_payload()
        if extra:
            payload.update(extra)
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        checkpoint_embed = int(payload.get("embed_size", self.embed_size))
        checkpoint_use_bert = bool(payload.get("use_bert", self.use_bert))

        if checkpoint_embed != self.embed_size:
            raise ValueError(
                f"Checkpoint embed_size {checkpoint_embed} does not match service embed_size {self.embed_size}."
            )
        if checkpoint_use_bert != self.use_bert:
            raise ValueError("Checkpoint use_bert setting does not match service configuration.")

        self.image_encoder.load_state_dict(payload["image_encoder"])
        self.text_encoder.load_state_dict(payload["text_encoder"])

    def get_feature_dim(self) -> int:
        return self.embed_size
