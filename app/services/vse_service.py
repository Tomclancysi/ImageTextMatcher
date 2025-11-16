import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel


class ImageEncoder(nn.Module):
    """图像编码器：使用ResNet提取特征"""
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
        """提取图像特征"""
        with torch.set_grad_enabled(self.training):
            features = self.resnet(images)
            features = features.view(features.size(0), -1)
            features = self.linear(features)
            features = self.bn(features)
        return features


class TextEncoder(nn.Module):
    """文本编码器：使用GRU或BERT"""
    def __init__(self, vocab_size: int, word_dim: int = 300, embed_size: int = 1024,
                 num_layers: int = 1, use_bert: bool = True):
        super().__init__()
        self.use_bert = use_bert
        self.embed_size = embed_size
        
        if use_bert:
            self.bert = AutoModel.from_pretrained('bert-base-uncased', use_safetensors=True)
            bert_dim = self.bert.config.hidden_size
            self.linear = nn.Linear(bert_dim, embed_size)
            self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        else:
            self.embed = nn.Embedding(vocab_size, word_dim)
            self.gru = nn.GRU(word_dim, embed_size, num_layers, batch_first=True)
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """提取文本特征"""
        if self.use_bert:
            outputs = self.bert(x['input_ids'], attention_mask=x['attention_mask'])
            features = outputs.last_hidden_state[:, 0, :]
            features = self.linear(features)
            features = self.bn(features)
        else:
            embedded = self.embed(x)
            if lengths is not None:
                embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
            output, hidden = self.gru(embedded)
            if lengths is not None:
                output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            features = hidden[-1]
        return features


class VSEService:
    """VSE++服务：实现视觉-语义嵌入匹配"""
    def __init__(self, embed_size: int = 1024, use_bert: bool = True, device: Optional[str] = None):
        self.embed_size = embed_size
        self.use_bert = use_bert
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"VSEService device: {self.device}")
        
        self.image_encoder = ImageEncoder(embed_size=embed_size).to(self.device)
        self.text_encoder = TextEncoder(
            vocab_size=30522,
            embed_size=embed_size,
            use_bert=use_bert
        ).to(self.device)
        
        self.image_encoder.eval()
        self.text_encoder.eval()
        
        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if use_bert:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    @torch.inference_mode()
    def encode_images(self, image_paths: List[str], batch_size: int = 32) -> torch.Tensor:
        """编码图像为特征向量（L2归一化）"""
        embeds: List[torch.Tensor] = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    img = self.image_transform(img)
                    images.append(img)
                except Exception as e:
                    images.append(torch.zeros(3, 224, 224))
            
            images_tensor = torch.stack(images).to(self.device)
            features = self.image_encoder(images_tensor)
            features = F.normalize(features, p=2, dim=1)
            embeds.append(features.detach().cpu())
        
        return torch.cat(embeds, dim=0) if embeds else torch.empty((0, self.embed_size))
    
    @torch.inference_mode()
    def encode_texts(self, texts: List[str], batch_size: int = 64) -> torch.Tensor:
        """编码文本为特征向量（L2归一化）"""
        embeds: List[torch.Tensor] = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            if self.use_bert:
                encoded = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                features = self.text_encoder(encoded)
            else:
                raise NotImplementedError("GRU编码器需要词汇表，请使用BERT模式")
            
            features = F.normalize(features, p=2, dim=1)
            embeds.append(features.detach().cpu())
        
        return torch.cat(embeds, dim=0) if embeds else torch.empty((0, self.embed_size))
    
    def get_feature_dim(self) -> int:
        """返回特征维度"""
        return self.embed_size

