import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel


class ImageRegionEncoder(nn.Module):
    """图像区域编码器：提取图像区域特征"""
    def __init__(self, embed_size: int = 1024):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(2048, embed_size)
        
        for param in self.resnet.parameters():
            param.requires_grad = False
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """提取图像区域特征
        Args:
            images: [batch_size, 3, H, W]
        Returns:
            features: [batch_size, num_regions, embed_size]
        """
        with torch.no_grad():
            features = self.resnet(images)
            batch_size, channels, height, width = features.size()
            
            global_feat = F.adaptive_avg_pool2d(features, (1, 1)).view(batch_size, channels, -1)
            spatial_feat = F.adaptive_avg_pool2d(features, (7, 7)).view(batch_size, channels, -1)
            
            combined = torch.cat([global_feat, spatial_feat], dim=2)
            combined = combined.permute(0, 2, 1)
            
            features = self.linear(combined)
        
        return features


class TextWordEncoder(nn.Module):
    """文本词汇编码器：提取文本词汇特征"""
    def __init__(self, embed_size: int = 1024, use_bert: bool = True):
        super().__init__()
        self.use_bert = use_bert
        self.embed_size = embed_size
        
        if use_bert:
            self.bert = AutoModel.from_pretrained('bert-base-uncased', use_safetensors=True)
            bert_dim = self.bert.config.hidden_size
            self.linear = nn.Linear(bert_dim, embed_size)
        else:
            self.embed = nn.Embedding(30522, 300)
            self.gru = nn.GRU(300, embed_size // 2, num_layers=1,
                            batch_first=True, bidirectional=True)
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """提取文本词汇特征
        Args:
            x: 输入文本（BERT tokens或word indices）
            lengths: 序列长度（用于GRU）
        Returns:
            features: [batch_size, seq_len, embed_size]
        """
        if self.use_bert:
            outputs = self.bert(x['input_ids'], attention_mask=x['attention_mask'])
            word_features = outputs.last_hidden_state
            word_features = self.linear(word_features)
        else:
            embedded = self.embed(x)
            if lengths is not None:
                embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
            output, _ = self.gru(embedded)
            if lengths is not None:
                output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            word_features = output
        
        return word_features


class CrossAttention(nn.Module):
    """交叉注意力模块"""
    def __init__(self, embed_size: int = 1024):
        super().__init__()
        self.embed_size = embed_size
        self.W = nn.Linear(embed_size, embed_size)
        self.v = nn.Linear(embed_size, 1)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算交叉注意力
        Args:
            query: [batch, query_len, embed_size]
            key: [batch, key_len, embed_size]
            value: [batch, value_len, embed_size] (通常等于key)
            mask: [batch, key_len] 注意力掩码
        Returns:
            attended: [batch, query_len, embed_size]
            attention_weights: [batch, query_len, key_len]
        """
        query_proj = self.W(query)
        scores = torch.bmm(query_proj, key.transpose(1, 2))
        scores = scores / (self.embed_size ** 0.5)
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=2)
        
        attended = torch.bmm(attention_weights, value)
        
        return attended, attention_weights


class SCANService:
    """SCAN服务：基于堆叠交叉注意力的图像文本匹配"""
    def __init__(self, embed_size: int = 1024, use_bert: bool = True, device: Optional[str] = None):
        self.embed_size = embed_size
        self.use_bert = use_bert
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.image_encoder = ImageRegionEncoder(embed_size=embed_size).to(self.device)
        self.text_encoder = TextWordEncoder(embed_size=embed_size, use_bert=use_bert).to(self.device)
        
        self.image_to_text_attention = CrossAttention(embed_size=embed_size).to(self.device)
        self.text_to_image_attention = CrossAttention(embed_size=embed_size).to(self.device)
        
        self.image_encoder.eval()
        self.text_encoder.eval()
        self.image_to_text_attention.eval()
        self.text_to_image_attention.eval()
        
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
        """编码图像为区域特征"""
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
            embeds.append(features.detach().cpu())
        
        return torch.cat(embeds, dim=0) if embeds else torch.empty((0, 50, self.embed_size))
    
    @torch.inference_mode()
    def encode_texts(self, texts: List[str], batch_size: int = 64) -> torch.Tensor:
        """编码文本为词汇特征"""
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
                features = self.text_encoder(encoded, lengths=None)
            else:
                raise NotImplementedError("GRU编码器需要词汇表，请使用BERT模式")
            
            embeds.append(features.detach().cpu())
        
        return torch.cat(embeds, dim=0) if embeds else torch.empty((0, 77, self.embed_size))
    
    @torch.inference_mode()
    def compute_similarity(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """计算图像和文本的相似度分数
        Args:
            image_features: [batch, num_regions, embed_size] 或 [batch, embed_size]
            text_features: [batch, seq_len, embed_size] 或 [batch, embed_size]
        Returns:
            similarity_scores: [batch] 相似度分数
        """
        if len(image_features.shape) == 2:
            image_features = image_features.unsqueeze(1)
        if len(text_features.shape) == 2:
            text_features = text_features.unsqueeze(1)
        
        image_features = image_features.to(self.device)
        text_features = text_features.to(self.device)
        
        img_attended, _ = self.image_to_text_attention(
            query=image_features,
            key=text_features,
            value=text_features
        )
        
        txt_attended, _ = self.text_to_image_attention(
            query=text_features,
            key=image_features,
            value=image_features
        )
        
        img_agg = img_attended.mean(dim=1)
        txt_agg = txt_attended.mean(dim=1)
        
        img_agg = F.normalize(img_agg, p=2, dim=1)
        txt_agg = F.normalize(txt_agg, p=2, dim=1)
        similarity = (img_agg * txt_agg).sum(dim=1)
        
        return similarity.cpu()
    
    def get_feature_dim(self) -> int:
        """返回特征维度"""
        return self.embed_size

