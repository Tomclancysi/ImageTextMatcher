#!/usr/bin/env python3
"""
计算不同方法对于相同的正例和负例预测的平均分数
生成表格展示结果
"""

import argparse
import csv
import json
import os
import sys
import random
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from tabulate import tabulate

# 添加项目根目录到路径
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.index_service import IndexService


def load_dataset(csv_path: str) -> List[Dict]:
    """从CSV文件加载数据集"""
    dataset = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get('url', '')
            cap_seg = row.get('cap_seg', '')
            
            if not url or not cap_seg:
                continue
            
            filename = os.path.basename(url)
            
            # 解析JSON描述
            try:
                description = json.loads(cap_seg)
            except json.JSONDecodeError:
                try:
                    fixed_str = cap_seg.replace('""', '"')
                    description = json.loads(fixed_str)
                except json.JSONDecodeError:
                    continue
            
            dataset.append({
                'filename': filename,
                'url': url,
                'global_caption': description.get('global_caption', ''),
                'local_captions': description.get('local_caption', [])
            })
    
    return dataset


def compute_similarity_for_pair(
    index_service: IndexService,
    image_path: str,
    text: str
) -> float:
    """计算单个图像-文本对的相似度"""
    try:
        if not index_service.is_ready():
            index_service.load_index()
        
        # 找到图像在索引中的位置
        img_idx = None
        image_basename = os.path.basename(image_path)
        for i, path in enumerate(index_service.meta):
            if os.path.basename(path) == image_basename:
                img_idx = i
                break
        
        if img_idx is None:
            # 如果图像不在索引中，直接编码计算
            img_features = index_service.encoder.encode_images([image_path])
            text_features = index_service.encoder.encode_texts([text])
            
            if index_service.method == "scan":
                similarity = index_service.encoder.compute_similarity(img_features, text_features)
                return float(similarity[0])
            else:
                # CLIP和VSE++：向量已经归一化，直接点积
                img_vec = img_features[0].numpy()
                text_vec = text_features[0].numpy()
                similarity = np.dot(img_vec, text_vec)
                return float(similarity)
        
        if index_service.method == "scan":
            # SCAN方法
            img_feat = index_service.image_features[img_idx:img_idx+1]
            text_features = index_service.encoder.encode_texts([text])
            similarity = index_service.encoder.compute_similarity(img_feat, text_features)
            return float(similarity[0])
        else:
            # CLIP和VSE++方法
            # 获取图像向量
            img_vector = index_service.index.reconstruct(int(img_idx))
            
            # 编码文本
            text_emb = index_service.encoder.encode_texts([text])
            text_vector = text_emb[0].numpy().astype(np.float32)
            
            # 计算余弦相似度（向量已经L2归一化，所以点积就是余弦相似度）
            similarity = np.dot(img_vector, text_vector)
            return float(similarity)
    except Exception as e:
        print(f"Error computing similarity for {image_path} and '{text[:30]}...': {e}")
        return None


def build_positive_negative_pairs(
    dataset: List[Dict],
    image_root: str,
    num_samples: int = 100,
    num_negatives_per_positive: int = 3
) -> Tuple[List[Tuple[str, str, bool]], List[Tuple[str, str, bool]]]:
    """
    构建正例和负例对
    返回: (正例列表, 负例列表)
    每个元素是 (image_path, text, is_positive) 的元组
    """
    positives = []
    negatives = []
    
    # 随机选择样本
    random.seed(42)
    np.random.seed(42)
    selected_samples = random.sample(
        dataset, 
        min(num_samples, len(dataset))
    )
    
    for sample in selected_samples:
        image_path = os.path.join(image_root, sample['filename'])
        if not os.path.exists(image_path):
            continue
        
        # 构建正例：使用当前图像的描述
        if sample['local_captions']:
            # 优先使用local_caption作为正例
            for local_cap in sample['local_captions'][:2]:  # 每个图像最多2个正例
                positives.append((image_path, local_cap, True))
        elif sample['global_caption']:
            positives.append((image_path, sample['global_caption'], True))
        
        # 构建负例：使用其他图像的描述
        other_samples = [item for item in dataset if item['filename'] != sample['filename']]
        np.random.shuffle(other_samples)
        
        negative_count = 0
        for other_sample in other_samples:
            if negative_count >= num_negatives_per_positive:
                break
            
            # 随机选择其他样本的global_caption或local_caption
            if other_sample['local_captions']:
                neg_text = np.random.choice(other_sample['local_captions'])
            elif other_sample['global_caption']:
                neg_text = other_sample['global_caption']
            else:
                continue
            
            negatives.append((image_path, neg_text, False))
            negative_count += 1
    
    return positives, negatives


def evaluate_method_on_pairs(
    index_service: IndexService,
    pairs: List[Tuple[str, str, bool]]
) -> Dict[str, List[float]]:
    """
    评估方法在正例和负例上的表现
    返回: {'positive': [scores...], 'negative': [scores...]}
    """
    positive_scores = []
    negative_scores = []
    
    for image_path, text, is_positive in pairs:
        similarity = compute_similarity_for_pair(index_service, image_path, text)
        if similarity is not None:
            if is_positive:
                positive_scores.append(similarity)
            else:
                negative_scores.append(similarity)
    
    return {
        'positive': positive_scores,
        'negative': negative_scores
    }


def create_results_table(
    methods_results: Dict[str, Dict[str, List[float]]],
    output_path: str = None
) -> pd.DataFrame:
    """
    创建结果表格
    methods_results: {method_name: {'positive': [scores...], 'negative': [scores...]}}
    """
    rows = []
    
    for method_name in ['clip', 'vse', 'scan']:
        if method_name not in methods_results:
            continue
        
        result = methods_results[method_name]
        pos_scores = result['positive']
        neg_scores = result['negative']
        
        # 计算统计信息
        pos_mean = np.mean(pos_scores) if pos_scores else 0.0
        pos_std = np.std(pos_scores) if pos_scores else 0.0
        pos_min = np.min(pos_scores) if pos_scores else 0.0
        pos_max = np.max(pos_scores) if pos_scores else 0.0
        
        neg_mean = np.mean(neg_scores) if neg_scores else 0.0
        neg_std = np.std(neg_scores) if neg_scores else 0.0
        neg_min = np.min(neg_scores) if neg_scores else 0.0
        neg_max = np.max(neg_scores) if neg_scores else 0.0
        
        # 计算分离度（正例平均分 - 负例平均分）
        separation = pos_mean - neg_mean
        
        # 方法名称格式化
        method_display = method_name.upper()
        if method_name == 'vse':
            method_display = 'VSE++'
        
        rows.append({
            'Method': method_display,
            'Positive Count': len(pos_scores),
            'Positive Mean': f'{pos_mean:.4f}',
            'Positive Std': f'{pos_std:.4f}',
            'Positive Min': f'{pos_min:.4f}',
            'Positive Max': f'{pos_max:.4f}',
            'Negative Count': len(neg_scores),
            'Negative Mean': f'{neg_mean:.4f}',
            'Negative Std': f'{neg_std:.4f}',
            'Negative Min': f'{neg_min:.4f}',
            'Negative Max': f'{neg_max:.4f}',
            'Separation': f'{separation:.4f}'
        })
    
    df = pd.DataFrame(rows)
    
    # 打印表格
    print("\n" + "="*120)
    print("Method Comparison Results")
    print("="*120)
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    print("="*120)
    
    # 保存到CSV
    if output_path:
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\nResults saved to {output_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Evaluate different methods on positive and negative pairs')
    parser.add_argument('--csv', default='data/dataset_en.csv', help='CSV file with dataset')
    parser.add_argument('--image_root', default='data/images', help='Root directory of images')
    parser.add_argument('--index_dir', default='data/index', help='Directory containing indices')
    parser.add_argument('--output', default='method_evaluation_results.csv', help='Output CSV file path')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to evaluate')
    parser.add_argument('--num_negatives', type=int, default=3, help='Number of negative pairs per positive')
    
    args = parser.parse_args()
    
    # 加载数据集
    print("Loading dataset...")
    dataset = load_dataset(args.csv)
    print(f"Loaded {len(dataset)} items from dataset")
    
    if len(dataset) == 0:
        print("Error: No data loaded from CSV file")
        return
    
    # 构建正例和负例对
    print(f"\nBuilding positive and negative pairs (samples: {args.num_samples}, negatives per positive: {args.num_negatives})...")
    positives, negatives = build_positive_negative_pairs(
        dataset, 
        args.image_root,
        num_samples=args.num_samples,
        num_negatives_per_positive=args.num_negatives
    )
    print(f"Built {len(positives)} positive pairs and {len(negatives)} negative pairs")
    
    # 初始化三种方法
    print("\nInitializing methods...")
    methods = {
        'clip': IndexService(
            image_root=args.image_root,
            index_dir=args.index_dir,
            method='clip',
            dataset_csv=args.csv
        ),
        'vse': IndexService(
            image_root=args.image_root,
            index_dir=args.index_dir,
            method='vse',
            dataset_csv=args.csv
        ),
        'scan': IndexService(
            image_root=args.image_root,
            index_dir=args.index_dir,
            method='scan',
            dataset_csv=args.csv
        )
    }
    
    # 加载索引
    for method_name, service in methods.items():
        print(f"Loading index for {method_name}...")
        try:
            service.load_index()
            print(f"  {method_name}: {len(service.meta)} images indexed")
        except Exception as e:
            print(f"  Warning: Failed to load {method_name} index: {e}")
            print(f"  Please build index first using: python scripts/build_index.py --method {method_name}")
    
    # 合并所有对进行评估
    all_pairs = positives + negatives
    
    # 对每种方法进行评估
    methods_results = {}
    for method_name, service in methods.items():
        print(f"\nEvaluating {method_name}...")
        try:
            result = evaluate_method_on_pairs(service, all_pairs)
            methods_results[method_name] = result
            
            pos_mean = np.mean(result['positive']) if result['positive'] else 0.0
            neg_mean = np.mean(result['negative']) if result['negative'] else 0.0
            print(f"  Positive pairs: {len(result['positive'])}, Mean score: {pos_mean:.4f}")
            print(f"  Negative pairs: {len(result['negative'])}, Mean score: {neg_mean:.4f}")
            print(f"  Separation: {pos_mean - neg_mean:.4f}")
        except Exception as e:
            print(f"  Error evaluating {method_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # 生成结果表格
    if methods_results:
        create_results_table(methods_results, output_path=args.output)
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()

