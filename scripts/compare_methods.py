#!/usr/bin/env python3
"""
对比不同方法的图像-文本匹配性能
生成可视化表格，展示不同方法在相同图像-文本对上的表现
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

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
        import traceback
        traceback.print_exc()
        return 0.0


def get_all_candidates(dataset: List[Dict], current_item: Dict) -> List[str]:
    """获取所有候选描述（包括当前项的描述和其他项的描述）"""
    candidates = []
    
    # 添加当前项的global_caption和local_captions
    if current_item['global_caption']:
        candidates.append(('global', current_item['global_caption']))
    for local_cap in current_item['local_captions']:
        candidates.append(('local', local_cap))
    
    # 从其他项中随机选择一些描述作为干扰项
    other_items = [item for item in dataset if item['filename'] != current_item['filename']]
    np.random.shuffle(other_items)
    
    for item in other_items[:10]:  # 选择10个其他项
        if item['global_caption']:
            candidates.append(('other_global', item['global_caption']))
        for local_cap in item['local_captions'][:1]:  # 每个项只取一个local_caption
            candidates.append(('other_local', local_cap))
    
    return candidates


def evaluate_method(
    index_service: IndexService,
    image_path: str,
    correct_texts: List[str],
    candidate_texts: List[Tuple[str, str]]
) -> Dict:
    """评估方法在给定图像-文本对上的表现"""
    results = []
    
    for label_type, text in candidate_texts:
        similarity = compute_similarity_for_pair(index_service, image_path, text)
        is_correct = text in correct_texts
        results.append({
            'text': text,
            'similarity': similarity,
            'label_type': label_type,
            'is_correct': is_correct
        })
    
    # 按相似度排序
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    # 找到正确标签的排名
    correct_ranks = []
    for i, result in enumerate(results):
        if result['is_correct']:
            correct_ranks.append(i + 1)
    
    best_correct_rank = min(correct_ranks) if correct_ranks else None
    best_correct_prob = results[best_correct_rank - 1]['similarity'] if best_correct_rank else 0.0
    
    return {
        'results': results,
        'best_correct_rank': best_correct_rank,
        'best_correct_prob': best_correct_prob,
        'top_predictions': results[:5]  # Top 5预测
    }


def create_comparison_plot(
    image_path: str,
    correct_label: str,
    methods_results: Dict[str, Dict],
    output_path: str,
    dataset_name: str = "Custom Dataset"
):
    """创建对比可视化图表，类似CLIP论文中的展示方式"""
    num_methods = len(methods_results)
    fig = plt.figure(figsize=(18, 6 * num_methods))
    
    # 加载图像
    try:
        img = Image.open(image_path)
        # 调整图像大小以适应显示
        img.thumbnail((300, 300), Image.Resampling.LANCZOS)
        img_array = np.array(img)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)
    
    method_names = list(methods_results.keys())
    
    for method_idx, method_name in enumerate(method_names):
        result = methods_results[method_name]
        top_predictions = result['top_predictions']
        best_rank = result['best_correct_rank']
        best_prob = result['best_correct_prob']
        
        # 为每个方法创建一行，包含3列：图像、信息、预测
        # 增加列间距，避免文字重叠
        gs = GridSpec(1, 3, figure=fig, 
                     left=0.03, right=0.97,
                     top=0.98 - method_idx * (0.95 / num_methods),
                     bottom=0.02 + (num_methods - method_idx - 1) * (0.95 / num_methods),
                     wspace=0.25, hspace=0.1,
                     width_ratios=[1.2, 1.5, 2.0])  # 调整列宽比例
        
        # 左侧：图像
        ax_img = fig.add_subplot(gs[0, 0])
        ax_img.imshow(img_array)
        ax_img.axis('off')
        method_title = method_name.upper()
        if method_name == 'vse':
            method_title = 'VSE++'
        ax_img.set_title(method_title, fontsize=16, fontweight='bold', pad=10)
        
        # 中间：正确标签和排名信息
        ax_info = fig.add_subplot(gs[0, 1])
        ax_info.axis('off')
        # 设置x轴范围，限制文字区域
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)
        
        # 格式化正确标签文本，限制每行长度
        correct_text = correct_label
        if len(correct_text) > 50:  # 减少每行长度
            # 智能换行
            words = correct_text.split()
            lines = []
            current_line = []
            current_len = 0
            for word in words:
                if current_len + len(word) + 1 > 50:  # 减少到50字符
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_len = len(word)
                else:
                    current_line.append(word)
                    current_len += len(word) + 1
            if current_line:
                lines.append(' '.join(current_line))
            correct_text = '\n'.join(lines)
        
        info_y = 0.75
        ax_info.text(0.05, info_y, 'Correct Label:', fontsize=12, fontweight='bold',
                    verticalalignment='top', transform=ax_info.transAxes,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax_info.text(0.05, info_y - 0.12, correct_text, fontsize=10,
                    verticalalignment='top', transform=ax_info.transAxes,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
        
        # 排名和概率信息
        rank_y = 0.45
        if best_rank:
            rank_text = f"Correct Rank: {best_rank}"
            prob_text = f"Probability: {best_prob:.2%}"
            color = 'green' if best_rank == 1 else 'orange'
        else:
            rank_text = "Correct Rank: N/A"
            prob_text = "Probability: 0.00%"
            color = 'red'
        
        ax_info.text(0.05, rank_y, rank_text, fontsize=12, fontweight='bold',
                    color=color, verticalalignment='top', transform=ax_info.transAxes,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax_info.text(0.05, rank_y - 0.08, prob_text, fontsize=11,
                    color=color, verticalalignment='top', transform=ax_info.transAxes,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # 右侧：预测标签和概率条
        ax_pred = fig.add_subplot(gs[0, 2])
        
        # 获取最大相似度用于归一化
        max_sim = max(pred['similarity'] for pred in top_predictions) if top_predictions else 1.0
        min_sim = min(pred['similarity'] for pred in top_predictions) if top_predictions else 0.0
        
        # 绘制预测标签和概率条
        y_positions = []
        for i, pred in enumerate(top_predictions):
            similarity = pred['similarity']
            text = pred['text']
            is_correct = pred['is_correct']
            
            # 归一化相似度（假设相似度在合理范围内，对于余弦相似度通常是-1到1）
            # 如果相似度是负数，先映射到0-1
            if similarity < 0:
                normalized_score = (similarity + 1) / 2  # 映射-1到1 -> 0到1
            else:
                # 如果都是正数，使用相对归一化
                if max_sim > min_sim:
                    normalized_score = (similarity - min_sim) / (max_sim - min_sim)
                else:
                    normalized_score = 0.5
            
            y_pos = len(top_predictions) - i - 1
            y_positions.append(y_pos)
            
            # 选择颜色
            if is_correct:
                color = '#2ecc71'  # 绿色
                edgecolor = '#27ae60'
            else:
                if pred['label_type'] in ['global', 'local']:
                    color = '#3498db'  # 蓝色
                    edgecolor = '#2980b9'
                else:
                    color = '#e67e22'  # 橙色
                    edgecolor = '#d35400'
            
            # 绘制条形图
            bar = ax_pred.barh(y_pos, normalized_score, height=0.7, 
                              color=color, alpha=0.8, edgecolor=edgecolor, linewidth=1.5)
            
            # 添加文本标签（在条形图左侧，使用数据坐标而非轴坐标，避免重叠）
            text_short = text
            if len(text) > 45:  # 减少文本长度
                # 智能截断
                text_short = text[:42] + '...'
            
            # 使用数据坐标，确保文本在条形图区域内
            ax_pred.text(-0.35, y_pos, text_short, fontsize=9,
                        verticalalignment='center', horizontalalignment='right',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                 edgecolor='gray', alpha=0.8, linewidth=0.5))
            
            # 添加相似度分数（在条形图右侧）
            score_text = f'{similarity:.3f}'
            ax_pred.text(normalized_score + 0.01, y_pos, score_text, 
                        fontsize=9, verticalalignment='center',
                        fontweight='bold' if is_correct else 'normal')
        
        # 调整x轴范围，为左侧文本标签留出空间
        ax_pred.set_xlim(-0.6, 1.15)
        ax_pred.set_ylim(-0.5, len(top_predictions) - 0.5)
        ax_pred.set_xlabel('Normalized Similarity Score', fontsize=11, fontweight='bold')
        ax_pred.set_ylabel('Predicted Labels', fontsize=11, fontweight='bold')
        ax_pred.grid(axis='x', alpha=0.3, linestyle='--')
        ax_pred.set_yticklabels([])
        
        # 添加图例
        if method_idx == 0:
            legend_elements = [
                mpatches.Patch(facecolor='#2ecc71', label='Correct Label', alpha=0.8),
                mpatches.Patch(facecolor='#3498db', label='Related Caption', alpha=0.8),
                mpatches.Patch(facecolor='#e67e22', label='Other Caption', alpha=0.8)
            ]
            ax_pred.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # 添加总标题
    fig.suptitle(f'{dataset_name} - Method Comparison', 
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved comparison plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare different image-text matching methods')
    parser.add_argument('--csv', default='data/dataset_en.csv', help='CSV file with dataset')
    parser.add_argument('--image_root', default='data/images', help='Root directory of images')
    parser.add_argument('--index_dir', default='data/index', help='Directory containing indices')
    parser.add_argument('--output', default='method_comparison.png', help='Output image path')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples to compare')
    parser.add_argument('--dataset_name', default='Custom Dataset', help='Dataset name for display')
    
    args = parser.parse_args()
    
    # 加载数据集
    print("Loading dataset...")
    dataset = load_dataset(args.csv)
    print(f"Loaded {len(dataset)} items from dataset")
    
    if len(dataset) == 0:
        print("Error: No data loaded from CSV file")
        return
    
    # 随机选择几个样本
    np.random.seed(42)
    selected_samples = np.random.choice(dataset, min(args.num_samples, len(dataset)), replace=False)
    
    # 初始化三种方法
    print("Initializing methods...")
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
    
    # 对每个样本进行评估
    for sample_idx, sample in enumerate(selected_samples):
        print(f"\nProcessing sample {sample_idx + 1}/{len(selected_samples)}: {sample['filename']}")
        
        # 构建图像路径
        image_path = os.path.join(args.image_root, sample['filename'])
        if not os.path.exists(image_path):
            print(f"  Warning: Image not found: {image_path}")
            continue
        
        # 获取正确标签（使用第一个local_caption作为正确标签）
        correct_texts = []
        if sample['local_captions']:
            correct_texts.append(sample['local_captions'][0])
        elif sample['global_caption']:
            correct_texts.append(sample['global_caption'])
        
        if not correct_texts:
            print(f"  Warning: No captions found for {sample['filename']}")
            continue
        
        correct_label = correct_texts[0]
        
        # 获取所有候选描述
        candidates = get_all_candidates(dataset, sample)
        
        # 对每种方法进行评估
        methods_results = {}
        for method_name, service in methods.items():
            print(f"  Evaluating {method_name}...")
            try:
                result = evaluate_method(service, image_path, correct_texts, candidates)
                methods_results[method_name] = result
                
                print(f"    Best rank: {result['best_correct_rank']}, "
                      f"Probability: {result['best_correct_prob']:.4f}")
            except Exception as e:
                print(f"    Error evaluating {method_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # 生成可视化
        if methods_results:
            output_path = args.output.replace('.png', f'_sample{sample_idx+1}.png')
            create_comparison_plot(
                image_path=image_path,
                correct_label=correct_label,
                methods_results=methods_results,
                output_path=output_path,
                dataset_name=args.dataset_name
            )
    
    print("\nComparison complete!")


if __name__ == '__main__':
    main()

