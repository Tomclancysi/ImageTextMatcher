# 方法评估脚本使用说明

## 功能描述

`evaluate_methods.py` 脚本用于计算不同图像-文本匹配方法（CLIP、VSE++、SCAN）在相同的正例和负例上的平均相似度分数，并生成统计表格。

## 功能特点

1. **构建正例和负例**：自动从数据集中构建匹配的图像-文本对（正例）和不匹配的对（负例）
2. **多方法评估**：同时使用CLIP、VSE++、SCAN三种方法计算相似度
3. **统计分析**：计算正例和负例的平均分、标准差、最小值、最大值等统计信息
4. **表格输出**：生成包含所有统计信息的表格，并保存为CSV文件

## 使用方法

### 1. 确保索引已构建

在运行评估脚本之前，需要先为所有方法构建索引：

```bash
# 构建CLIP索引
python scripts/build_index.py --method clip

# 构建VSE++索引
python scripts/build_index.py --method vse

# 构建SCAN索引
python scripts/build_index.py --method scan
```

### 2. 运行评估脚本

```bash
# 使用默认参数
python scripts/evaluate_methods.py

# 自定义参数
python scripts/evaluate_methods.py \
    --csv data/dataset_en.csv \
    --image_root data/images \
    --index_dir data/index \
    --output method_evaluation_results.csv \
    --num_samples 100 \
    --num_negatives 3
```

### 3. 参数说明

- `--csv`: CSV数据集文件路径（默认：`data/dataset_en.csv`）
- `--image_root`: 图像根目录（默认：`data/images`）
- `--index_dir`: 索引文件目录（默认：`data/index`）
- `--output`: 输出CSV文件路径（默认：`method_evaluation_results.csv`）
- `--num_samples`: 要评估的样本数量（默认：100）
- `--num_negatives`: 每个正例对应的负例数量（默认：3）

## 输出说明

脚本会在控制台打印一个格式化的表格，并保存为CSV文件。

### 表格包含的列

- **Method**: 方法名称（CLIP、VSE++、SCAN）
- **Positive Count**: 正例数量
- **Positive Mean**: 正例平均相似度分数
- **Positive Std**: 正例标准差
- **Positive Min**: 正例最小分数
- **Positive Max**: 正例最大分数
- **Negative Count**: 负例数量
- **Negative Mean**: 负例平均相似度分数
- **Negative Std**: 负例标准差
- **Negative Min**: 负例最小分数
- **Negative Max**: 负例最大分数
- **Separation**: 分离度（正例平均分 - 负例平均分）

### 分离度（Separation）

分离度是一个重要的指标，表示正例和负例之间的区分度：
- **正值越大**：方法越好地区分正例和负例
- **接近0或负值**：方法难以区分正例和负例

## 正例和负例的构建方式

### 正例（Positive Pairs）
- 使用图像自身的描述（优先使用`local_caption`，如果没有则使用`global_caption`）
- 每个图像最多生成2个正例（如果有多个`local_caption`）

### 负例（Negative Pairs）
- 使用其他图像的描述与当前图像配对
- 每个正例对应N个负例（由`--num_negatives`参数控制，默认3个）
- 随机选择其他图像的描述作为负例

## 示例输出

```
========================================================================================================================
Method Comparison Results
========================================================================================================================
+----------+-----------------+----------------+----------------+----------------+----------------+------------------+
| Method   | Positive Count  | Positive Mean  | Positive Std   | Positive Min   | Positive Max   | Negative Count   |
+----------+-----------------+----------------+----------------+----------------+----------------+------------------+
| CLIP     | 150             | 0.3245         | 0.0523         | 0.2100         | 0.4500         | 300              |
| VSE++    | 150             | 0.2856         | 0.0489         | 0.1800         | 0.4200         | 300              |
| SCAN     | 150             | 0.3012         | 0.0512         | 0.1900         | 0.4300         | 300              |
+----------+-----------------+----------------+----------------+----------------+----------------+------------------+
...
+----------+------------------+
| Separation |
+----------+------------------+
| 0.1234    |
| 0.0987    |
| 0.1123    |
+----------+------------------+
```

## 注意事项

1. **索引构建**：确保所有三种方法的索引都已构建完成
2. **样本数量**：`--num_samples`参数控制评估的样本数量，较大的值会得到更可靠的统计结果，但需要更长的计算时间
3. **负例数量**：`--num_negatives`参数控制每个正例对应的负例数量，建议设置为2-5之间
4. **计算时间**：首次运行可能需要较长时间，因为需要加载模型和索引
5. **内存要求**：SCAN方法需要加载所有图像特征到内存，可能需要较大内存

## 故障排除

### 问题：找不到图像文件
- 检查`--image_root`路径是否正确
- 确认CSV中的文件名与实际文件名匹配

### 问题：索引未找到
- 运行`scripts/build_index.py`为相应方法构建索引
- 检查`--index_dir`路径是否正确

### 问题：内存不足
- 减少`--num_samples`参数值
- 对于SCAN方法，考虑使用更小的批次大小构建索引

### 问题：缺少依赖库
- 安装pandas和tabulate：`pip install pandas tabulate`

## 与compare_methods.py的区别

- **compare_methods.py**：生成可视化图表，展示每个样本的详细预测结果
- **evaluate_methods.py**：生成统计表格，展示所有样本的聚合统计信息

两个脚本可以配合使用，从不同角度评估方法的性能。

