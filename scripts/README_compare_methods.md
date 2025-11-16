# 方法对比可视化脚本使用说明

## 功能描述

`compare_methods.py` 脚本用于对比不同图像-文本匹配方法（CLIP、VSE++、SCAN）在相同图像-文本对上的表现，生成类似CLIP论文中的可视化表格。

## 功能特点

1. **从数据集提取图像-文本对**：自动从CSV文件中提取图像和对应的描述信息
2. **多方法对比**：同时使用CLIP、VSE++、SCAN三种方法计算相似度
3. **可视化展示**：生成包含以下信息的对比图表：
   - 输入图像
   - 正确标签（Correct Label）
   - 正确排名和概率（Correct Rank & Probability）
   - Top-5预测标签及其相似度分数（带颜色条）

## 使用方法

### 1. 确保索引已构建

在运行对比脚本之前，需要先为所有方法构建索引：

```bash
# 构建CLIP索引
python scripts/build_index.py --method clip

# 构建VSE++索引
python scripts/build_index.py --method vse

# 构建SCAN索引
python scripts/build_index.py --method scan
```

### 2. 运行对比脚本

```bash
# 使用默认参数
python scripts/compare_methods.py

# 自定义参数
python scripts/compare_methods.py \
    --csv data/dataset_en.csv \
    --image_root data/images \
    --index_dir data/index \
    --output method_comparison.png \
    --num_samples 3 \
    --dataset_name "Custom Dataset"
```

### 3. 参数说明

- `--csv`: CSV数据集文件路径（默认：`data/dataset_en.csv`）
- `--image_root`: 图像根目录（默认：`data/images`）
- `--index_dir`: 索引文件目录（默认：`data/index`）
- `--output`: 输出图像路径（默认：`method_comparison.png`）
- `--num_samples`: 要对比的样本数量（默认：3）
- `--dataset_name`: 数据集名称，用于显示（默认：`Custom Dataset`）

## 输出说明

脚本会为每个样本生成一个对比图像文件，文件名格式为：`method_comparison_sample{N}.png`

每个图像包含：
- **三行**：分别对应CLIP、VSE++、SCAN三种方法
- **三列**：
  1. **左侧**：输入图像
  2. **中间**：正确标签、排名和概率信息
  3. **右侧**：Top-5预测标签及其相似度条形图

### 颜色说明

- **绿色条**：正确标签（Correct Label）
- **蓝色条**：相关描述（来自同一图像的global_caption或local_caption）
- **橙色条**：其他描述（来自其他图像的描述，作为干扰项）

## 注意事项

1. **索引构建**：确保所有三种方法的索引都已构建完成
2. **内存要求**：SCAN方法需要加载所有图像特征到内存，可能需要较大内存
3. **计算时间**：首次运行可能需要较长时间，因为需要加载模型和索引
4. **图像路径**：确保CSV中的文件名与`image_root`目录中的实际文件名匹配

## 示例输出

生成的图像将显示：
- 每个方法在相同图像-文本对上的表现
- 正确标签的排名（Rank 1表示最佳）
- 所有候选描述的相似度分数
- 直观的颜色编码帮助快速识别正确和错误预测

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

