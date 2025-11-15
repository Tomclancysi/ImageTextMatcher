项目：ImageTextMatcher - 文本到图片语义检索

一、功能概述
- 使用预训练 CLIP 模型将文本与图片映射到共享向量空间
- 基于 FAISS 的向量索引实现近似最近邻检索
- Flask 提供 Web 与 REST API：输入自然语言，返回 Top-N 图片

二、目录结构
- app/
  - main.py               Flask 应用入口
  - services/
    - clip_service.py     CLIP 编码器（文本/图片）
    - index_service.py    FAISS 索引构建/加载/检索
  - templates/
    - index.html          首页查询页
    - results.html        结果页
  - static/
    - styles.css          基础样式
- scripts/
  - build_index.py        构建索引脚本
- data/
  - images/               放置图片库（自建）
  - index/                索引输出（自动生成）
- requirements.txt        依赖清单
- readme.txt              本说明

三、环境要求（Windows 11/10）
1) 安装 Python 3.10+（建议 64 位）
2) 建议创建虚拟环境：
   python -m venv .venv
   .venv\\Scripts\\activate
3) 安装依赖：
   pip install -r requirements.txt
   注：首次运行会自动下载 CLIP 模型（需联网）

四、准备数据
- 将待检索图片放入 data/images/ 下，可多级子目录。
- 支持格式：.jpg/.jpeg/.png/.bmp/.gif/.webp

五、构建索引
- 方式一（默认路径）：
   python scripts/build_index.py
- 方式二（自定义路径）：
   python scripts/build_index.py --image_root D:\\images --index_dir D:\\index

六、运行服务
- Windows PowerShell：
   $env:ITM_IMAGE_ROOT="D:\\CodeWorks\\Pythons\\ImageTextMatcher\\data\\images"
   $env:ITM_INDEX_DIR="D:\\CodeWorks\\Pythons\\ImageTextMatcher\\data\\index"

   python -m app.main
   
- 浏览器访问：http://localhost:5000/
- REST API 示例：
   GET /api/search?q=red%20car&k=10

七、注意事项
- 首次检索前需先构建索引。
- 默认模型：openai/clip-vit-base-patch32。如需更换，可设置环境变量 ITM_MODEL_NAME。
- 检索相似度基于归一化向量的内积（等价于余弦相似度）。

八、图片下载脚本
- 使用 scripts/download_images.py 从 dataset.csv 下载图片
- 支持断点续传、错误重试、进度显示
- 使用方法：
  python scripts/download_images.py --csv dataset.csv --output data/images
  python scripts/download_images.py --max 100  # 限制下载数量
  python scripts/download_images.py --start 50  # 从第50个开始（断点续传）

九、下一步计划
- 构建小规模领域数据集，进行轻量微调
- 增强召回与排序，并加入拼写纠错、查询改写
- 增加 Recall@K、MRR 与延迟监控的评测脚本
