# OmniParser 开发指南

本文档介绍如何搭建 OmniParser 的开发环境，包括依赖安装和权重下载。

## 1. 环境准备

### 基础要求
- Python 3.8 或更高版本
- 推荐使用 CUDA 兼容的 GPU (用于 Florence-2 和 YOLO 加速)

### 克隆代码仓库

```bash
git clone <repository_url>
cd <repository_name>
```

### 安装依赖

```bash
pip install -r requirements.txt
```

### 安装 PyTorch

请根据你的 CUDA 版本安装对应的 PyTorch。例如，对于 CUDA 11.8：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

如果需要使用 PaddleOCR GPU 版本：
```bash
python -m pip install paddlepaddle-gpu==2.6.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

## 2. 权重下载

项目运行需要预先下载模型权重并放置在 `weights/` 目录下。

### 目录结构要求
```
weights/
├── icon_detect/
│   └── model.pt              # YOLO 模型权重
└── icon_caption_florence/    # Florence-2 模型文件夹
    ├── config.json
    ├── pytorch_model.bin
    └── ...
```

请确保下载并将权重放置在上述位置。

## 3. 项目结构说明

- `demo.py`: 演示脚本入口
- `server/`: API 服务器代码
  - `gui_server.py`: 服务器入口
  - `omniparser.py`: 解析逻辑核心类
- `utils/`: 工具函数
  - `util.py`: 图像处理、OCR 调用等工具
  - `model_load.py`: 模型加载辅助函数
- `config/`: 配置文件
- `imgs/`: 输入输出图片目录
- `weights/`: 模型权重目录
