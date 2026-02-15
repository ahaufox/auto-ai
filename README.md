# OmniParser

OmniParser 是一个用于屏幕解析的综合工具，能够检测并描述 UI 元素（图标和文本）。它结合了以下技术：
- **YOLO**: 用于图标检测
- **Florence-2** (或 BLIP-2): 用于图标描述
- **EasyOCR** (或 PaddleOCR): 用于文字识别

## 文档导航

请参考 `docs/` 目录下的详细文档：

- [🛠️ 开发指南 (Development Guide)](docs/develop.md)
  - 环境搭建、依赖安装、权重下载说明。

- [🚀 运行指南 (Run Guide)](docs/run.md)
  - 如何运行演示脚本 (`demo.py`) 和 API 服务器 (`server/gui_server.py`)。

## 快速开始

1. **安装依赖**:
   ```bash
   pip install -r requirements.txt
   ```

2. **运行演示**:
   ```bash
   python demo.py
   ```

更多详细信息，请查阅上述文档链接。
