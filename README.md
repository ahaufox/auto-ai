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

## 🤝 贡献指南 (Contributing)

我们非常欢迎并感谢您的贡献！为了确保代码质量和协作顺畅，请在提交 Pull Request (PR) 前阅读以下指南：

1. **Fork 本仓库**: 点击右上角的 Fork 按钮，将项目复制到您的 GitHub 账户。
2. **创建分支**: 在您的仓库中创建一个新分支，分支名应清晰描述您的更改（例如 `feature/add-new-model` 或 `fix/typo-in-docs`）。
3. **提交更改**:
    - 请确保代码风格与现有代码保持一致。
    - 如果添加了新功能，请务必添加相应的测试。
    - 提交信息 (Commit Message) 应清晰明了。
4. **提交 PR**: 将您的分支推送到 GitHub，然后向本仓库的 `main` 分支提交 Pull Request。
5. **代码审查**: 请耐心等待维护者的审查，我们会尽快反馈。
