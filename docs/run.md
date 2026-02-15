# OmniParser 运行指南

本文档介绍如何运行 OmniParser 的演示脚本和 API 服务器。

## 1. 运行演示 (Demo)

`demo.py` 是一个简单的演示脚本，它会加载模型并处理一张测试图片。

### 运行步骤
确保你已经按照 [开发指南](develop.md) 完成了环境配置和权重下载。

```bash
python demo.py
```

该脚本会：
1. 读取默认图片（或 `screenshot.png`）。
2. 使用 YOLO 进行图标检测。
3. 使用 Florence-2 进行图标描述。
4. 使用 OCR 识别文字。
5. 将结果保存到 `imgs/out/out_image.png`。

## 2. 运行 API 服务器

OmniParser 提供了一个基于 FastAPI 的服务器，可以通过 HTTP 请求进行图片解析。

### 启动服务器

```bash
python server/gui_server.py
```

服务器默认将在 `http://0.0.0.0:8007` 启动。

### API 使用示例

#### 解析图片 (POST /parse/)

**请求 URL:** `http://localhost:8007/parse/`

**请求体 (JSON):**
```json
{
  "base64_image": "<图片的Base64编码字符串>"
}
```

**响应:**
```json
{
  "som_image_base64": "<标注后的图片Base64>",
  "parsed_content_list": ["图标1描述", "图标2描述", ...],
  "latency": 1.23
}
```

### 参数配置

可以通过命令行参数调整服务器配置：

```bash
python server/gui_server.py --port 8008 --device cpu --BOX_THRESHOLD 0.1
```

主要参数说明：
- `--host`: 监听地址 (默认 0.0.0.0)
- `--port`: 监听端口 (默认 8007)
- `--device`: 运行设备 (cuda 或 cpu)
- `--BOX_THRESHOLD`: 图标检测阈值 (默认 0.05)
