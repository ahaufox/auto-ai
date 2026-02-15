# OmniParser

OmniParser is a comprehensive screen parsing tool designed to detect and caption UI elements (icons and text) from screenshots. It leverages:
- **YOLO** for icon detection.
- **Florence-2** (or BLIP-2) for icon captioning/description.
- **EasyOCR** (or PaddleOCR) for text recognition.

## Prerequisites

- Python 3.8 or higher.
- CUDA-compatible GPU is highly recommended for performance (especially for Florence-2 and YOLO).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install PyTorch:**
    You must install a version of PyTorch compatible with your CUDA version. For example, for CUDA 11.8:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

4.  **Download Weights:**
    Ensure you have the model weights placed in the `weights/` directory as expected by the configuration:
    - `weights/icon_detect/model.pt` (YOLO model)
    - `weights/icon_caption_florence/` (Florence-2 model)

## Usage

### Running the Demo

To run a quick demonstration on an image:

```bash
python demo.py
```
This will process the default image (or fallback to `screenshot.png` if available) and save the result to `imgs/out/out_image.png`.

### Running the Server

To start the API server:

```bash
python server/gui_server.py
```

The server will start on `http://0.0.0.0:8007`.

### API Endpoints

#### POST `/parse/`

Parses a base64-encoded image.

**Request Body:**
```json
{
  "base64_image": "<base64_encoded_string>"
}
```

**Response:**
```json
{
  "som_image_base64": "<annotated_image_base64>",
  "parsed_content_list": ["list", "of", "descriptions"],
  "latency": 1.23
}
```

## Configuration

Configuration is managed via `config/default.yaml` and environment variables. Key settings include:
- `INPUT_IMG_DIR`: Directory for input images.
- `OUTPUT_IMG_DIR`: Directory for output results.
- `BOX_THRESHOLD`: Confidence threshold for icon detection (default: 0.05).
