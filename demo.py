import os
import io
import base64
from pathlib import Path
from PIL import Image

from config.settings import load_config
from utils.model_load import get_yolo_model, get_caption_model_processor
from utils.util import check_ocr_box, get_som_labeled_img

# Use Path for robust path handling
input_dir = Path(load_config()['INPUT_IMG_DIR'])
output_dir = Path(load_config()['OUTPUT_IMG_DIR'])

# Ensure output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

# Default to screenshot.png if the specific file is not found
image_filename = '微信图片_20250224054531.jpg'
image_read_path = input_dir / image_filename

if not image_read_path.exists():
    # Fallback to a known existing file or warn
    fallback_path = Path('screenshot.png')
    if fallback_path.exists():
        print(f"Warning: {image_read_path} not found. Using {fallback_path} instead.")
        image_read_path = fallback_path
    else:
        print(f"Warning: {image_read_path} not found and no fallback available.")

image_save_path = output_dir / 'out_image.png'

try:
    image = Image.open(image_read_path)
except FileNotFoundError:
    print(f"Error: Image file not found at {image_read_path}")
    exit(1)

box_overlay_ratio = image.size[0] / 3200
draw_bbox_config = {
    'text_scale': 0.8 * box_overlay_ratio,
    'text_thickness': max(int(2 * box_overlay_ratio), 1),
    'text_padding': max(int(3 * box_overlay_ratio), 1),
    'thickness': max(int(3 * box_overlay_ratio), 1),
}

ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_read_path, display_img=False,
                                                output_bb_format='xyxy',
                                                goal_filtering=None,
                                                easyocr_args={'paragraph': False, 'text_threshold': 0.9},
                                                use_paddleocr=False)
text, ocr_bbox = ocr_bbox_rslt
print('text:', text, 'ocr_bbox:', ocr_bbox)

# Fix path separators for cross-platform compatibility
yolo_model_path = Path('weights/icon_detect/model.pt')
caption_model_path = Path('weights/icon_caption_florence')

yolo_model = get_yolo_model(model_path=str(yolo_model_path))
caption_model_processor = get_caption_model_processor(model_name="florence2",
                                                      model_name_or_path=str(caption_model_path),
                                                      device='cuda')

dino_labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image_read_path, yolo_model,
                                                                              BOX_THRESHOLD=0.001,
                                                                              output_coord_in_ratio=True,
                                                                              ocr_bbox=ocr_bbox,
                                                                              draw_bbox_config=draw_bbox_config,
                                                                              caption_model_processor=caption_model_processor,
                                                                              ocr_text=text,
                                                                              iou_threshold=0.9,
                                                                              imgsz=None, )

image = Image.open(io.BytesIO(base64.b64decode(dino_labeled_img)))
image.show()

# Save the image
image.save(image_save_path)
print(f"Saved output to {image_save_path}")

parsed_content_list = '\n'.join([f'icon {i}: ' + str(v) for i,v in enumerate(parsed_content_list)])
print(parsed_content_list)
