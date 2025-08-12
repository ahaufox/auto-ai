import os,io,base64
from config.settings import load_config
from PIL import Image

from model_load import get_yolo_model, get_caption_model_processor

from utils.util import check_ocr_box,get_som_labeled_img

image_read_path = load_config()['INPUT_IMG_DIR'] + '微信图片_20250224054531.jpg'
image_save_path =load_config()['OUTPUT_IMG_DIR'] +'out_image.png'
image = Image.open(image_read_path)
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
print('text:',text,'ocr_bbox:', ocr_bbox)

yolo_model = get_yolo_model(model_path='weights\\icon_detect\\model.pt')
caption_model_processor = get_caption_model_processor(model_name="florence2",
                                                      model_name_or_path="weights/icon_caption_florence",device='cuda')
dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image_read_path, yolo_model,
                                                                              BOX_TRESHOLD=0.001,
                                                                              output_coord_in_ratio=True,
                                                                              ocr_bbox=ocr_bbox,
                                                                              draw_bbox_config=draw_bbox_config,
                                                                              caption_model_processor=caption_model_processor,
                                                                              ocr_text=text,
                                                                              iou_threshold= 0.9,
                                                                              imgsz=None, )

image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
image.show()
if not os.path.exists(image_save_path):
    with open(image_save_path, 'w') as f:
        pass
image.save(image_save_path)

parsed_content_list = '\n'.join([f'icon {i}: ' + str(v) for i,v in enumerate(parsed_content_list)])
print(parsed_content_list)