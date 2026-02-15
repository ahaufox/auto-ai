# 从ultralytics导入YOLO（已注释）
import os
import io
import base64
import time
from PIL import Image, ImageDraw, ImageFont
import json
import requests
# 工具函数
import os
from openai import AzureOpenAI

import json
import sys
import os
import cv2
import numpy as np
# %matplotlib inline （Jupyter Notebook专用，已注释）
from matplotlib import pyplot as plt
from typing import List, Optional, Union, Tuple
import time
import base64


import os
import ast
import torch
from typing import Tuple, List, Union
from torchvision.ops import box_convert
import re
from torchvision.transforms import ToPILImage
import supervision as sv
import torchvision.transforms as T
from supervision.draw.color import Color, ColorPalette
from supervision.detection.core import Detections
# from paddleocr import PaddleOCR
# paddle_ocr = PaddleOCR(
#             lang='ch',  # 语言设置为中文
#             use_angle_cls=False,  # 不使用角度分类器
#             use_gpu=False,  # 不使用GPU加速（与同一进程中的PyTorch会冲突）
#             show_log=False,  # 不显示日志信息
#             max_batch_size=32,  # 最大批处理大小
#             use_dilation=np.True_,  # 使用膨胀操作，提高识别准确率
#             det_db_score_mode='slow',  # 使用慢速评分模式，提高检测准确率
#             rec_batch_num=32)  # 识别批处理数量

def get_project_dir():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'


class BoxAnnotator:
    """
    用于在图像上绘制边界框的类，使用提供的检测结果。

    属性:
        color (Union[Color, ColorPalette]): 绘制边界框的颜色，
            可以是单一颜色或调色板
        thickness (int): 边界框线条的粗细，默认为2
        text_color (Color): 边界框上文本的颜色，默认为白色
        text_scale (float): 边界框上文本的缩放比例，默认为0.5
        text_thickness (int): 边界框上文本的粗细，
            默认为1
        text_padding (int): 边界框上文本周围的填充，
            默认为5

    """

    def __init__(
            self,
            color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
            thickness: int = 3,  # 1 用于 seeclick，2 用于 mind2web，3 用于演示
            text_color: Color = Color.BLACK,
            text_scale: float = 0.3,  # 移动端/网页端为0.8，桌面端为0.3，mind2web为0.4
            text_thickness: int = 1,  # 1，演示用2
            text_padding: int = 10,
            avoid_overlap: bool = True,
    ):
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        self.text_padding: int = text_padding
        self.avoid_overlap: bool = avoid_overlap

    def annotate(
            self,
            scene: np.ndarray,
            detections: Detections,
            labels: Optional[List[str]] = None,
            skip_label: bool = False,
            image_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        使用提供的检测结果在图像上绘制边界框。

        参数:
            scene (np.ndarray): 要绘制边界框的图像
            detections (Detections): 要绘制边界框的检测结果
            labels (Optional[List[str]]): 可选的标签列表，
                对应每个检测结果。如果未提供`labels`，
                将使用对应的`class_id`作为标签。
            skip_label (bool): 如果设置为`True`，则跳过边界框标签注释。
        返回:
            np.ndarray: 绘制了边界框的图像

        示例:
            ```python
            import supervision as sv

            classes = ['person', ...]  # 类别列表
            image = ...  # 图像
            detections = sv.Detections(...)  # 检测结果

            box_annotator = sv.BoxAnnotator()  # 创建边界框注释器
            labels = [
                f"{classes[class_id]} {confidence:0.2f}"
                for _, _, confidence, class_id, _ in detections
            ]  # 创建标签列表
            annotated_frame = box_annotator.annotate(
                scene=image.copy(),
                detections=detections,
                labels=labels
            )  # 绘制边界框和标签
            ```
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            class_id = (
                detections.class_id[i] if detections.class_id is not None else None
            )
            idx = class_id if class_id is not None else i
            color = (
                self.color.by_idx(idx)
                if isinstance(self.color, ColorPalette)
                else self.color
            )
            cv2.rectangle(
                img=scene,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=color.as_bgr(),
                thickness=self.thickness,
            )
            if skip_label:
                continue

            text = (
                f"{class_id}"
                if (labels is None or len(detections) != len(labels))
                else labels[i]
            )

            text_width, text_height = cv2.getTextSize(
                text=text,
                fontFace=font,
                fontScale=self.text_scale,
                thickness=self.text_thickness,
            )[0]

            if not self.avoid_overlap:
                text_x = x1 + self.text_padding
                text_y = y1 - self.text_padding

                text_background_x1 = x1
                text_background_y1 = y1 - 2 * self.text_padding - text_height

                text_background_x2 = x1 + 2 * self.text_padding + text_width
                text_background_y2 = y1
                # text_x = x1 - self.text_padding - text_width
                # text_y = y1 + self.text_padding + text_height
                # text_background_x1 = x1 - 2 * self.text_padding - text_width
                # text_background_y1 = y1
                # text_background_x2 = x1
                # text_background_y2 = y1 + 2 * self.text_padding + text_height
            else:
                text_x, text_y, text_background_x1, text_background_y1, text_background_x2, text_background_y2 = get_optimal_label_pos(
                    self.text_padding, text_width, text_height, x1, y1, x2, y2, detections, image_size)

            cv2.rectangle(
                img=scene,
                pt1=(text_background_x1, text_background_y1),
                pt2=(text_background_x2, text_background_y2),
                color=color.as_bgr(),
                thickness=cv2.FILLED,
            )
            # import pdb; pdb.set_trace()
            box_color = color.as_rgb()
            luminance = 0.299 * box_color[0] + 0.587 * box_color[1] + 0.114 * box_color[2]
            text_color = (0, 0, 0) if luminance > 160 else (255, 255, 255)
            cv2.putText(
                img=scene,
                text=text,
                org=(text_x, text_y),
                fontFace=font,
                fontScale=self.text_scale,
                # color=self.text_color.as_rgb(),
                color=text_color,
                thickness=self.text_thickness,
                lineType=cv2.LINE_AA,
            )
        return scene


def get_optimal_label_pos(text_padding, text_width, text_height, x1, y1, x2, y2, detections, image_size):
    """ 检查文本和背景检测框的重叠，并获取最佳标签位置，
        pos: str, 文本的位置，必须是'top left'、'top right'、'outer left'、'outer right'之一
        
        TODO: 如果全部重叠，返回最后一个，即outer right
        阈值: 默认为0.3
    """

    def get_is_overlap(detections, text_background_x1, text_background_y1, text_background_x2, text_background_y2,
                       image_size):
        is_overlap = False
        for i in range(len(detections)):
            detection = detections.xyxy[i].astype(int)
            if IoU([text_background_x1, text_background_y1, text_background_x2, text_background_y2], detection) > 0.3:
                is_overlap = True
                break
        # check if the text is out of the image
        if text_background_x1 < 0 or text_background_x2 > image_size[
            0] or text_background_y1 < 0 or text_background_y2 > image_size[1]:
            is_overlap = True
        return is_overlap

    # if pos == 'top left':
    text_x = x1 + text_padding
    text_y = y1 - text_padding

    text_background_x1 = x1
    text_background_y1 = y1 - 2 * text_padding - text_height

    text_background_x2 = x1 + 2 * text_padding + text_width
    text_background_y2 = y1
    is_overlap = get_is_overlap(detections, text_background_x1, text_background_y1, text_background_x2,
                                text_background_y2, image_size)
    if not is_overlap:
        return text_x, text_y, text_background_x1, text_background_y1, text_background_x2, text_background_y2

    # elif pos == 'outer left':
    text_x = x1 - text_padding - text_width
    text_y = y1 + text_padding + text_height

    text_background_x1 = x1 - 2 * text_padding - text_width
    text_background_y1 = y1

    text_background_x2 = x1
    text_background_y2 = y1 + 2 * text_padding + text_height
    is_overlap = get_is_overlap(detections, text_background_x1, text_background_y1, text_background_x2,
                                text_background_y2, image_size)
    if not is_overlap:
        return text_x, text_y, text_background_x1, text_background_y1, text_background_x2, text_background_y2

    # elif pos == 'outer right':
    text_x = x2 + text_padding
    text_y = y1 + text_padding + text_height

    text_background_x1 = x2
    text_background_y1 = y1

    text_background_x2 = x2 + 2 * text_padding + text_width
    text_background_y2 = y1 + 2 * text_padding + text_height

    is_overlap = get_is_overlap(detections, text_background_x1, text_background_y1, text_background_x2,
                                text_background_y2, image_size)
    if not is_overlap:
        return text_x, text_y, text_background_x1, text_background_y1, text_background_x2, text_background_y2

    # elif pos == 'top right':
    text_x = x2 - text_padding - text_width
    text_y = y1 - text_padding

    text_background_x1 = x2 - 2 * text_padding - text_width
    text_background_y1 = y1 - 2 * text_padding - text_height

    text_background_x2 = x2
    text_background_y2 = y1

    is_overlap = get_is_overlap(detections, text_background_x1, text_background_y1, text_background_x2,
                                text_background_y2, image_size)
    if not is_overlap:
        return text_x, text_y, text_background_x1, text_background_y1, text_background_x2, text_background_y2

    return text_x, text_y, text_background_x1, text_background_y1, text_background_x2, text_background_y2

def intersection_area(box1, box2):
    """计算两个边界框的交集面积"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return max(0, x2 - x1) * max(0, y2 - y1)

def box_area(box):
    """计算边界框的面积"""
    return (box[2] - box[0]) * (box[3] - box[1])

def IoU(box1, box2, return_max=True):
    """计算两个边界框的交并比(IoU)
    
    参数:
        box1: 第一个边界框
        box2: 第二个边界框
        return_max: 是否返回最大值(IoU和两个覆盖率的最大值)
    
    返回:
        交并比或最大覆盖率
    """
    intersection = intersection_area(box1, box2)
    union = box_area(box1) + box_area(box2) - intersection
    if box_area(box1) > 0 and box_area(box2) > 0:
        ratio1 = intersection / box_area(box1)  # box1被覆盖的比例
        ratio2 = intersection / box_area(box2)  # box2被覆盖的比例
    else:
        ratio1, ratio2 = 0, 0
    if return_max:
        return max(intersection / union, ratio1, ratio2)
    else:
        return intersection / union

def predict(model, image, caption, box_threshold, text_threshold):
    """ 使用huggingface模型替代原始模型进行预测
    """
    model, processor = model['model'], model['processor']
    device = model.device

    inputs = processor(images=image, text=caption, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    # 取消引入的包

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,  # 0.4,
        text_threshold=text_threshold,  # 0.3,
        target_sizes=[image.size[::-1]]
    )[0]
    boxes, logits, phrases = results["boxes"], results["scores"], results["labels"]
    return boxes, logits, phrases

def remove_overlap_new(boxes, iou_threshold, ocr_bbox=None):
    '''
    移除重叠的边界框
    
    ocr_bbox格式: [{'type': 'text', 'bbox':[x,y], 'interactivity':False, 'content':str }, ...]
    boxes格式: [{'type': 'icon', 'bbox':[x,y], 'interactivity':True, 'content':None }, ...]

    '''
    assert ocr_bbox is None or isinstance(ocr_bbox, List)

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection_area(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def IoU(box1, box2):
        intersection = intersection_area(box1, box2)
        union = box_area(box1) + box_area(box2) - intersection + 1e-6
        if box_area(box1) > 0 and box_area(box2) > 0:
            ratio1 = intersection / box_area(box1)
            ratio2 = intersection / box_area(box2)
        else:
            ratio1, ratio2 = 0, 0
        return max(intersection / union, ratio1, ratio2)

    def is_inside(box1, box2):
        # return box1[0] >= box2[0] and box1[1] >= box2[1] and box1[2] <= box2[2] and box1[3] <= box2[3]
        intersection = intersection_area(box1, box2)
        ratio1 = intersection / box_area(box1)
        return ratio1 > 0.80

    # boxes = boxes.tolist()
    filtered_boxes = []
    if ocr_bbox:
        filtered_boxes.extend(ocr_bbox)
    # print('ocr_bbox!!!', ocr_bbox)
    for i, box1_elem in enumerate(boxes):
        box1 = box1_elem['bbox']
        is_valid_box = True
        for j, box2_elem in enumerate(boxes):
            # keep the smaller box
            box2 = box2_elem['bbox']
            if i != j and IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2):
                is_valid_box = False
                break
        if is_valid_box:
            if ocr_bbox:
                # keep yolo boxes + prioritize ocr label
                box_added = False
                ocr_labels = ''
                for box3_elem in ocr_bbox:
                    if not box_added:
                        box3 = box3_elem['bbox']
                        if is_inside(box3, box1): # ocr inside icon
                            # box_added = True
                            # delete the box3_elem from ocr_bbox
                            try:
                                # gather all ocr labels
                                ocr_labels += box3_elem['content'] + ' '
                                filtered_boxes.remove(box3_elem)
                            except:
                                continue
                            # break
                        elif is_inside(box1, box3): # icon inside ocr, don't added this icon box, no need to check other ocr bbox bc no overlap between ocr bbox, icon can only be in one ocr box
                            box_added = True
                            break
                        else:
                            continue
                if not box_added:
                    if ocr_labels:
                        filtered_boxes.append({'type': 'icon', 'bbox': box1_elem['bbox'], 'interactivity': True, 'content': ocr_labels, 'source':'box_yolo_content_ocr'})
                    else:
                        filtered_boxes.append({'type': 'icon', 'bbox': box1_elem['bbox'], 'interactivity': True, 'content': None, 'source':'box_yolo_content_yolo'})
            else:
                filtered_boxes.append(box1)
    return filtered_boxes # torch.tensor(filtered_boxes)

def predict_yolo(model, image, box_threshold, imgsz, scale_img, iou_threshold=0.7):
    """ 使用huggingface模型替代原始模型进行YOLO预测
    """
    # model = model['model']
    if scale_img:
        result = model.predict(
        source=image,
        conf=box_threshold,
        imgsz=imgsz,
        iou=iou_threshold, # default 0.7
        )
    else:
        result = model.predict(
        source=image,
        conf=box_threshold,
        iou=iou_threshold, # default 0.7
        )
    del model
    boxes = result[0].boxes.xyxy#.tolist() # in pixel space
    conf = result[0].boxes.conf
    phrases = [str(i) for i in range(len(boxes))]

    return boxes, conf, phrases
def int_box_area(box, w, h):
    """计算边界框的实际面积"""
    x1, y1, x2, y2 = box
    int_box = [int(x1*w), int(y1*h), int(x2*w), int(y2*h)]
    area = (int_box[2] - int_box[0]) * (int_box[3] - int_box[1])
    return area


@torch.inference_mode()
def get_parsed_content_icon(filtered_boxes, starting_idx, image_source, caption_model_processor, prompt=None,
                            batch_size=32):
    # 每批样本数量，对于florence v2模型，128个样本大约需要4GB GPU内存
    to_pil = ToPILImage()
    if starting_idx:
        non_ocr_boxes = filtered_boxes[starting_idx:]
    else:
        non_ocr_boxes = filtered_boxes
    croped_pil_image = []
    for i, coord in enumerate(non_ocr_boxes):
        try:
            xmin, xmax = int(coord[0] * image_source.shape[1]), int(coord[2] * image_source.shape[1])
            ymin, ymax = int(coord[1] * image_source.shape[0]), int(coord[3] * image_source.shape[0])
            cropped_image = image_source[ymin:ymax, xmin:xmax, :]
            cropped_image = cv2.resize(cropped_image, (64, 64))
            croped_pil_image.append(to_pil(cropped_image))
        except:
            continue

    model, processor = caption_model_processor['model'], caption_model_processor['processor']
    if not prompt:
        if 'florence' in model.config.name_or_path:
            prompt = "<CAPTION>"
        else:
            prompt = "The image shows"

    generated_texts = []
    device = model.device
    for i in range(0, len(croped_pil_image), batch_size):
        start = time.time()
        batch = croped_pil_image[i:i + batch_size]
        t1 = time.time()
        if model.device.type == 'cuda':
            inputs = processor(images=batch, text=[prompt] * len(batch), return_tensors="pt", do_resize=False).to(
                device=device, dtype=torch.float16)
        else:
            inputs = processor(images=batch, text=[prompt] * len(batch), return_tensors="pt").to(device=device)
        if 'florence' in model.config.name_or_path:
            generated_ids = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                                           max_new_tokens=20, num_beams=1, do_sample=False)
        else:
            generated_ids = model.generate(**inputs, max_length=100, num_beams=5, no_repeat_ngram_size=2,
                                           early_stopping=True,
                                           num_return_sequences=1)  # temperature=0.01, do_sample=True,
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_text = [gen.strip() for gen in generated_text]
        generated_texts.extend(generated_text)

    return generated_texts


def get_parsed_content_icon_phi3v(filtered_boxes, ocr_bbox, image_source, caption_model_processor):
    """使用phi3v模型解析图标内容"""
    to_pil = ToPILImage()
    if ocr_bbox:
        non_ocr_boxes = filtered_boxes[len(ocr_bbox):]
    else:
        non_ocr_boxes = filtered_boxes
    croped_pil_image = []
    for i, coord in enumerate(non_ocr_boxes):
        xmin, xmax = int(coord[0] * image_source.shape[1]), int(coord[2] * image_source.shape[1])
        ymin, ymax = int(coord[1] * image_source.shape[0]), int(coord[3] * image_source.shape[0])
        cropped_image = image_source[ymin:ymax, xmin:xmax, :]
        croped_pil_image.append(to_pil(cropped_image))

    model, processor = caption_model_processor['model'], caption_model_processor['processor']
    device = model.device
    messages = [{"role": "user", "content": "<|image_1|>\ndescribe the icon in one sentence"}]
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    batch_size = 5  # Number of samples per batch
    generated_texts = []

    for i in range(0, len(croped_pil_image), batch_size):
        images = croped_pil_image[i:i + batch_size]
        image_inputs = [processor.image_processor(x, return_tensors="pt") for x in images]
        inputs = {'input_ids': [], 'attention_mask': [], 'pixel_values': [], 'image_sizes': []}
        texts = [prompt] * len(images)
        for i, txt in enumerate(texts):
            input = processor._convert_images_texts_to_inputs(image_inputs[i], txt, return_tensors="pt")
            inputs['input_ids'].append(input['input_ids'])
            inputs['attention_mask'].append(input['attention_mask'])
            inputs['pixel_values'].append(input['pixel_values'])
            inputs['image_sizes'].append(input['image_sizes'])
        max_len = max([x.shape[1] for x in inputs['input_ids']])
        for i, v in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = torch.cat(
                [processor.tokenizer.pad_token_id * torch.ones(1, max_len - v.shape[1], dtype=torch.long), v], dim=1)
            inputs['attention_mask'][i] = torch.cat(
                [torch.zeros(1, max_len - v.shape[1], dtype=torch.long), inputs['attention_mask'][i]], dim=1)
        inputs_cat = {k: torch.concatenate(v).to(device) for k, v in inputs.items()}

        generation_args = {
            "max_new_tokens": 25,
            "temperature": 0.01,
            "do_sample": False,
        }
        generate_ids = model.generate(**inputs_cat, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)
        # # remove input tokens
        generate_ids = generate_ids[:, inputs_cat['input_ids'].shape[1]:]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        response = [res.strip('\n').strip() for res in response]
        generated_texts.extend(response)

    return generated_texts
def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str], text_scale: float,
             text_padding=5, text_thickness=2, thickness=3) -> np.ndarray:
    """
    该函数在图像上标注边界框和标签。

    参数:
    image_source (np.ndarray): 要标注的源图像。
    boxes (torch.Tensor): 包含边界框坐标的张量，采用cxcywh格式，像素尺度
    logits (torch.Tensor): 包含每个边界框置信度分数的张量。
    phrases (List[str]): 每个边界框的标签列表。
    text_scale (float): 要显示的文本比例，移动端/网页端为0.8，桌面端为0.3，mind2web为0.4

    返回:
    np.ndarray: 标注后的图像。
    """
    h, w, _ = image_source.shape

    from torchvision.ops import box_convert
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    xywh = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xywh").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [f"{phrase}" for phrase in range(boxes.shape[0])]

    box_annotator = BoxAnnotator(text_scale=text_scale, text_padding=text_padding,text_thickness=text_thickness,thickness=thickness) # 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web
    annotated_frame = image_source.copy()
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels, image_size=(w,h))

    label_coordinates = {f"{phrase}": v for phrase, v in zip(phrases, xywh)}
    return annotated_frame, label_coordinates



def get_som_labeled_img(image_source: Union[str, Image.Image], model=None, BOX_THRESHOLD=0.01,
                        output_coord_in_ratio=False, ocr_bbox=None, text_scale=0.4, text_padding=5,
                        draw_bbox_config=None, caption_model_processor=None, ocr_text=[], use_local_semantics=True,
                        iou_threshold=0.9, prompt=None, scale_img=False, imgsz=None, batch_size=128):
    """处理图像路径或Image对象

    参数:
        image_source: 图像路径(str)或PIL Image对象
        ...
    """
    if not isinstance(image_source, Image.Image):
        image_source = Image.open(image_source)
    image_source = image_source.convert("RGB")  # for CLIP
    w, h = image_source.size
    if not imgsz:
        imgsz = (h, w)
    # print('image size:', w, h)
    xyxy, logits, phrases = predict_yolo(model=model, image=image_source, box_threshold=BOX_THRESHOLD, imgsz=imgsz,
                                         scale_img=scale_img, iou_threshold=0.1)
    xyxy = xyxy / torch.Tensor([w, h, w, h]).to(xyxy.device)
    image_source = np.asarray(image_source)
    phrases = [str(i) for i in range(len(phrases))]

    # annotate the image with labels
    if ocr_bbox and ocr_text:
        ocr_bbox = torch.tensor(ocr_bbox) / torch.Tensor([w, h, w, h])
        ocr_bbox = ocr_bbox.tolist()
        ocr_bbox_elem = \
            [{'type': 'text', 'bbox': box, 'interactivity': False, 'content': txt, 'source': 'box_ocr_content_ocr'} for
             box, txt in zip(ocr_bbox, ocr_text) if int_box_area(box, w, h) > 0]
    else:
        print('no ocr bbox!!!')
        ocr_bbox_elem = []
    xyxy_elem = [{'type': 'icon', 'bbox': box, 'interactivity': True, 'content': None} for box in xyxy.tolist() if
                 int_box_area(box, w, h) > 0]
    filtered_boxes_dict = remove_overlap_new(boxes=xyxy_elem, iou_threshold=iou_threshold, ocr_bbox=ocr_bbox_elem)

    # sort the filtered_boxes_dict so that the one with 'content': None is at the end, and get the index of the first 'content': None
    filtered_boxes_elem = sorted(filtered_boxes_dict, key=lambda x: x.get('content') is None)
    # get the index of the first 'content': None
    starting_idx = next((i for i, box in enumerate(filtered_boxes_elem) if box.get('content') is None), -1)
    filtered_boxes = torch.tensor([box.get('bbox', [0, 0, 0, 0]) for box in filtered_boxes_elem])
    print('len(filtered_boxes):', len(filtered_boxes), starting_idx)

    # get parsed icon local semantics
    time1 = time.time()
    if use_local_semantics:
        caption_model = caption_model_processor['model']
        if 'phi3_v' in caption_model.config.model_type:
            parsed_content_icon = get_parsed_content_icon_phi3v(filtered_boxes, ocr_bbox, image_source,
                                                                caption_model_processor)
        else:
            parsed_content_icon = get_parsed_content_icon(filtered_boxes, starting_idx, image_source,
                                                          caption_model_processor, prompt=prompt, batch_size=batch_size)
        ocr_text = [f"Text Box ID {i}: {txt}" for i, txt in enumerate(ocr_text)]
        icon_start = len(ocr_text)
        parsed_content_icon_ls = []
        # fill the filtered_boxes_elem None content with parsed_content_icon in order
        for i, box in enumerate(filtered_boxes_elem):
            if box.get('content') is None:
                box['content'] = parsed_content_icon.pop(0)
        for i, txt in enumerate(parsed_content_icon):
            parsed_content_icon_ls.append(f"Icon Box ID {str(i + icon_start)}: {txt}")
        parsed_content_merged = ocr_text + parsed_content_icon_ls
    else:
        ocr_text = [f"Text Box ID {i}: {txt}" for i, txt in enumerate(ocr_text)]
        parsed_content_merged = ocr_text
    print('time to get parsed content:', time.time() - time1)

    filtered_boxes = box_convert(boxes=filtered_boxes, in_fmt="xyxy", out_fmt="cxcywh")

    phrases = [i for i in range(len(filtered_boxes))]

    # draw boxes
    if draw_bbox_config:
        annotated_frame, label_coordinates = annotate(image_source=image_source, boxes=filtered_boxes, logits=logits,
                                                      phrases=phrases, **draw_bbox_config)
    else:
        annotated_frame, label_coordinates = annotate(image_source=image_source, boxes=filtered_boxes, logits=logits,
                                                      phrases=phrases, text_scale=text_scale, text_padding=text_padding)

    pil_img = Image.fromarray(annotated_frame)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode('ascii')
    if output_coord_in_ratio:
        label_coordinates = {k: [v[0] / w, v[1] / h, v[2] / w, v[3] / h] for k, v in label_coordinates.items()}
        assert w == annotated_frame.shape[1] and h == annotated_frame.shape[0]

    return encoded_image, label_coordinates, filtered_boxes_elem


def get_xywh_yolo(input):
    """从YOLO格式的输入中获取xywh坐标"""
    x, y, w, h = input[0], input[1], input[2] - input[0], input[3] - input[1]
    x, y, w, h = int(x), int(y), int(w), int(h)
    return x, y, w, h


def get_xyxy(input):
    """从输入中获取xyxy坐标"""
    x, y, xp, yp = input[0][0], input[0][1], input[2][0], input[2][1]
    x, y, xp, yp = int(x), int(y), int(xp), int(yp)
    return x, y, xp, yp


def get_xywh(input):
    """从输入中获取xywh坐标"""
    x, y, w, h = input[0][0], input[0][1], input[2][0] - input[0][0], input[2][1] - input[0][1]
    x, y, w, h = int(x), int(y), int(w), int(h)
    return x, y, w, h


def api_based_ocr(image_source: Union[str, Image.Image], display_img=True, output_bb_format='xywh', goal_filtering=None,
                  api_args=None):
    """使用API服务进行OCR识别
    
    参数:
        image_source: 图像路径(str)或PIL Image对象
        display_img: 是否显示图像
        output_bb_format: 输出边界框格式，可选'xywh'或'xyxy'
        goal_filtering: 目标过滤
        api_args: API调用参数，包含API密钥、URL等
    """
    # 处理图像源
    if not isinstance(image_source, Image.Image):
        image_source = Image.open(image_source)
    if image_source.mode == 'RGBA':
        image_source = image_source.convert('RGB')
    image_np = np.array(image_source)
    w, h = image_source.size
    
    # 准备API调用参数
    if api_args is None:
        api_args = {}
    
    # 将图像转换为base64编码，以便通过API发送
    buffered = io.BytesIO()
    image_source.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('ascii')
    
    # 调用OCR API
    api_url = api_args.get('url', 'http://127.0.0.1:802/ocr')
    
    payload = {
        "file": image_base64,
        "fileType": 1
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status()  # 如果请求失败，抛出异常
        
        # 解析API响应
        result = response.json()
        # with open('api_response.json', 'w') as f:
        #     json.dump(result, f)
        #     print('api_response.json saved')
        result_data = result["result"]["ocrResults"][0]["prunedResult"]
        print(result_data)
        # 提取文本和边界框
        text = []
        bb = []
        # 检查是否有rec_texts和rec_boxes字段（基于实际API响应格式）
        if "rec_texts" not in result_data or "rec_boxes" not in result_data:
            print("Warning: API response missing 'rec_texts' or 'rec_boxes' field")
            return ([], []), goal_filtering
            
        rec_texts = result_data["rec_texts"]
        rec_boxes = result_data["rec_boxes"]
        
        if not rec_texts or not rec_boxes:
            print("no ocr bbox!!!")
            return ([], []), goal_filtering
        
        # 确保文本和边界框数量一致
        min_len = min(len(rec_texts), len(rec_boxes))
        for i in range(min_len):
            text_content = rec_texts[i]
            bbox = rec_boxes[i]
            
            if text_content and bbox and len(bbox) >= 4:
                text.append(text_content)
                
                # 根据output_bb_format格式化边界框
                if output_bb_format == 'xywh':
                    # 假设bbox是[x1, y1, x2, y2]格式，转换为xywh
                    x1, y1, x2, y2 = bbox[:4]
                    width = x2 - x1
                    height = y2 - y1
                    bb.append([int(x1), int(y1), int(width), int(height)])
                else:  # xyxy
                    bb.append([int(x) for x in bbox[:4]])
        
    except requests.exceptions.RequestException as e:
        print(f"OCR API request failed: {e}")
        return ([], []), goal_filtering
    except json.JSONDecodeError as e:
        print(f"Failed to parse OCR API response: {e}")
        return ([], []), goal_filtering
    except Exception as e:
        print(f"Unexpected error in OCR: {e}")
        return ([], []), goal_filtering
    
    # 如果需要显示图像
    if display_img and bb:
        opencv_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        for box in bb:
            if output_bb_format == 'xywh':
                x, y, w, h = box
                cv2.rectangle(opencv_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:  # xyxy
                x, y, x2, y2 = box
                cv2.rectangle(opencv_img, (x, y), (x2, y2), (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB))
    
    print(f'OCR found {len(text)} text boxes')
    return (text, bb), goal_filtering

def check_ocr_box(image_source: Union[str, Image.Image], display_img=True, output_bb_format='xywh', goal_filtering=None,
                  easyocr_args=None, use_paddleocr=False):
    """检查OCR边界框
    
    参数:
        image_source: 图像路径(str)或PIL Image对象
        display_img: 是否显示图像
        output_bb_format: 输出边界框格式，可选'xywh'或'xyxy'
        goal_filtering: 目标过滤
        easyocr_args: EasyOCR参数
        use_paddleocr: 是否使用PaddleOCR
    """
    if not isinstance(image_source, Image.Image):
        image_source = Image.open(image_source)
    if image_source.mode == 'RGBA':
        # Convert RGBA to RGB to avoid alpha channel issues
        image_source = image_source.convert('RGB')
    image_np = np.array(image_source)
    w, h = image_source.size
    # TODO 增加对图片进行复杂度判断，自动分割 分割后每一块单独去识别 最后再聚合到一起 避免一个页面 ico太多

    if use_paddleocr:
        if easyocr_args is None:
            text_threshold = 0.5
        else:
            text_threshold = easyocr_args['text_threshold']

        result = paddle_ocr.ocr(image_np, cls=False)[0]
        coord = [item[0] for item in result if item[1][1] > text_threshold]
        text = [item[1][0] for item in result if item[1][1] > text_threshold]
    else:  # EasyOCR
        if easyocr_args is None:
            easyocr_args = {}
        import easyocr
        reader = easyocr.Reader(['en'])
        result = reader.readtext(image_np, **easyocr_args)
        coord = [item[0] for item in result]
        text = [item[1] for item in result]
    if display_img:
        opencv_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        bb = []
        for item in coord:
            x, y, a, b = get_xywh(item)
            bb.append((x, y, a, b))
            cv2.rectangle(opencv_img, (x, y), (x + a, y + b), (0, 255, 0), 2)
        #  matplotlib expects RGB
        plt.imshow(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB))
    else:
        if output_bb_format == 'xywh':
            bb = [get_xywh(item) for item in coord]
        elif output_bb_format == 'xyxy':
            bb = [get_xyxy(item) for item in coord]
    return (text, bb), goal_filtering
