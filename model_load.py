import torch
import os
from PIL import Image
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['HF_DATASETS_OFFLINE']='1'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['INPUT_IMG_DIR']='./imgs/input/'
os.environ['OUTPUT_IMG_DIR']='./imgs/out/'
def get_caption_model_processor(model_name, model_name_or_path="Salesforce/blip2-opt-2.7b", device=None):
    """
    获取模型 Salesforce/blip2-opt-2.7b 或 microsoft/Florence-2-base 给检测到的图标生成自然语言描述

    
    model_name: 模型名称
    model_name_or_path: 模型路径
    device: 设备
    return: 模型和处理器
    """
    
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name == "blip2":
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        if device == 'cpu':
            model = Blip2ForConditionalGeneration.from_pretrained(
            model_name_or_path, device_map=None, torch_dtype=torch.float32
        )
        else:
            model = Blip2ForConditionalGeneration.from_pretrained(
            model_name_or_path, device_map=None, torch_dtype=torch.float16
        ).to(device)
    elif model_name == "florence2":
        from transformers import AutoProcessor, AutoModelForCausalLM
        processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
        if device == 'cpu':
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float32, trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    return {'model': model.to(device), 'processor': processor}

def get_yolo_model(model_path):
    from ultralytics import YOLO
    print( 'Load YOLO the model.')
    model = YOLO(model_path)
    return model

