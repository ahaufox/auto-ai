import os,sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)



from utils.util import api_based_ocr

image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "imgs", "input", "test1.jpg")
(text, ocr_bbox), _ = api_based_ocr(image_path, display_img=False, output_bb_format='xyxy', api_args={'text_threshold': 0.8})


print('text:',text,'\n\n','ocr_bbox:', ocr_bbox)


