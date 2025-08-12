
from paddleocr import PaddleOCR
ocr = PaddleOCR(lang='chinese_cht') # need to run only once to load model into memory
img_path = 'PaddleOCR/doc/imgs_words_en/word_10.png'
result = ocr.ocr(img_path, det=False, cls=False)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)