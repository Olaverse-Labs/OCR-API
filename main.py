from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
import pytesseract
import io
import requests
from langdetect import detect_langs
import time
import cv2
import numpy as np

app = FastAPI()

def deskew_image(pil_img):
    # Convert to OpenCV format
    img = np.array(pil_img.convert('L'))
    coords = np.column_stack(np.where(img > 0))
    angle = 0
    if coords.size > 0:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(img)

@app.post("/extract-text/")
async def extract_text(
    file: UploadFile = File(None),
    image_url: str = Form(None),
    preprocess: str = Form(None, description="Preprocessing: grayscale, threshold, denoise, or comma-separated list"),
    ocr_lang: str = Form('eng', description="Tesseract language(s), e.g. 'eng', 'fra', 'eng+fra'"),
    deskew: bool = Form(False, description="Deskew image before OCR")
):
    start_time = time.time()
    # Load image from file or URL
    if file:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
    elif image_url:
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content))
    else:
        return JSONResponse(content={"status": False, "error": "No image provided."}, status_code=400)

    # Deskew if requested
    if deskew:
        image = deskew_image(image)

    # Preprocessing
    if preprocess:
        img_cv = np.array(image)
        if 'grayscale' in preprocess:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        if 'threshold' in preprocess:
            if len(img_cv.shape) == 3:
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
            _, img_cv = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if 'denoise' in preprocess:
            img_cv = cv2.fastNlMeansDenoising(img_cv, None, 30, 7, 21)
        image = Image.fromarray(img_cv)

    # OCR with bounding boxes, confidence, and structure
    custom_config = f'-l {ocr_lang} --oem 3 --psm 3'
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=custom_config)
    n_boxes = len(data['level'])
    width, height = image.size

    # Group words into lines, paragraphs, blocks
    blocks = {}
    for i in range(n_boxes):
        block_num = data['block_num'][i]
        par_num = data['par_num'][i]
        line_num = data['line_num'][i]
        word_text = data['text'][i].strip()
        conf = float(data['conf'][i]) if data['conf'][i] != '-1' else None
        # Block
        if block_num not in blocks:
            blocks[block_num] = {'box': [], 'paragraphs': {}, 'text': '', 'boxes': []}
        block = blocks[block_num]
        # Paragraph
        if par_num not in block['paragraphs']:
            block['paragraphs'][par_num] = {'box': [], 'lines': {}, 'text': '', 'boxes': []}
        paragraph = block['paragraphs'][par_num]
        # Line
        if line_num not in paragraph['lines']:
            paragraph['lines'][line_num] = {'box': [], 'words': [], 'text': '', 'boxes': []}
        line = paragraph['lines'][line_num]
        if word_text:
            box = [
                data['left'][i] / width,
                data['top'][i] / height,
                data['width'][i] / width,
                data['height'][i] / height
            ]
            word = {
                'text': word_text,
                'boxCoordinates': box,
                'confidence': conf
            }
            line['words'].append(word)
            line['text'] += (word_text + ' ')
            line['boxes'].append([
                data['left'][i],
                data['top'][i],
                data['left'][i] + data['width'][i],
                data['top'][i] + data['height'][i]
            ])
        # For line/paragraph/block boxes
        if word_text:
            paragraph['boxes'].append([
                data['left'][i],
                data['top'][i],
                data['left'][i] + data['width'][i],
                data['top'][i] + data['height'][i]
            ])
            block['boxes'].append([
                data['left'][i],
                data['top'][i],
                data['left'][i] + data['width'][i],
                data['top'][i] + data['height'][i]
            ])
    # Format output
    output_blocks = []
    all_boxes = []
    for block_num, block in blocks.items():
        block_paragraphs = []
        # Block box
        if block['boxes']:
            lefts = [b[0] for b in block['boxes']]
            tops = [b[1] for b in block['boxes']]
            rights = [b[2] for b in block['boxes']]
            bottoms = [b[3] for b in block['boxes']]
            block_box = [
                min(lefts) / width,
                min(tops) / height,
                (max(rights) - min(lefts)) / width,
                (max(bottoms) - min(tops)) / height
            ]
            all_boxes.append([
                min(lefts), min(tops), max(rights), max(bottoms)
            ])
        else:
            block_box = [0, 0, 0, 0]
        for par_num, paragraph in block['paragraphs'].items():
            paragraph_lines = []
            # Paragraph box
            if paragraph['boxes']:
                lefts = [b[0] for b in paragraph['boxes']]
                tops = [b[1] for b in paragraph['boxes']]
                rights = [b[2] for b in paragraph['boxes']]
                bottoms = [b[3] for b in paragraph['boxes']]
                par_box = [
                    min(lefts) / width,
                    min(tops) / height,
                    (max(rights) - min(lefts)) / width,
                    (max(bottoms) - min(tops)) / height
                ]
            else:
                par_box = [0, 0, 0, 0]
            for line_num, line in paragraph['lines'].items():
                # Line box
                if line['boxes']:
                    lefts = [b[0] for b in line['boxes']]
                    tops = [b[1] for b in line['boxes']]
                    rights = [b[2] for b in line['boxes']]
                    bottoms = [b[3] for b in line['boxes']]
                    line_box = [
                        min(lefts) / width,
                        min(tops) / height,
                        (max(rights) - min(lefts)) / width,
                        (max(bottoms) - min(tops)) / height
                    ]
                else:
                    line_box = [0, 0, 0, 0]
                paragraph_lines.append({
                    'text': line['text'].strip(),
                    'words': line['words'],
                    'boxCoordinates': line_box
                })
            block_paragraphs.append({
                'lines': paragraph_lines,
                'boxCoordinates': par_box
            })
        output_blocks.append({
            'paragraphs': block_paragraphs,
            'boxCoordinates': block_box
        })
    # Find the largest box (by area) for the whole text
    if all_boxes:
        lefts = [b[0] for b in all_boxes]
        tops = [b[1] for b in all_boxes]
        rights = [b[2] for b in all_boxes]
        bottoms = [b[3] for b in all_boxes]
        main_box = [
            min(lefts) / width,
            min(tops) / height,
            (max(rights) - min(lefts)) / width,
            (max(bottoms) - min(tops)) / height
        ]
    else:
        main_box = [0, 0, 0, 0]
    # Full text for language detection
    full_text = ' '.join([line['text'] for block in output_blocks for par in block['paragraphs'] for line in par['lines']]).strip()
    detected_languages = []
    if full_text:
        try:
            langs = detect_langs(full_text)
            for lang in langs:
                detected_languages.append({
                    'languageCode': lang.lang,
                    'confidence': lang.prob
                })
        except Exception:
            pass
    execution_time = int((time.time() - start_time) * 1000)
    return JSONResponse(content={
        "status": True,
        "text": full_text,
        "boxCoordinates": main_box,
        "blocks": output_blocks,
        "detectedLanguages": detected_languages,
        "executionTimeMS": execution_time
    }) 