from paddleocr import PaddleOCR
import cv2
import numpy as np

# Initialize once (very important)
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en'
)


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Denoise
    blur = cv2.GaussianBlur(gray, (3,3), 0)

    # Convert back to 3 channel (VERY IMPORTANT)
    processed = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)

    return processed



def extract_text_boxes(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return []

    processed = preprocess(image)

    results = ocr.ocr(processed)

    boxes = []

    if results:
        for line in results[0]:
            bbox = line[0]
            text = line[1][0]
            confidence = line[1][1]

            if confidence > 0.6:
                x = int(bbox[0][0])
                y = int(bbox[0][1])
                width = int(bbox[2][0] - bbox[0][0])
                height = int(bbox[2][1] - bbox[0][1])

                boxes.append({
                    "text": text,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height
                })

    return boxes
