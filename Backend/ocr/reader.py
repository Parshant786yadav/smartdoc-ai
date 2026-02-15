import easyocr
import cv2

reader = easyocr.Reader(['en'])

def extract_text_boxes(image_path):

    image = cv2.imread(image_path)

    results = reader.readtext(image)

    boxes = []

    for (bbox, text, confidence) in results:

        if confidence > 0.3:   # adjust if needed
            (top_left, top_right, bottom_right, bottom_left) = bbox

            x = int(top_left[0])
            y = int(top_left[1])
            width = int(top_right[0] - top_left[0])
            height = int(bottom_left[1] - top_left[1])

            boxes.append({
                "text": text,
                "x": x,
                "y": y,
                "width": width,
                "height": height
            })

    return boxes
