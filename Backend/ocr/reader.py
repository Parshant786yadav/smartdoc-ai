import pytesseract
import cv2

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_boxes(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    data = pytesseract.image_to_data(
        gray,
        output_type=pytesseract.Output.DICT,
        config="--psm 6"
    )

    results = []

    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if text:
            results.append({
                "text": text,
                "x": data["left"][i],
                "y": data["top"][i],
                "width": data["width"][i],
                "height": data["height"][i]
            })

    return results
