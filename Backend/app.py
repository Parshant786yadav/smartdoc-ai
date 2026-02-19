from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uuid
import os
import cv2

from ocr.reader import (
    extract_text_boxes,
    extract_text_style,
    detect_text_in_box,
    remove_text,
    replace_text,
)

app = FastAPI()

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Upload Folder
# =========================
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Serve uploaded images
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


# =========================
# Root
# =========================
@app.get("/")
def root():
    return {"status": "SmartDoc AI backend running"}


# =========================
# Upload Endpoint
# =========================
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    ext = file.filename.split(".")[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    return {
        "message": "File uploaded successfully",
        "filename": filename
    }


# =========================
# OCR Endpoint
# =========================
@app.post("/ocr")
async def run_ocr(filename: str = "", mode: str = "words"):

    if not filename:
        return {"error": "Filename is missing"}

    file_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(file_path):
        return {"error": "File not found"}

    if mode not in ("words", "lines"):
        mode = "words"

    result = extract_text_boxes(file_path, mode=mode)

    return {
        "filename": filename,
        "mode": mode,
        "total_boxes": len(result["boxes"]),
        "boxes": result["boxes"],
        "width": result["width"],
        "height": result["height"]
    }


# =========================
# Detect text in selected box (before replace)
# =========================
class DetectInBoxRequest(BaseModel):
    filename: str
    box: dict


@app.post("/detect-in-box")
async def detect_in_box(data: DetectInBoxRequest):
    """Detect the complete text in the given box so user can confirm before replacing."""
    file_path = os.path.join(UPLOAD_DIR, data.filename)
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    image = cv2.imread(file_path)
    if image is None:
        return {"error": "Could not load image"}
    detected_text = detect_text_in_box(image, data.box)
    return {"detected_text": detected_text}


# =========================
# Replace Model
# =========================
class ReplaceRequest(BaseModel):
    filename: str
    box: dict
    new_text: str


# =========================
# Replace Endpoint
# =========================
@app.post("/replace")
async def replace_word(data: ReplaceRequest):

    file_path = os.path.join(UPLOAD_DIR, data.filename)

    if not os.path.exists(file_path):
        return {"error": "File not found"}

    image = cv2.imread(file_path)

    if image is None:
        return {"error": "Could not load image"}

    # Extract style from actual text pixels (height, weight, color)
    style = extract_text_style(image, data.box)

    # Remove old text in the box
    image = remove_text(image, data.box)

    # Replace with new text in same style/design/weight/height
    image = replace_text(image, data.box, data.new_text, style)

    # Save updated image
    cv2.imwrite(file_path, image)

    return {
        "message": "Text replaced successfully",
        "image_url": f"http://127.0.0.1:8000/uploads/{data.filename}"
    }
