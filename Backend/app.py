from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os

from ocr.reader import extract_text_boxes

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"status": "SmartDoc AI backend running"}


# ✅ CLEAN UPLOAD ENDPOINT (FILE ONLY)
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


# ✅ CLEAN OCR ENDPOINT (FILENAME ONLY)
@app.post("/ocr")
async def run_ocr(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(file_path):
        return {"error": "File not found"}

    boxes = extract_text_boxes(file_path)

    return {
        "filename": filename,
        "total_boxes": len(boxes),
        "data": boxes
    }
