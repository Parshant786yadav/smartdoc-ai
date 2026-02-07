from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import uuid
import os

app = FastAPI()

# Allow frontend later
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


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith(("image/", "application/pdf")):
        return {"error": "Only image or PDF files are allowed"}

    # Generate unique filename
    ext = file.filename.split(".")[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    # Save file
    with open(file_path, "wb") as f:
        f.write(await file.read())

    return {
        "message": "File uploaded successfully",
        "filename": filename,
        "content_type": file.content_type
    }
