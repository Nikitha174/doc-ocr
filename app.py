"""
app.py
FastAPI entrypoint for the document ingestion pipeline.

Flow:
  Upload (PDF/Image) -> pdf2image -> OpenCV preprocessing ->
  PaddleOCR (text/tables) + heuristic stamp/signature detection ->
  Tesseract validation -> structured JSON response (also saved to /output)
"""

import os
import json
import shutil
import uuid
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from utils.pdf_utils import file_to_images
from utils.preprocess import preprocess_image
from utils.extractor import process_page

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Document Ingestion & OCR Service", version="1.0.0")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/extract")
async def extract_document(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # 1. Save upload
    doc_id = str(uuid.uuid4())
    saved_path = os.path.join(UPLOAD_DIR, f"{doc_id}{ext}")
    with open(saved_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # 2. PDF/Image -> list of page images
        page_images = file_to_images(saved_path)

        # 3. Preprocess + 4. OCR pipeline per page
        pages_result = []
        for i, raw_image in enumerate(page_images, start=1):
            clean_image = preprocess_image(raw_image)
            page_result = process_page(clean_image, page_number=i)
            pages_result.append(page_result)

        # 5. Structured JSON response
        response = {
            "document_id": doc_id,
            "filename": file.filename,
            "processed_at": datetime.utcnow().isoformat() + "Z",
            "num_pages": len(pages_result),
            "pages": pages_result,
        }

        # Persist result to /output for later retrieval
        output_path = os.path.join(OUTPUT_DIR, f"{doc_id}.json")
        with open(output_path, "w") as f:
            json.dump(response, f, indent=2)

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/result/{document_id}")
def get_result(document_id: str):
    output_path = os.path.join(OUTPUT_DIR, f"{document_id}.json")
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Result not found")
    with open(output_path, "r") as f:
        return JSONResponse(content=json.load(f))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
