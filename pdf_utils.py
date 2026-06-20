"""
pdf_utils.py
Converts uploaded PDFs into a list of images (one per page).
If the uploaded file is already an image, it's just loaded as-is.
"""

import os
from typing import List
import numpy as np
from pdf2image import convert_from_path
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}


def file_to_images(file_path: str, dpi: int = 300) -> List[np.ndarray]:
    """
    Converts a PDF or image file into a list of OpenCV-compatible (numpy) images.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        pages = convert_from_path(file_path, dpi=dpi)
        return [np.array(page.convert("RGB"))[:, :, ::-1] for page in pages]  # RGB -> BGR

    elif ext in IMAGE_EXTENSIONS:
        img = Image.open(file_path).convert("RGB")
        return [np.array(img)[:, :, ::-1]]  # RGB -> BGR

    else:
        raise ValueError(f"Unsupported file type: {ext}")
