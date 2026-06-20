"""
extractor.py
Layout-aware OCR pipeline:
  - PaddleOCR (PP-OCR)      -> text + bounding boxes
  - PaddleOCR PPStructure   -> table detection/extraction
  - OpenCV heuristics       -> stamp detection (color blobs) & signature
                               detection (irregular ink regions)
  - Tesseract               -> cross-validation of PaddleOCR text
"""

import cv2
import numpy as np
import pytesseract
from typing import List, Dict, Any

from paddleocr import PaddleOCR, PPStructure

# Initialize once (heavy models) — reused across requests
_paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
_table_engine = PPStructure(show_log=False, layout=True, table=True, ocr=True)


# ---------------------------------------------------------------------
# TEXT EXTRACTION
# ---------------------------------------------------------------------
def extract_text(image: np.ndarray) -> List[Dict[str, Any]]:
    """Runs PaddleOCR and returns text blocks with boxes + confidence."""
    result = _paddle_ocr.ocr(image, cls=True)
    blocks = []
    if not result or result[0] is None:
        return blocks

    for line in result[0]:
        box, (text, confidence) = line
        blocks.append({
            "text": text,
            "confidence": round(float(confidence), 4),
            "box": box,  # 4 corner points
        })
    return blocks


# ---------------------------------------------------------------------
# TABLE DETECTION
# ---------------------------------------------------------------------
def extract_tables(image: np.ndarray) -> List[Dict[str, Any]]:
    """Uses PaddleOCR's PPStructure to detect tables and return their HTML + bbox."""
    result = _table_engine(image)
    tables = []
    for region in result:
        if region.get("type") == "table":
            tables.append({
                "bbox": region.get("bbox"),
                "html": region.get("res", {}).get("html", ""),
            })
    return tables


# ---------------------------------------------------------------------
# STAMP DETECTION (heuristic: colored, roughly circular/rectangular blobs)
# ---------------------------------------------------------------------
def detect_stamps(image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Stamps are usually red/blue/purple ink, distinct from black printed text.
    We isolate strongly colored (non-grayscale) regions and find blob contours.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red ink range (wraps around hue 0)
    red_mask1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    # Blue/purple ink range (common for official stamps)
    blue_mask = cv2.inRange(hsv, (100, 70, 50), (140, 255, 255))

    mask = cv2.bitwise_or(cv2.bitwise_or(red_mask1, red_mask2), blue_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    stamps = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 800:  # filter noise
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        if 0.5 <= aspect_ratio <= 2.0:  # stamps are roughly round/square
            stamps.append({"bbox": [int(x), int(y), int(w), int(h)], "area": int(area)})

    return stamps


# ---------------------------------------------------------------------
# SIGNATURE DETECTION (heuristic: dense, irregular, non-linear ink strokes)
# ---------------------------------------------------------------------
def detect_signatures(image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Signatures are usually black/dark, freeform cursive strokes with high
    contour irregularity, found in lower regions of forms. This is a
    lightweight heuristic, not a trained classifier.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5, 15), np.uint8))

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    signatures = []
    h_img, w_img = gray.shape[:2]

    for c in contours:
        area = cv2.contourArea(c)
        if area < 1500:
            continue
        x, y, w, h = cv2.boundingRect(c)

        # Heuristics: wide-ish, short-ish blob, irregular boundary (low solidity)
        aspect_ratio = w / float(h)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        solidity = area / hull_area if hull_area > 0 else 0

        is_signature_shape = 1.5 <= aspect_ratio <= 8.0 and solidity < 0.55
        in_lower_half = y > (h_img * 0.4)  # signatures usually near bottom of forms

        if is_signature_shape and in_lower_half:
            signatures.append({"bbox": [int(x), int(y), int(w), int(h)], "solidity": round(solidity, 3)})

    return signatures


# ---------------------------------------------------------------------
# TESSERACT VALIDATION
# ---------------------------------------------------------------------
def validate_with_tesseract(image: np.ndarray, paddle_text_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Runs Tesseract on the full image and compares overall extracted text
    against PaddleOCR's output as a simple cross-validation/confidence check.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tesseract_text = pytesseract.image_to_string(gray).strip()

    paddle_text = " ".join(b["text"] for b in paddle_text_blocks).strip()

    paddle_words = set(paddle_text.lower().split())
    tesseract_words = set(tesseract_text.lower().split())

    if not paddle_words:
        overlap_ratio = 0.0
    else:
        overlap_ratio = len(paddle_words & tesseract_words) / len(paddle_words)

    return {
        "tesseract_text": tesseract_text,
        "word_overlap_ratio": round(overlap_ratio, 3),
        "validated": overlap_ratio >= 0.5,  # simple agreement threshold
    }


# ---------------------------------------------------------------------
# FULL PIPELINE FOR A SINGLE PAGE IMAGE
# ---------------------------------------------------------------------
def process_page(image: np.ndarray, page_number: int) -> Dict[str, Any]:
    text_blocks = extract_text(image)
    tables = extract_tables(image)
    stamps = detect_stamps(image)
    signatures = detect_signatures(image)
    validation = validate_with_tesseract(image, text_blocks)

    return {
        "page": page_number,
        "text_blocks": text_blocks,
        "tables": tables,
        "stamps": stamps,
        "signatures": signatures,
        "tesseract_validation": validation,
    }
