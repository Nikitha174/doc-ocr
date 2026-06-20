"""
preprocess.py
Basic image cleanup before OCR: grayscale, denoise, deskew, and
thresholding to improve text/table/stamp recognition accuracy.
"""

import cv2
import numpy as np


def upscale_if_small(image: np.ndarray, min_width: int = 1500) -> np.ndarray:
    """
    Upscales an image if its width is below `min_width`.
    Low-resolution scans/photos (e.g. small phone images, thumbnails)
    don't carry enough pixel detail for OCR to resolve characters —
    this brings them up to a usable size before the rest of the
    pipeline runs. Uses INTER_CUBIC for smoother enlargement.
    """
    h, w = image.shape[:2]
    if w >= min_width:
        return image

    scale = min_width / float(w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def deskew(image: np.ndarray) -> np.ndarray:
    """Corrects slight rotation/skew in a scanned document."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] == 0:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Real scanner skew is usually a few degrees. Large "corrections" are
    # almost always a false read (e.g. on clean, already-straight digital
    # PDFs) and would do more harm than good — skip those.
    if abs(angle) < 0.3 or abs(angle) > 10:
        return image

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Cleans an image for OCR:
    1. Upscale if resolution is too low
    2. Deskew (only correcting genuine skew, see clamp above)
    3. Light color-preserving denoise
    4. Contrast enhancement on luminance only (keeps color intact —
       required for stamp detection downstream)
    Returns a BGR image.
    """
    image = upscale_if_small(image)
    image = deskew(image)

    # Gentle denoise that preserves color and avoids blurring fine gaps
    # between text lines (aggressive denoising can merge adjacent lines
    # and confuse the OCR layout/detection model).
    denoised = cv2.fastNlMeansDenoisingColored(
        image, None, h=5, hColor=5, templateWindowSize=7, searchWindowSize=21
    )

    # Contrast-enhance only the lightness channel (LAB color space) so
    # stamp/seal colors survive for detect_stamps() later in the pipeline.
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    return result
