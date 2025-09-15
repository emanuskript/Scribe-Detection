# pre_processor.py
"""
Preprocessing for manuscript pages:
1) Grayscale
2) Illumination/background correction
3) Small-angle deskew (±5°)
4) Sauvola adaptive binarization

Primary entry point: preprocess(bgr_img) -> bw uint8 (0 or 255)
Optional: preprocess_debug(bgr_img) -> dict of intermediate stages
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import Dict, Tuple

try:
    from skimage.filters import threshold_sauvola
    _HAS_SAUVOLA = True
except Exception:
    _HAS_SAUVOLA = False


# -------------------------------
# Utilities
# -------------------------------

def _ensure_odd(n: int) -> int:
    return int(n + (n % 2 == 0))

def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert BGR/RGB to single-channel grayscale (uint8)."""
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def _auto_kernel_size(shape: Tuple[int, int], frac: float = 0.035, min_ks: int = 21, max_ks: int = 151) -> int:
    """
    Choose an odd kernel size as a fraction of the smaller image dimension.
    Good defaults for background estimation on manuscript scans.
    """
    h, w = shape[:2]
    base = int(min(h, w) * frac)
    base = np.clip(base, min_ks, max_ks)
    return _ensure_odd(base)

# -------------------------------
# Illumination / background correction
# -------------------------------

def illumination_correct(gray: np.ndarray, method: str = "morph_open", frac: float = 0.035) -> np.ndarray:
    """
    Remove slow-varying background (parchment shading) and normalize contrast.
    method:
      - "morph_open": morphological opening to estimate background (default; robust)
      - "gauss": large Gaussian blur as background estimate (faster fallback)
    """
    gray = gray if gray.dtype == np.uint8 else cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ks = _auto_kernel_size(gray.shape, frac=frac)

    if method == "gauss":
        bg = cv2.GaussianBlur(gray, (ks, ks), 0)
    else:
        # morphological opening with an elliptical kernel approximates background
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        bg = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    corrected = cv2.subtract(gray, bg)
    # stretch to full range
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
    return corrected

# -------------------------------
# Small-angle deskew
# -------------------------------

def _estimate_small_skew_angle(gray: np.ndarray, max_angle: float = 5.0) -> float:
    """
    Estimate small skew (±max_angle degrees) using Hough on Canny edges.
    Works best if illumination-corrected. Returns 0.0 if not enough evidence.
    """
    # scale down for speed & stability in Hough
    h, w = gray.shape[:2]
    scale = 1200.0 / max(h, w) if max(h, w) > 1200 else 1.0
    small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    edges = cv2.Canny(small, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 1800, max(200, int(0.15 * edges.size / 1000)))
    if lines is None:
        return 0.0

    angles = []
    for rho_theta in lines[:300]:
        for _, theta in rho_theta:
            # Convert to degrees; text lines are ~ horizontal (theta ~ 0 or ~pi)
            angle = (theta * 180.0 / np.pi) - 90.0
            # keep only near-horizontal angles
            if -max_angle <= angle <= max_angle:
                angles.append(angle)

    if not angles:
        return 0.0
    return float(np.median(angles))

def deskew_small(gray: np.ndarray, max_angle: float = 5.0, border_value: int = 255) -> Tuple[np.ndarray, float]:
    """
    Rotate image by a small estimated angle; returns (rotated, angle_degrees).
    """
    angle = _estimate_small_skew_angle(gray, max_angle=max_angle)
    if abs(angle) < 0.05:  # negligible
        return gray, 0.0
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    rot = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=border_value)
    return rot, angle

# -------------------------------
# Binarization (Sauvola with fallback)
# -------------------------------

def binarize_sauvola(gray: np.ndarray, window: int = 31, k: float = 0.2) -> np.ndarray:
    """
    Sauvola adaptive threshold; returns binary uint8 image {0,255}.
    If skimage is missing, falls back to OpenCV adaptive mean thresholding.
    """
    if _HAS_SAUVOLA:
        # skimage expects float or uint; we'll pass uint8; it computes fine.
        t = threshold_sauvola(gray, window_size=_ensure_odd(window), k=k)
        bw = (gray > t).astype(np.uint8) * 255
        return bw
    else:
        # Fallback: not identical to Sauvola, but reasonable
        win = _ensure_odd(window)
        bw = cv2.adaptiveThreshold(
            gray,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=win,
            C=10
        )
        return bw

# -------------------------------
# Main entry points
# -------------------------------

def preprocess(bgr_img: np.ndarray,
               illum_method: str = "morph_open",
               illum_frac: float = 0.035,
               do_deskew: bool = True,
               sauvola_window: int = 31,
               sauvola_k: float = 0.2) -> np.ndarray:
    """
    Full pipeline → returns a binary image (uint8 with values {0, 255}).
    Parameters are tuned for manuscript pages; adjust if needed.
    """
    gray = to_gray(bgr_img)
    corrected = illumination_correct(gray, method=illum_method, frac=illum_frac)
    if do_deskew:
        corrected, _ = deskew_small(corrected, max_angle=5.0, border_value=255)
    bw = binarize_sauvola(corrected, window=sauvola_window, k=sauvola_k)
    return bw

def preprocess_debug(bgr_img: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Debug version returning intermediate stages for QA/visualization.
    Keys: 'gray', 'illum', 'deskew', 'bw'. Also returns 'angle' (float).
    """
    out: Dict[str, np.ndarray] = {}
    gray = to_gray(bgr_img)
    out["gray"] = gray

    illum = illumination_correct(gray, method="morph_open", frac=0.035)
    out["illum"] = illum

    deskewed, angle = deskew_small(illum, max_angle=5.0, border_value=255)
    out["deskew"] = deskewed
    out["angle"] = np.array([angle], dtype=np.float32)  # store angle as small array for convenience

    bw = binarize_sauvola(deskewed, window=31, k=0.2)
    out["bw"] = bw
    return out


# Optional explicit exports
__all__ = [
    "preprocess",
    "preprocess_debug",
    "to_gray",
    "illumination_correct",
    "deskew_small",
    "binarize_sauvola",
]
