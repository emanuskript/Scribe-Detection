# feature_extractor.py
"""
Line embeddings for scribe detection (fixed-dim):
- Central-band crop (focus on x-height)
- LBP (texture) -> length = P+2 (default 10)
- HOG (orientation histogram aggregated over blocks) -> length = orientations (default 9)
- Optional color stats on ink (Hue circular mean/var, Value mean/std) -> length = 5

Final embedding is [w_lbp*LBP || w_hog*HOG || w_color*COLOR] then global L2-normalized.
"""

from __future__ import annotations
import numpy as np
import cv2
from typing import Tuple, Optional
from skimage.feature import local_binary_pattern, hog

# -------------------------------
# Basic helpers
# -------------------------------

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def _resize_keep_aspect(img: np.ndarray, target_h: int = 128) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = float(target_h) / max(h, 1)
    new_w = max(int(round(w * scale)), 1)
    return cv2.resize(img, (new_w, target_h),
                      interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)

def _l1_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    s = float(np.sum(np.abs(v))) + eps
    return (v / s).astype(np.float32)

def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v) + eps)
    return (v / n).astype(np.float32)

def _moving_sum(y: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return y.copy()
    pad = win // 2
    yp = np.pad(y, (pad, pad), mode="reflect")
    k = np.ones(win, dtype=np.float64)
    return np.convolve(yp, k, mode="valid")

# -------------------------------
# Central-band cropping
# -------------------------------

def central_band_coords(
    gray_line: np.ndarray,
    frac: float = 0.6,
    pad: int = 2,
    method: str = "maxsum",
    min_band_px: int = 12
) -> Tuple[int, int]:
    g = _to_gray(gray_line)
    H, _ = g.shape[:2]
    if H == 0:
        return 0, 0
    if frac is None or not (0.0 < frac <= 1.01):
        return 0, H

    band = int(round(max(min_band_px, frac * H)))
    band = min(band, H)

    if band >= H or method != "maxsum":
        center = H // 2
        top = max(0, center - band // 2)
        bot = min(H, top + band)
    else:
        ink = (255.0 - g.astype(np.float64))
        row_profile = ink.mean(axis=1)
        s = _moving_sum(row_profile, max(1, band))
        top = int(np.argmax(s))
        top = max(0, min(top, H - band))
        bot = top + band

    top = max(0, top - pad)
    bot = min(H, bot + pad)
    return top, bot

def central_band_crop(img: np.ndarray, frac: float = 0.6, pad: int = 2,
                      method: str = "maxsum", min_band_px: int = 12) -> np.ndarray:
    g = _to_gray(img)
    top, bot = central_band_coords(g, frac=frac, pad=pad, method=method, min_band_px=min_band_px)
    return g[top:bot, :]

# -------------------------------
# LBP (fixed length = P+2)
# -------------------------------

def lbp_hist(img: np.ndarray, P: int = 8, R: int = 1, method: str = "uniform") -> np.ndarray:
    g = _to_gray(img)
    lbp = local_binary_pattern(g, P=P, R=R, method=method)
    bins = P + 2  # 'uniform' pattern count
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins), density=False)
    return _l1_normalize(hist.astype(np.float32))

# -------------------------------
# HOG (fixed length = orientations)
# -------------------------------

def _hog_orient_hist(
    g: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (16, 16),
    cells_per_block: Tuple[int, int] = (2, 2),
    transform_sqrt: bool = True,
    block_norm: str = "L2-Hys",
) -> np.ndarray:
    """
    Compute HOG with feature_vector=False and aggregate across all blocks/cells,
    returning a single orientation histogram of length = orientations.
    This makes the feature dimension independent of image width/height.
    """
    H, W = g.shape[:2]
    if H == 0 or W == 0:
        return np.zeros((orientations,), dtype=np.float32)

    ppc_options = [(16, 16), (12, 12), (8, 8)]
    for ppc in ppc_options:
        try:
            feat = hog(
                g,
                orientations=orientations,
                pixels_per_cell=ppc,
                cells_per_block=cells_per_block,
                transform_sqrt=transform_sqrt,
                block_norm=block_norm,
                feature_vector=False,  # crucial
            )
            # skimage returns shape (n_blocks_row, n_blocks_col, cb_r, cb_c, orientations)
            if feat.ndim == 5 and feat.shape[-1] == orientations:
                orient_hist = feat.sum(axis=(0, 1, 2, 3))
            elif feat.ndim == 3 and feat.shape[-1] == orientations:
                orient_hist = feat.sum(axis=(0, 1))
            else:
                # Fallback: flatten then fold into orientations bins
                flat = feat.ravel().astype(np.float32)
                k = flat.size // orientations
                if k > 0:
                    orient_hist = flat[:k*orientations].reshape(-1, orientations).sum(axis=0)
                else:
                    orient_hist = np.zeros((orientations,), dtype=np.float32)
            # L2 normalize the orientation histogram
            return _l2_normalize(orient_hist.astype(np.float32))
        except Exception:
            continue

    # Last resort: zero vector
    return np.zeros((orientations,), dtype=np.float32)

def hog_hist(
    img: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (16, 16),
    cells_per_block: Tuple[int, int] = (2, 2),
    transform_sqrt: bool = True,
    block_norm: str = "L2-Hys",
) -> np.ndarray:
    g = _to_gray(img)
    return _hog_orient_hist(
        g,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        transform_sqrt=transform_sqrt,
        block_norm=block_norm,
    )

# -------------------------------
# Color features on ink (length = 5)
# -------------------------------

def _ink_mask_from_gray(g: np.ndarray) -> np.ndarray:
    if g.size == 0:
        return np.zeros_like(g, dtype=bool)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (bw == 0)
    ink = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))).astype(bool)
    return ink

def color_stats_hv(color_band: np.ndarray, gray_band: np.ndarray) -> np.ndarray:
    H, W = gray_band.shape[:2]
    if gray_band.size == 0:
        return np.zeros((5,), dtype=np.float32)
    if color_band is None or color_band.ndim != 3 or color_band.shape[2] != 3:
        v = (gray_band.astype(np.float32) / 255.0).ravel()
        v_mean = float(v.mean()) if v.size else 0.0
        v_std = float(v.std()) if v.size else 0.0
        return np.array([0.0, 0.0, 0.0, v_mean, v_std], dtype=np.float32)

    hsv = cv2.cvtColor(color_band, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].astype(np.float32) * (2.0 * np.pi / 180.0)
    v = hsv[..., 2].astype(np.float32) / 255.0

    ink = _ink_mask_from_gray(gray_band)
    if np.count_nonzero(ink) < 20:
        ink = np.ones_like(ink, dtype=bool)

    h_sel = h[ink]; v_sel = v[ink]
    c_mean = float(np.cos(h_sel).mean()) if h_sel.size else 0.0
    s_mean = float(np.sin(h_sel).mean()) if h_sel.size else 0.0
    R = float(np.hypot(c_mean, s_mean))
    h_var = 1.0 - R
    v_mean = float(v_sel.mean()) if v_sel.size else 0.0
    v_std = float(v_sel.std()) if v_sel.size else 0.0
    return np.array([c_mean, s_mean, h_var, v_mean, v_std], dtype=np.float32)

# -------------------------------
# Line embedding (fixed-dim)
# -------------------------------

def line_embedding(
    line_img: np.ndarray,
    central_band_frac: Optional[float] = 0.6,
    central_band_pad: int = 2,
    resize_height: int = 128,
    use_color: bool = True,
    w_lbp: float = 1.0,
    w_hog: float = 1.0,
    w_color: float = 0.5,
    lbp_P: int = 8,
    lbp_R: int = 1,
    hog_orientations: int = 9,
    hog_pixels_per_cell: Tuple[int, int] = (16, 16),
    hog_cells_per_block: Tuple[int, int] = (2, 2),
) -> np.ndarray:
    # Central band (same for gray & color)
    gray_full = _to_gray(line_img)
    H, W = gray_full.shape[:2]
    if central_band_frac is not None and 0.0 < central_band_frac <= 1.01 and H > 0 and W > 0:
        top, bot = central_band_coords(gray_full, frac=central_band_frac,
                                       pad=central_band_pad, method="maxsum", min_band_px=12)
    else:
        top, bot = 0, H

    gray_band = gray_full[top:bot, :]
    color_band = line_img[top:bot, :] if (line_img.ndim == 3 and line_img.shape[2] == 3) else None

    # Normalize scale
    g = _resize_keep_aspect(gray_band, target_h=resize_height)

    # Fixed-dim features
    lbp_vec = lbp_hist(g, P=lbp_P, R=lbp_R)  # length = P+2
    hog_vec = hog_hist(
        g,
        orientations=hog_orientations,
        pixels_per_cell=hog_pixels_per_cell,
        cells_per_block=hog_cells_per_block,
    )  # length = orientations

    parts = []
    if w_lbp != 0.0: parts.append((lbp_vec, float(w_lbp)))
    if w_hog != 0.0: parts.append((hog_vec, float(w_hog)))

    if use_color:
        color_feat = color_stats_hv(color_band if color_band is not None else gray_band, gray_band)
        color_feat = _l2_normalize(color_feat)
        if w_color != 0.0:
            parts.append((color_feat, float(w_color)))

    if not parts:
        return np.zeros((1,), dtype=np.float32)

    vecs = [v * w for (v, w) in parts]
    emb = np.concatenate(vecs).astype(np.float32)
    return _l2_normalize(emb)

__all__ = [
    "line_embedding",
    "lbp_hist",
    "hog_hist",
    "central_band_crop",
    "central_band_coords",
    "color_stats_hv",
]
