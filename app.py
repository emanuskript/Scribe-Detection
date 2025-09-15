# app.py
from __future__ import annotations
import os, re, json, uuid, time, shutil, subprocess, logging
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
from flask import Flask, request, render_template, abort
from werkzeug.utils import secure_filename

from similarity import ImageProcessor  # updated class with scores + reasons

# ------------------ Config ------------------
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = BASE_DIR / "uploads"
JOBS_DIR = STATIC_DIR / "jobs"

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20MB

KRAKEN_BIN = os.getenv("KRAKEN_BIN", "kraken")
KRAKEN_SEG_FLAGS = ["segment", "-bl"]  # same as before

# heuristics for cropping lines
MIN_LINE_WIDTH = 150
MAX_LINE_HEIGHT = 350

JOB_TTL_SEC = 6 * 60 * 60  # cleanup old jobs after 6h

# ------------------ App ------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

for d in (STATIC_DIR, UPLOADS_DIR, JOBS_DIR):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("scribe-app")

# ------------------ Utils ------------------
def allowed_file(name: str) -> bool:
    return Path(name).suffix.lower() in ALLOWED_EXT

def new_job_id() -> str:
    return uuid.uuid4().hex[:12]

def job_paths(job_id: str) -> Dict[str, Path]:
    root = JOBS_DIR / job_id
    lines_dir = root / "lines"
    root.mkdir(parents=True, exist_ok=True)
    lines_dir.mkdir(parents=True, exist_ok=True)
    return {"root": root, "lines": lines_dir}

def cleanup_old_jobs(ttl: int = JOB_TTL_SEC):
    now = time.time()
    for p in JOBS_DIR.glob("*"):
        try:
            if p.is_dir() and (now - p.stat().st_mtime) > ttl:
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass

def clamp_bbox(x0, y0, x1, y1, W, H):
    x0 = max(0, min(int(x0), W - 1)); x1 = max(0, min(int(x1), W))
    y0 = max(0, min(int(y0), H - 1)); y1 = max(0, min(int(y1), H))
    if x1 <= x0: x1 = min(W, x0 + 1)
    if y1 <= y0: y1 = min(H, y0 + 1)
    return x0, y0, x1, y1

# ------------------ Kraken segmentation ------------------
def run_kraken_segment(img_path: Path, out_json: Path) -> List[Dict]:
    cmd = [KRAKEN_BIN, "-i", str(img_path), str(out_json)] + KRAKEN_SEG_FLAGS
    log.info("Running: %s", " ".join(cmd))
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    if res.stdout: log.info("kraken stdout: %s", res.stdout.strip()[:300])
    if res.stderr: log.info("kraken stderr: %s", res.stderr.strip()[:300])
    with out_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    lines = data.get("lines", [])
    return lines if isinstance(lines, list) else []

def crop_lines(img_path: Path, lines: List[Dict], out_dir: Path) -> Tuple[List[str], List[List[Tuple[int,int]]]]:
    """Save line crops and also return their polygon boundaries (for animation)."""
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None: raise RuntimeError("Failed to read uploaded image")
    H, W = img.shape[:2]

    rel_paths: List[str] = []
    polys: List[List[Tuple[int,int]]] = []

    idx = 1
    for ln in lines:
        boundary = ln.get("boundary")
        if not boundary or not isinstance(boundary, list) or len(boundary) < 2: continue
        xs = [pt[0] for pt in boundary]; ys = [pt[1] for pt in boundary]
        x0, y0, x1, y1 = clamp_bbox(min(xs), min(ys), max(xs), max(ys), W, H)
        crop = img[y0:y1, x0:x1]
        if crop.size == 0: continue

        h, w = crop.shape[:2]
        if w < MIN_LINE_WIDTH: continue
        if h > MAX_LINE_HEIGHT: continue

        out_name = f"line_{idx}.jpg"
        cv2.imwrite(str(out_dir / out_name), crop)
        rel_paths.append(str((out_dir / out_name).relative_to(STATIC_DIR)).replace("\\", "/"))

        # normalize polygon points to ints and clamp
        poly = [(int(max(0, min(px, W-1))), int(max(0, min(py, H-1)))) for px, py in boundary]
        polys.append(poly)
        idx += 1

    return rel_paths, polys

# ------------------ Routes ------------------
@app.errorhandler(413)
def too_large(_e):
    return "File too large (max 20 MB).", 413

@app.route("/", methods=["GET", "POST"])
def index():
    cleanup_old_jobs()

    if request.method == "GET":
        return render_template("index.html")

    # POST: upload image → segment → crop → detect changes → render results
    file = request.files.get("image")
    if not file or not file.filename:
        return render_template("index.html", error="No file selected.")
    if not allowed_file(file.filename):
        return render_template("index.html", error="Unsupported file type.")

    job_id = new_job_id()
    paths = job_paths(job_id)

    # save upload (also copy to job root so front-end can display it)
    fname = secure_filename(file.filename)
    ext = Path(fname).suffix.lower()
    up_path = UPLOADS_DIR / f"{job_id}{ext}"
    file.save(str(up_path))
    page_copy = paths["root"] / f"page{ext}"
    shutil.copy(str(up_path), str(page_copy))
    page_rel = str(page_copy.relative_to(STATIC_DIR)).replace("\\", "/")

    # run kraken + crop
    try:
        seg_json = paths["root"] / "segmentation.json"
        kr_lines = run_kraken_segment(up_path, seg_json)
        line_rel_paths, polygons = crop_lines(up_path, kr_lines, paths["lines"])
    except subprocess.CalledProcessError as e:
        return render_template("index.html", error=f"Segmentation failed. {e}")
    except Exception as e:
        return render_template("index.html", error=f"Error: {e}")

    # detect scribe changes + build reasons
    processor = ImageProcessor()
    line_abs_paths = [str(STATIC_DIR / rp) for rp in line_rel_paths]
    result = processor.detect_with_reasons(line_abs_paths)

    # Build cards for UI: [(left_rel, right_rel, score_pct, reason), ...]
    cards = []
    for ch in result["changes"]:
        i = ch["index"]
        if i < 0 or i >= len(line_rel_paths) - 1: continue
        left_rel = line_rel_paths[i]
        right_rel = line_rel_paths[i + 1]
        score_pct = int(round(ch["confidence"] * 100))
        reason = ch["reason"]
        cards.append({"left": left_rel, "right": right_rel, "score": score_pct, "reason": reason})

    # Pass polygons for canvas animation
    # (We draw them client-side; simple sequential animation)
    return render_template(
        "results.html",
        page_image=page_rel,
        polygons=json.dumps(polygons),
        cards=cards,
        job_id=job_id
    )

if __name__ == "__main__":
    # Change default port if 5000 is busy: PORT=5050 python3 app.py
    port = int(os.getenv("PORT", "5050"))
    app.run(host="0.0.0.0", port=port, debug=True)
