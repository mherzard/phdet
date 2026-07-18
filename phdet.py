#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phdet.py
Պարզ Python ծրագիր՝ լուսանկարում գույնը pH գունապնակի հետ համեմատելու և
ամենամոտիկ թվային արժեքը (pH) գտնելու համար։

Օգտագործում.
    python phdet.py պալիտրա.png թեստ.jpg --point 120 180 --size 20
    python phdet.py պալիտրա.png թեստ.jpg --interactive
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    from skimage.color import deltaE_cie76, rgb2lab

    SKIMAGE = True
except Exception:
    SKIMAGE = False


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def load_image(path):
    """Load image as RGB numpy array (H, W, 3) float32."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32)


def _weighted_kmeans1d(points, weights, k, max_iter=100, tol=0.01):
    """1D weighted K-Means. Returns sorted cluster centers."""
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(points) == 0:
        return np.linspace(0, k - 1, k)

    # initialize evenly across the data range
    lo, hi = points.min(), points.max()
    centers = np.linspace(lo + (hi - lo) / (2 * k), hi - (hi - lo) / (2 * k), k)

    for _ in range(max_iter):
        dists = np.abs(points[:, None] - centers[None, :])
        labels = np.argmin(dists, axis=1)
        new_centers = np.empty_like(centers)
        moved = False
        for i in range(k):
            idx = labels == i
            if idx.any():
                new_centers[i] = np.average(points[idx], weights=weights[idx])
            else:
                # empty cluster: move to a point far from other centers
                other = np.delete(centers, i)
                if other.size:
                    far_idx = np.argmax(np.min(np.abs(points[:, None] - other[None, :]), axis=1))
                    new_centers[i] = points[far_idx]
                else:
                    new_centers[i] = centers[i]
            if abs(new_centers[i] - centers[i]) > tol:
                moved = True
        centers = new_centers
        if not moved:
            break

    return np.sort(centers)


def _color_mask(hsv):
    """Return a mask of non-gray/non-white/non-black-ish pixels."""
    s = hsv[:, :, 1].astype(float)
    v = hsv[:, :, 2].astype(float)
    return (s > 5.0) & (v > 30.0)


# ---------------------------------------------------------------------------
# Palette extraction
# ---------------------------------------------------------------------------

def extract_palette(image, n_colors=15):
    """
    Extract a palette of `n_colors` colors from a horizontal pH scale image.

    Returns:
        palette   : (n_colors, 3) float32 mean RGB values
        positions : list of (center_x, center_y) in the original image
        band      : (y1, y2) row range of the detected color band
    """
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    hsv = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2HSV)
    mask = _color_mask(hsv)

    # 1) Detect horizontal color band
    row_score = mask.mean(axis=1)
    colored_rows = np.where(row_score > 0.20)[0]
    if len(colored_rows) == 0:
        h = arr.shape[0]
        y1, y2 = int(h * 0.35), int(h * 0.65)
    else:
        y1, y2 = int(colored_rows.min()), int(colored_rows.max())

    band = arr[y1:y2]
    band_hsv = cv2.cvtColor(band.astype(np.uint8), cv2.COLOR_RGB2HSV)
    band_mask = _color_mask(band_hsv)

    # 2) Find vertical bar positions using a 1D weighted K-Means on column color scores.
    #    Color score = standard deviation across RGB channels (quick colorfulness).
    col_std = np.std(band, axis=(0, 2))
    try:
        from skimage.filters import threshold_otsu

        thr = threshold_otsu(col_std)
    except Exception:
        thr = np.percentile(col_std, 60)

    cand = np.where(col_std > thr)[0]
    if len(cand) < n_colors:
        # fallback: use every column, weighted by colorfulness
        cand = np.arange(band.shape[1])
        weights = col_std + 1e-6
    else:
        weights = col_std[cand]

    centers = _weighted_kmeans1d(cand, weights, n_colors)

    # 3) Compute average color per bar using pixels inside each bar segment.
    #    Boundaries are midpoints between neighboring centers.
    bounds = [0]
    for i in range(n_colors - 1):
        bounds.append(int((centers[i] + centers[i + 1]) / 2))
    bounds.append(band.shape[1])

    palette = []
    positions = []
    for i in range(n_colors):
        x1 = max(0, bounds[i])
        x2 = min(band.shape[1], bounds[i + 1])
        patch = band[:, x1:x2]
        patch_mask = band_mask[:, x1:x2]
        if patch_mask.sum() == 0:
            patch_mask = np.ones((patch.shape[0], patch.shape[1]), dtype=bool)
        mean_rgb = patch[patch_mask].mean(axis=0)
        palette.append(mean_rgb)
        positions.append((int(centers[i]), (y1 + y2) // 2))

    return np.array(palette, dtype=np.float32), positions, (y1, y2)


# ---------------------------------------------------------------------------
# Sample color handling
# ---------------------------------------------------------------------------

def pick_sample_color(image, point=None, crop=None, size=20, white_balance=False):
    """
    Return the average RGB color of the sample region.

    point : (x, y) center of the square region
    crop  : (x, y, w, h) rectangle
    size  : half side of the square if only point is given
    """
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]

    if crop is not None:
        x, y, cw, ch = crop
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + cw)
        y2 = min(h, y + ch)
    else:
        if point is None:
            x, y = w // 2, h // 2
        else:
            x, y = point
        x1 = max(0, x - size)
        x2 = min(w, x + size)
        y1 = max(0, y - size)
        y2 = min(h, y + size)

    patch = arr[y1:y2, x1:x2]
    if patch.size == 0:
        raise ValueError("Ընտրված տարածքը դատարկ է։ Ստուգեք կոորդինատները։")

    if white_balance:
        patch = gray_world_white_balance(patch)

    return patch.mean(axis=(0, 1)), (x1, y1, x2 - x1, y2 - y1)


def gray_world_white_balance(patch):
    """Simple gray-world white balance on a patch."""
    means = patch.mean(axis=(0, 1))
    avg = means.mean()
    scale = avg / (means + 1e-6)
    return np.clip(patch * scale, 0, 255)


# ---------------------------------------------------------------------------
# Color matching
# ---------------------------------------------------------------------------

def rgb_to_hsv360(rgb):
    """Return H in degrees [0,360), S and V in [0,1]."""
    hsv = cv2.cvtColor((np.asarray(rgb).reshape(1, 1, 3) / 255.0).astype(np.float32), cv2.COLOR_RGB2HSV)
    h, s, v = hsv[0, 0]
    # OpenCV H is 0..179 for 8-bit images and 0..359 for float images when using
    # COLOR_RGB2HSV; make sure we always have 0..360 degrees.
    h_deg = (float(h) % 180.0) * 2.0 if float(h) >= 180.0 else float(h) * 2.0
    # simpler robust normalization:
    h_deg = ((float(h) * 2.0) % 360.0 + 360.0) % 360.0
    return h_deg, float(s), float(v)


def hue_distance(h1, h2):
    """Circular distance between two hues in degrees, returning 0..180."""
    h1 = float(h1) % 360.0
    h2 = float(h2) % 360.0
    d = abs(h1 - h2)
    return float(min(d, 360.0 - d))


def chroma(s, v):
    """Chroma proxy = saturation * value (0..1)."""
    return float(s) * float(v)


def match_color(
    sample_rgb,
    palette,
    use_interp=False,
    method="combined",
    lab_weight=0.75,
    hsv_weight=0.25,
):
    """
    Compare sample RGB to palette and return the nearest pH value(s).

    method options:
        "lab"      : only CIE Lab ΔE76
        "hsv"      : only HSV hue+saturation distance (chroma-aware)
        "combined" : Lab ΔE plus a hue-penalty derived from HSV
        "rgb"      : simple Euclidean RGB distance (fallback if skimage missing)

    Returns dict with:
        ph          : nearest integer pH (0..n-1)
        distance    : color distance to nearest palette color
        distances   : distances to all palette colors
        ph_interp   : optional weighted pH value
        per_method  : dict of results from each method
    """
    sample_rgb = np.asarray(sample_rgb, dtype=float)
    palette = np.asarray(palette, dtype=float)
    n = len(palette)

    per_method = {}

    # 1) Lab ΔE76 (perceptually uniform color distance)
    if SKIMAGE:
        lab_pal = rgb2lab(palette.reshape(1, n, 3) / 255.0)
        lab_smp = rgb2lab(sample_rgb.reshape(1, 1, 3) / 255.0)
        lab_dists = deltaE_cie76(lab_smp, lab_pal).reshape(n)
    else:
        lab_dists = np.linalg.norm(palette - sample_rgb, axis=1)
    per_method["lab"] = _score_from_dists(lab_dists)

    # 2) HSV hue+saturation+value distance, reliability-scaled by chroma
    sample_h, sample_s, sample_v = rgb_to_hsv360(sample_rgb)
    sample_chroma = chroma(sample_s, sample_v)
    pal_hsv = np.array([rgb_to_hsv360(p) for p in palette])
    pal_h, pal_s, pal_v = pal_hsv[:, 0], pal_hsv[:, 1], pal_hsv[:, 2]
    pal_chromas = pal_s * pal_v

    hue_d = np.array([hue_distance(sample_h, h) for h in pal_h])
    sat_d = np.abs(sample_s - pal_s) * 180.0
    val_d = np.abs(sample_v - pal_v) * 180.0
    # chroma reliability: 0 for grayish colors, 1 for vivid colors
    chroma_reliability = (sample_chroma * pal_chromas) / (0.05 + sample_chroma + pal_chromas)
    # when chroma is low, reduce hue influence; keep sat/value terms
    hsv_dists = hue_d * chroma_reliability + sat_d * (1.0 - chroma_reliability) * 0.5 + val_d * 0.1
    per_method["hsv"] = _score_from_dists(hsv_dists)

    # 3) RGB Euclidean fallback
    rgb_dists = np.linalg.norm(palette - sample_rgb, axis=1)
    per_method["rgb"] = _score_from_dists(rgb_dists)

    # 4) Combined score
    if method == "lab":
        dists = lab_dists
    elif method == "hsv":
        dists = hsv_dists
    elif method == "rgb":
        dists = rgb_dists
    else:
        # combine Lab ΔE with a hue-agreement penalty that is small near the
        # sample hue and large for colors on the opposite side of the wheel.
        # This keeps Lab's accuracy while resolving spectral ambiguity.
        hue_penalty = ((hue_d / 180.0) ** 2) * 30.0 * chroma_reliability
        dists = lab_weight * lab_dists + hsv_weight * hue_penalty

    dists = np.maximum(dists, 0.0)

    nearest = int(np.argmin(dists))
    out = {
        "ph": nearest,
        "distance": float(dists[nearest]),
        "distances": dists.astype(float).tolist(),
        "per_method": {m: int(r["ph"]) for m, r in per_method.items()},
    }

    if use_interp:
        # weighted interpolation using inverse distance^2
        w = 1.0 / (dists ** 2 + 1e-6)
        interp = np.sum(w * np.arange(n)) / np.sum(w)
        out["ph_interp"] = float(interp)

    return out


def _score_from_dists(dists):
    """Return standard result sub-dict from a raw distance array."""
    nearest = int(np.argmin(dists))
    return {"ph": nearest, "distance": float(dists[nearest])}


# ---------------------------------------------------------------------------
# Interactive sample picker
# ---------------------------------------------------------------------------

def interactive_pick(image_path, size=20):
    """OpenCV window: click on the sample and press any key."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Չհաջողվեց բեռնել {image_path}")

    state = {"point": None, "done": False}

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["point"] = (x, y)
            preview = img.copy()
            cv2.circle(preview, (x, y), 4, (0, 0, 255), -1)
            cv2.rectangle(
                preview,
                (max(0, x - size), max(0, y - size)),
                (min(img.shape[1], x + size), min(img.shape[0], y + size)),
                (0, 255, 0),
                2,
            )
            cv2.imshow("Click sample region, then press any key", preview)

    cv2.imshow("Click sample region, then press any key", img)
    cv2.setMouseCallback("Click sample region, then press any key", mouse_cb)
    print("Կտտացրեք թեստ-շերտի գունավոր հատվածին, ապա սեղմեք որևէ ստեղն։")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if state["point"] is None:
        raise RuntimeError("Կետ ընտրված չէ։")
    return state["point"]


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def draw_result(palette_img, sample_img, palette, pal_positions, band, sample_region, result, save_path=None, show=False):
    """Draw palette bars and sample region with the matched pH value."""
    pal = cv2.imread(str(palette_img))
    smp = cv2.imread(str(sample_img))
    if pal is None or smp is None:
        return

    # mark palette band
    y1, y2 = band
    cv2.rectangle(pal, (0, y1), (pal.shape[1], y2), (255, 255, 255), 2)
    for i, (cx, cy) in enumerate(pal_positions):
        color = tuple(int(c) for c in palette[i])
        # invert-ish text color for readability
        text_color = (255, 255, 255) if sum(color) < 380 else (0, 0, 0)
        cv2.putText(pal, str(i), (cx - 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    x, y, w, h = sample_region
    ph = result["ph"]
    cv2.rectangle(smp, (x, y), (x + w, y + h), (0, 255, 0), 2)
    label = f"pH ~ {ph}"
    if "ph_interp" in result:
        label += f" ({result['ph_interp']:.1f})"
    cv2.putText(smp, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if save_path:
        cv2.imwrite(str(save_path), smp)
        print(f"Նմուշի պատկերը պահպանվել է՝ {save_path}")

    if show:
        cv2.imshow("Palette", pal)
        cv2.imshow("Sample", smp)
        print("Սեղմեք որևէ ստեղն պատուհանները փակելու համար։")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description="pH գույնի ճանաչում պատկերից")
    parser.add_argument("palette", help="pH գունապնակի պատկեր (օր. shkala.png)")
    parser.add_argument("sample", help="թեստ-նկար (օր. indicator.jpg)")
    parser.add_argument("--palette-json", help="նախապես ստեղծված պալիտրայի JSON ֆայլ")
    parser.add_argument("--save-palette", help="պահել ավտոմատ ստացված պալիտրան JSON-ում")
    parser.add_argument("--point", nargs=2, type=int, metavar=("X", "Y"), help="նմուշի կենտրոնի կոորդինատները")
    parser.add_argument("--size", type=int, default=20, help="նմուշի քառակուսու կողմի կես երկարություն (պիքսել)")
    parser.add_argument("--crop", nargs=4, type=int, metavar=("X", "Y", "W", "H"), help="նմուշի ուղղանկյուն տարածք")
    parser.add_argument("--interactive", action="store_true", help="ընտրել կետը մկնիկով")
    parser.add_argument("--visualize", action="store_true", help="ցույց տալ արդյունքի պատուհանը")
    parser.add_argument("--save", help="պահել արդյունքի պատկերը")
    parser.add_argument("--white-balance", action="store_true", help="կիրառել պարզ white-balance նմուշի վրա")
    parser.add_argument("--interp", action="store_true", help="արտածել նաև weighted pH արժեքը")
    parser.add_argument(
        "--method",
        choices=["lab", "hsv", "rgb", "combined"],
        default="combined",
        help="համապատասխանեցման մեթոդ (default: combined)",
    )
    parser.add_argument(
        "--lab-weight",
        type=float,
        default=0.75,
        help="Lab բաղադրիչի կշիռը combined մեթոդում (default 0.75)",
    )
    parser.add_argument(
        "--hsv-weight",
        type=float,
        default=0.25,
        help="HSV բաղադրիչի կշիռը combined մեթոդում (default 0.25)",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    palette_path = Path(args.palette)
    sample_path = Path(args.sample)

    if not palette_path.exists():
        print(f"Պալիտրայի ֆայլը չի գտնվել՝ {palette_path}", file=sys.stderr)
        return 1
    if not sample_path.exists():
        print(f"Նմուշի ֆայլը չի գտնվել՝ {sample_path}", file=sys.stderr)
        return 1

    # Load or extract palette
    if args.palette_json:
        with open(args.palette_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        palette = np.array([entry["rgb"] for entry in data], dtype=np.float32)
        positions = [(0, 0)] * len(palette)
        band = (0, 0)
    else:
        pal_img = Image.open(palette_path)
        palette, positions, band = extract_palette(pal_img, n_colors=15)
        if args.save_palette:
            out = [{"ph": i, "rgb": palette[i].astype(int).tolist()} for i in range(len(palette))]
            with open(args.save_palette, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            print(f"Պալիտրան պահպանվել է՝ {args.save_palette}")

    # Pick sample region
    sample_img = Image.open(sample_path)
    point = tuple(args.point) if args.point else None
    crop = tuple(args.crop) if args.crop else None

    if args.interactive:
        point = interactive_pick(sample_path, size=args.size)

    sample_rgb, sample_region = pick_sample_color(
        sample_img, point=point, crop=crop, size=args.size, white_balance=args.white_balance
    )

    # Match
    result = match_color(
        sample_rgb,
        palette,
        use_interp=args.interp,
        method=args.method,
        lab_weight=args.lab_weight,
        hsv_weight=args.hsv_weight,
    )

    # Report
    r, g, b = sample_rgb.astype(int)
    print(f"Նմուշի միջին գույն (RGB): ({r}, {g}, {b})")
    print(f"Մեթոդ: {args.method}")
    print(f"Ամենամոտիկ pH = {result['ph']} (հեռավորություն = {result['distance']:.2f})")
    print(f"  - Lab մեթոդով pH ≈ {result['per_method']['lab']}")
    print(f"  - HSV մեթոդով pH ≈ {result['per_method']['hsv']}")
    print(f"  - RGB մեթոդով pH ≈ {result['per_method']['rgb']}")
    if args.interp:
        print(f"Հարթեցված pH ≈ {result['ph_interp']:.2f}")

    if args.visualize or args.save:
        draw_result(
            palette_path,
            sample_path,
            palette,
            positions,
            band,
            sample_region,
            result,
            save_path=args.save,
            show=args.visualize,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
