#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phdet_web.py
Flask վեբ-հավելված՝ լուսանկարից pH որոշելու համար։

Օգտագործում.
    python3 phdet_web.py
    Բացել http://127.0.0.1:5000 զննարկիչում։
"""

import json
import os
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request, send_from_directory
from PIL import Image

from phdet import extract_palette, match_color, pick_sample_color

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

PAGE = """
<!doctype html>
<html lang="hy">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>pH գույնի ճանաչում</title>
    <style>
        :root {
            --bg: #f4f6f8;
            --card: #ffffff;
            --accent: #2563eb;
            --accent-hover: #1d4ed8;
            --text: #1f2937;
            --muted: #6b7280;
            --border: #d1d5db;
        }
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 2rem 1rem;
        }
        .container {
            max-width: 960px;
            margin: 0 auto;
        }
        h1 { text-align: center; margin-bottom: 0.5rem; }
        .subtitle { text-align: center; color: var(--muted); margin-bottom: 2rem; }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }
        @media (max-width: 720px) {
            .grid { grid-template-columns: 1fr; }
        }
        .card {
            background: var(--card);
            border-radius: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
            padding: 1.5rem;
        }
        .card h2 { margin-top: 0; font-size: 1.1rem; }
        label { display: block; margin-bottom: 0.5rem; font-weight: 500; }
        input[type="file"], input[type="number"], select {
            width: 100%;
            padding: 0.6rem;
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            font-size: 0.95rem;
        }
        .drop-zone {
            border: 2px dashed var(--border);
            border-radius: 0.75rem;
            padding: 1.25rem;
            text-align: center;
            color: var(--muted);
            cursor: pointer;
            transition: border-color 0.2s, background 0.2s;
        }
        .drop-zone:hover { border-color: var(--accent); background: #eff6ff; }
        .preview {
            margin-top: 1rem;
            max-width: 100%;
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            display: none;
        }
        .btn {
            width: 100%;
            background: var(--accent);
            color: white;
            border: none;
            padding: 0.9rem;
            border-radius: 0.6rem;
            font-size: 1rem;
            cursor: pointer;
            transition: background 0.2s;
        }
        .btn:hover { background: var(--accent-hover); }
        .btn:disabled { background: var(--muted); cursor: not-allowed; }
        .result {
            margin-top: 1.5rem;
            padding: 1.25rem;
            border-radius: 0.75rem;
            background: #ecfdf5;
            border: 1px solid #a7f3d0;
            display: none;
        }
        .result h3 { margin: 0 0 0.5rem; }
        .result p { margin: 0.3rem 0; }
        .error {
            margin-top: 1rem;
            padding: 1rem;
            border-radius: 0.5rem;
            background: #fee2e2;
            border: 1px solid #fecaca;
            color: #991b1b;
            display: none;
        }
        .info {
            margin-top: 1rem;
            font-size: 0.85rem;
            color: var(--muted);
        }
        .swatch {
            display: inline-block;
            width: 1.2rem;
            height: 1.2rem;
            border-radius: 50%;
            vertical-align: middle;
            margin-right: 0.5rem;
            border: 1px solid var(--border);
        }
        #sample-canvas { cursor: crosshair; width: 100%; border-radius: 0.5rem; }
        .coords { font-family: monospace; background: #f3f4f6; padding: 0.2rem 0.4rem; border-radius: 0.3rem; }
    </style>
</head>
<body>
    <div class="container">
        <h1>pH գույնի ճանաչում</h1>
        <p class="subtitle">Բեռնիր պալիտրայի և թեստ-շերտի նկարները, ընտրիր կետ և ստացիր pH</p>

        <form id="ph-form" enctype="multipart/form-data">
            <div class="grid">
                <div class="card">
                    <h2>1. Պալիտրա</h2>
                    <div class="drop-zone" onclick="document.getElementById('palette').click()">
                        Կտտացրիր բեռնելու համար կամ քաշիր նկարը<br>
                        <small>(pH գունապնակ)</small>
                    </div>
                    <input type="file" id="palette" name="palette" accept="image/*" hidden>
                    <img id="palette-preview" class="preview" alt="Palette preview">
                </div>

                <div class="card">
                    <h2>2. Թեստ-շերտ</h2>
                    <div class="drop-zone" onclick="document.getElementById('sample').click()">
                        Կտտացրիր բեռնելու համար կամ քաշիր նկարը<br>
                        <small>(թեստ-շերտ)</small>
                    </div>
                    <input type="file" id="sample" name="sample" accept="image/*" hidden>
                    <div style="position:relative;">
                        <canvas id="sample-canvas"></canvas>
                    </div>
                    <p class="info">Ընտրված կետ՝ <span id="point" class="coords">դեռ չի ընտրվել</span></p>
                    <input type="hidden" id="point-x" name="point_x">
                    <input type="hidden" id="point-y" name="point_y">
                </div>
            </div>

            <div class="card" style="margin-top: 1.5rem;">
                <h2>3. Կարգավորումներ</h2>
                <div class="grid" style="grid-template-columns: 1fr 1fr 1fr;">
                    <div>
                        <label for="size">Նմուշի չափ (px)</label>
                        <input type="number" id="size" name="size" value="40" min="5" max="200">
                    </div>
                    <div>
                        <label for="method">Մեթոդ</label>
                        <select id="method" name="method">
                            <option value="combined" selected>Combined (Lab + HSV)</option>
                            <option value="lab">Lab ΔE</option>
                            <option value="hsv">HSV</option>
                            <option value="rgb">RGB</option>
                        </select>
                    </div>
                    <div>
                        <label for="n_colors">Գույների քանակ</label>
                        <input type="number" id="n_colors" name="n_colors" value="14" min="2" max="30">
                    </div>
                </div>
                <label>
                    <input type="checkbox" name="white_balance"> Կիրառել white-balance
                </label>
                <label style="margin-left: 1rem;">
                    <input type="checkbox" name="interp" checked> Ցույց տալ հարթեցված արժեք
                </label>
            </div>

            <button type="submit" class="btn" id="submit-btn" disabled>Հաշվել pH</button>
        </form>

        <div id="error" class="error"></div>
        <div id="result" class="result">
            <h3>Արդյունք</h3>
            <p><span class="swatch" id="sample-swatch"></span> Նմուշի RGB՝ <span id="rgb"></span></p>
            <p>Ամենամոտիկ pH՝ <strong id="ph" style="font-size:1.5rem;"></strong></p>
            <p id="interp-line">Հարթեցված pH՝ <span id="ph-interp"></span></p>
            <p>Հեռավորություն՝ <span id="distance"></span></p>
            <p style="font-size:0.9rem; color:var(--muted);">
                Lab՝ <span id="lab-ph"></span>, HSV՝ <span id="hsv-ph"></span>, RGB՝ <span id="rgb-ph"></span>
            </p>
            <img id="result-img" style="margin-top:1rem; max-width:100%; border-radius:0.5rem; display:none;" alt="Result">
        </div>
    </div>

    <script>
        const paletteInput = document.getElementById('palette');
        const sampleInput = document.getElementById('sample');
        const palettePreview = document.getElementById('palette-preview');
        const sampleCanvas = document.getElementById('sample-canvas');
        const ctx = sampleCanvas.getContext('2d');
        const pointSpan = document.getElementById('point');
        const pointX = document.getElementById('point-x');
        const pointY = document.getElementById('point-y');
        const submitBtn = document.getElementById('submit-btn');
        const form = document.getElementById('ph-form');
        const errorDiv = document.getElementById('error');
        const resultDiv = document.getElementById('result');

        let sampleImg = null;
        let scale = 1;

        function showError(msg) {
            errorDiv.textContent = msg;
            errorDiv.style.display = 'block';
            resultDiv.style.display = 'none';
        }
        function clearError() {
            errorDiv.textContent = '';
            errorDiv.style.display = 'none';
        }
        function updateSubmit() {
            submitBtn.disabled = !(paletteInput.files[0] && sampleInput.files[0] && pointX.value);
        }

        function handleFile(input, preview, isSample) {
            const file = input.files[0];
            if (!file) return;
            clearError();
            const url = URL.createObjectURL(file);
            if (isSample) {
                sampleImg = new Image();
                sampleImg.onload = function() {
                    sampleCanvas.width = sampleImg.naturalWidth;
                    sampleCanvas.height = sampleImg.naturalHeight;
                    // fit canvas to container while keeping aspect ratio
                    const maxWidth = sampleCanvas.parentElement.clientWidth;
                    scale = maxWidth / sampleImg.naturalWidth;
                    sampleCanvas.style.width = maxWidth + 'px';
                    sampleCanvas.style.height = (sampleImg.naturalHeight * scale) + 'px';
                    ctx.drawImage(sampleImg, 0, 0);
                    sampleCanvas.style.display = 'block';
                };
                sampleImg.src = url;
            } else {
                preview.src = url;
                preview.style.display = 'block';
            }
            updateSubmit();
        }

        paletteInput.addEventListener('change', () => handleFile(paletteInput, palettePreview, false));
        sampleInput.addEventListener('change', () => handleFile(sampleInput, null, true));

        sampleCanvas.addEventListener('click', function(e) {
            if (!sampleImg) return;
            const rect = sampleCanvas.getBoundingClientRect();
            const x = (e.clientX - rect.left) / scale;
            const y = (e.clientY - rect.top) / scale;
            pointX.value = Math.round(x);
            pointY.value = Math.round(y);
            pointSpan.textContent = `(${Math.round(x)}, ${Math.round(y)})`;

            // redraw with marker
            ctx.drawImage(sampleImg, 0, 0);
            ctx.strokeStyle = '#00ff00';
            ctx.lineWidth = 4;
            const size = parseInt(document.getElementById('size').value) || 40;
            ctx.strokeRect(x - size, y - size, size * 2, size * 2);
            ctx.fillStyle = '#00ff00';
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, 2 * Math.PI);
            ctx.fill();
            updateSubmit();
        });

        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            clearError();
            submitBtn.disabled = true;
            submitBtn.textContent = 'Հաշվում եմ...';
            const formData = new FormData(form);
            try {
                const resp = await fetch('/analyze', { method: 'POST', body: formData });
                const data = await resp.json();
                if (!resp.ok || data.error) {
                    showError(data.error || 'Անհայտ սխալ');
                    return;
                }
                document.getElementById('sample-swatch').style.backgroundColor = `rgb(${data.rgb.join(',')})`;
                document.getElementById('rgb').textContent = `(${data.rgb.join(', ')})`;
                document.getElementById('ph').textContent = data.ph;
                document.getElementById('distance').textContent = data.distance.toFixed(2);
                document.getElementById('lab-ph').textContent = data.per_method.lab;
                document.getElementById('hsv-ph').textContent = data.per_method.hsv;
                document.getElementById('rgb-ph').textContent = data.per_method.rgb;
                const interpLine = document.getElementById('interp-line');
                if (data.ph_interp !== undefined) {
                    interpLine.style.display = 'block';
                    document.getElementById('ph-interp').textContent = data.ph_interp.toFixed(2);
                } else {
                    interpLine.style.display = 'none';
                }
                if (data.result_image) {
                    const img = document.getElementById('result-img');
                    img.src = data.result_image;
                    img.style.display = 'block';
                }
                resultDiv.style.display = 'block';
            } catch (err) {
                showError(err.message);
            } finally {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Հաշվել pH';
            }
        });
    </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def save_upload(file_obj):
    """Save uploaded file to uploads dir and return Path."""
    ext = Path(file_obj.filename).suffix.lower() or ".png"
    name = f"{uuid.uuid4().hex}{ext}"
    path = UPLOAD_DIR / name
    file_obj.save(path)
    return path


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        palette_file = request.files.get("palette")
        sample_file = request.files.get("sample")
        if not palette_file or not sample_file:
            return jsonify({"error": "Անհրաժեշտ են երկու նկարները"}), 400

        palette_path = save_upload(palette_file)
        sample_path = save_upload(sample_file)

        point_x = request.form.get("point_x", type=int)
        point_y = request.form.get("point_y", type=int)
        size = request.form.get("size", 40, type=int)
        method = request.form.get("method", "combined")
        n_colors = request.form.get("n_colors", 14, type=int)
        white_balance = request.form.get("white_balance") == "on"
        use_interp = request.form.get("interp") == "on"

        if point_x is None or point_y is None:
            # default to center
            img = Image.open(sample_path)
            point_x = img.width // 2
            point_y = img.height // 2

        # Extract palette
        pal_img = Image.open(palette_path)
        palette, positions, band = extract_palette(pal_img, n_colors=n_colors)

        # Pick sample
        smp_img = Image.open(sample_path)
        sample_rgb, sample_region = pick_sample_color(
            smp_img,
            point=(point_x, point_y),
            size=size,
            white_balance=white_balance,
        )

        # Match
        result = match_color(sample_rgb, palette, use_interp=use_interp, method=method)

        # Draw result image
        result_name = f"result_{uuid.uuid4().hex}.jpg"
        result_path = UPLOAD_DIR / result_name
        pal_bgr = cv2.imread(str(palette_path))
        smp_bgr = cv2.imread(str(sample_path))
        if pal_bgr is not None and smp_bgr is not None:
            y1, y2 = band
            cv2.rectangle(pal_bgr, (0, y1), (pal_bgr.shape[1], y2), (255, 255, 255), 3)
            for i, (cx, cy) in enumerate(positions):
                color = tuple(int(c) for c in palette[i])
                text_color = (255, 255, 255) if sum(color) < 380 else (0, 0, 0)
                cv2.putText(pal_bgr, str(i), (cx - 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

            x, y, w, h = sample_region
            ph = result["ph"]
            cv2.rectangle(smp_bgr, (x, y), (x + w, y + h), (0, 255, 0), 3)
            label = f"pH ~ {ph}"
            if "ph_interp" in result:
                label += f" ({result['ph_interp']:.1f})"
            cv2.putText(smp_bgr, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # concatenate side by side
            target_h = 400
            pal_resized = cv2.resize(pal_bgr, None, fx=target_h / pal_bgr.shape[0], fy=target_h / pal_bgr.shape[0])
            smp_resized = cv2.resize(smp_bgr, None, fx=target_h / smp_bgr.shape[0], fy=target_h / smp_bgr.shape[0])
            combined = np.hstack([pal_resized, smp_resized])
            cv2.imwrite(str(result_path), combined)
            result_image_url = f"/uploads/{result_name}"
        else:
            result_image_url = None

        return jsonify(
            {
                "ph": result["ph"],
                "distance": result["distance"],
                "rgb": [int(c) for c in sample_rgb],
                "per_method": result.get("per_method", {}),
                "ph_interp": result.get("ph_interp"),
                "result_image": result_image_url,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Բացիր http://127.0.0.1:5000 զննարկիչում")
    app.run(host="127.0.0.1", port=5000, debug=True)
