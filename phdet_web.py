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

from phdet import aggregate_sample_colors, extract_palette, match_color

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
        .point-list {
            margin-top: 0.5rem;
            max-height: 120px;
            overflow-y: auto;
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            padding: 0.5rem;
            font-size: 0.85rem;
            display: none;
        }
        .point-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.25rem 0;
            border-bottom: 1px solid #f3f4f6;
        }
        .point-item:last-child { border-bottom: none; }
        .point-remove {
            background: #fee2e2;
            color: #991b1b;
            border: none;
            border-radius: 0.3rem;
            padding: 0.1rem 0.4rem;
            cursor: pointer;
            font-size: 0.8rem;
        }
        .point-actions {
            margin-top: 0.5rem;
            display: flex;
            gap: 0.5rem;
        }
        .point-actions button {
            flex: 1;
            padding: 0.4rem;
            border: 1px solid var(--border);
            border-radius: 0.4rem;
            background: white;
            cursor: pointer;
            font-size: 0.85rem;
        }
        .hint {
            font-size: 0.8rem;
            color: var(--muted);
            margin-top: 0.25rem;
        }
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
                    <p class="hint">Ձախ կտտոց՝ ավելացնել կետ, աջ կտտոց՝ ջնջել ամենամոտիկը</p>
                    <div id="point-list" class="point-list"></div>
                    <div class="point-actions">
                        <button type="button" id="clear-points">Մաքրել կետերը</button>
                    </div>
                    <input type="hidden" id="points-json" name="points_json">
                </div>
            </div>

            <div class="card" style="margin-top: 1.5rem;">
                <h2>3. Կարգավորումներ</h2>
                <div class="grid" style="grid-template-columns: 1fr 1fr 1fr 1fr;">
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
                        <label for="aggregate">Համախմբում</label>
                        <select id="aggregate" name="aggregate">
                            <option value="median" selected>Median</option>
                            <option value="mean">Mean</option>
                            <option value="robust">Robust</option>
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
            <p id="points-info" style="display:none;">Ընտրված կետերի քանակ՝ <strong id="n-points"></strong>, համախմբման մեթոդ՝ <span id="agg-method"></span>, վստահելիություն՝ <strong id="confidence"></strong></p>
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
        const pointsJson = document.getElementById('points-json');
        const pointList = document.getElementById('point-list');
        const clearPointsBtn = document.getElementById('clear-points');
        const submitBtn = document.getElementById('submit-btn');
        const form = document.getElementById('ph-form');
        const errorDiv = document.getElementById('error');
        const resultDiv = document.getElementById('result');

        let sampleImg = null;
        let scale = 1;
        let points = [];

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
            submitBtn.disabled = !(paletteInput.files[0] && sampleInput.files[0] && points.length > 0);
        }

        function updatePointsJson() {
            pointsJson.value = JSON.stringify(points);
        }

        function renderPoints() {
            pointList.innerHTML = '';
            if (points.length === 0) {
                pointList.style.display = 'none';
                updateSubmit();
                updatePointsJson();
                redrawCanvas();
                return;
            }
            pointList.style.display = 'block';
            points.forEach((p, i) => {
                const item = document.createElement('div');
                item.className = 'point-item';
                item.innerHTML = `
                    <span>#${i + 1}: (${Math.round(p.x)}, ${Math.round(p.y)})</span>
                    <button type="button" class="point-remove" data-index="${i}">×</button>
                `;
                pointList.appendChild(item);
            });
            pointList.querySelectorAll('.point-remove').forEach(btn => {
                btn.addEventListener('click', function() {
                    points.splice(parseInt(this.dataset.index), 1);
                    renderPoints();
                });
            });
            updatePointsJson();
            updateSubmit();
            redrawCanvas();
        }

        function redrawCanvas() {
            if (!sampleImg) return;
            ctx.drawImage(sampleImg, 0, 0);
            ctx.strokeStyle = '#00ff00';
            ctx.lineWidth = 4;
            ctx.fillStyle = '#00ff00';
            const size = parseInt(document.getElementById('size').value) || 40;
            points.forEach((p, i) => {
                ctx.strokeRect(p.x - size, p.y - size, size * 2, size * 2);
                ctx.beginPath();
                ctx.arc(p.x, p.y, 5, 0, 2 * Math.PI);
                ctx.fill();
                ctx.font = '16px sans-serif';
                ctx.fillText(String(i + 1), p.x + 8, p.y - 8);
            });
        }

        function handleFile(input, preview, isSample) {
            const file = input.files[0];
            if (!file) return;
            clearError();
            const url = URL.createObjectURL(file);
            if (isSample) {
                points = [];
                renderPoints();
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

        document.getElementById('size').addEventListener('input', function() {
            redrawCanvas();
        });

        paletteInput.addEventListener('change', () => handleFile(paletteInput, palettePreview, false));
        sampleInput.addEventListener('change', () => handleFile(sampleInput, null, true));

        sampleCanvas.addEventListener('click', function(e) {
            if (!sampleImg) return;
            const rect = sampleCanvas.getBoundingClientRect();
            const x = (e.clientX - rect.left) / scale;
            const y = (e.clientY - rect.top) / scale;
            points.push({x: x, y: y});
            renderPoints();
        });

        sampleCanvas.addEventListener('contextmenu', function(e) {
            e.preventDefault();
            if (!sampleImg || points.length === 0) return;
            const rect = sampleCanvas.getBoundingClientRect();
            const x = (e.clientX - rect.left) / scale;
            const y = (e.clientY - rect.top) / scale;
            let nearest = 0;
            let nearestDist = Infinity;
            points.forEach((p, i) => {
                const d = (p.x - x) ** 2 + (p.y - y) ** 2;
                if (d < nearestDist) {
                    nearestDist = d;
                    nearest = i;
                }
            });
            points.splice(nearest, 1);
            renderPoints();
        });

        clearPointsBtn.addEventListener('click', function() {
            points = [];
            renderPoints();
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
                const pointsInfo = document.getElementById('points-info');
                if (data.n_points && data.n_points > 1) {
                    pointsInfo.style.display = 'block';
                    document.getElementById('n-points').textContent = data.n_points;
                    document.getElementById('agg-method').textContent = data.aggregate || 'median';
                    document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(1) + '%';
                } else {
                    pointsInfo.style.display = 'none';
                }
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

        points_json = request.form.get("points_json")
        size = request.form.get("size", 40, type=int)
        method = request.form.get("method", "combined")
        aggregate = request.form.get("aggregate", "median")
        n_colors = request.form.get("n_colors", 14, type=int)
        white_balance = request.form.get("white_balance") == "on"
        use_interp = request.form.get("interp") == "on"

        # Parse points (fallback to image center if none provided)
        smp_img = Image.open(sample_path)
        if points_json:
            raw_points = json.loads(points_json)
            points = [(int(p["x"]), int(p["y"])) for p in raw_points]
        else:
            points = [(smp_img.width // 2, smp_img.height // 2)]

        # Extract palette
        pal_img = Image.open(palette_path)
        palette, positions, band, ph_min = extract_palette(pal_img, n_colors=n_colors)

        # Pick sample color(s)
        sample_rgb, confidence, details = aggregate_sample_colors(
            smp_img, points, size=size, method=aggregate, white_balance=white_balance
        )
        n_points = len(points)

        # Match
        result = match_color(sample_rgb, palette, use_interp=use_interp, method=method)

        # Convert internal 0-based index to actual pH value
        ph_value = result["ph"] + ph_min
        ph_interp_value = result.get("ph_interp")
        if ph_interp_value is not None:
            ph_interp_value += ph_min
        per_method_values = {m: v + ph_min for m, v in result.get("per_method", {}).items()}

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
                cv2.putText(pal_bgr, str(i + ph_min), (cx - 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

            for i, d in enumerate(details):
                x, y, w, h = d["region"]
                cv2.rectangle(smp_bgr, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.circle(smp_bgr, (x + w // 2, y + h // 2), 5, (0, 0, 255), -1)
                cv2.putText(smp_bgr, str(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            label = f"pH ~ {ph_value}"
            if ph_interp_value is not None:
                label += f" ({ph_interp_value:.1f})"
            # place final label at top-left of sample image
            cv2.putText(smp_bgr, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

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
                "ph": ph_value,
                "distance": result["distance"],
                "rgb": [int(c) for c in sample_rgb],
                "per_method": per_method_values,
                "ph_interp": ph_interp_value,
                "result_image": result_image_url,
                "n_points": n_points,
                "aggregate": aggregate,
                "confidence": confidence,
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
