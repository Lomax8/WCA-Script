#!/usr/bin/env python
"""
Water Contact Angle (WCA) Modern Web Application
=================================================
Interactive web UI for sessile droplet contact angle measurement with:
- Asymmetric Droplet Modeling (Apex Lateral Shift in 2D to account for droplet asymmetry & contact angle hysteresis)
- Full Original Image Framing (no artificial small bounding box)
- Center Repositioning (Pan) via Canvas Click-and-Drag & Sliders + "Center Droplet" button
- Expanded Crop Dimensions (up to 100% full original image resolution)
- Modeled Droplet Shape Curve (accurately traces droplet cap through apex handle)
- Multi-Image Batch Queue & Filmstrip
- Interactive Canvas for each image (Baseline handles + 2D Apex height & asymmetry handle)
- Real-time Laboratory Statistical Dashboard:
    * Mean Contact Angle (theta_mean)
    * Standard Deviation (SD, sigma)
    * Standard Error of the Mean (SEM = SD / sqrt(N))
    * 95% Confidence Interval (Student's t-distribution)
    * Droplet Asymmetry (theta_L vs theta_R)
    * Min / Max Range
- Individual include/exclude toggle per image to remove outliers from statistics
- Live synchronization: tweaking handles/sliders updates image & aggregate stats instantly
- Image Rotation (-10° to +10°) with fine nudges & Auto-Level
- High-Impact Curve Sliders (Angle offset +/- 25°, Height offset +/- 80px, Lateral Shift +/- 200px)
- Clean scientific rendering (zero pixel clutter on image)
- Full Color Customization & Toggles
- Comprehensive CSV Export (all raw measurements + full statistical block)
- One-Click Lab Notebook summary copy

Zero external GUI dependencies: Python http.server + OpenCV.
"""

import os
import sys
import io
import json
import base64
import glob
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import webbrowser
import cv2
import numpy as np

from wca_image_analyzer import (
    detect_baseline_y,
    extract_droplet_edges,
    fit_circle_wca,
)

PORT = 5000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def process_image_raw(img_bytes: bytes, filename="upload.jpg", manual_baseline=None):
    """
    Process raw image bytes and return full original image, edge points, and baseline handles in full image coordinates.
    """
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image file.")

    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if manual_baseline is not None and manual_baseline > 0:
        baseline_y = int(manual_baseline)
    else:
        baseline_y = detect_baseline_y(gray)

    edge_pts = extract_droplet_edges(gray, baseline_y)
    if len(edge_pts) < 25:
        raise RuntimeError("Could not detect sufficient droplet contour points. Try adjusting lighting or threshold.")

    fit_res = fit_circle_wca(edge_pts, baseline_y)
    if fit_res is None:
        raise RuntimeError("Droplet contour did not intersect baseline.")

    xc, yc = fit_res["xc"], fit_res["yc"]
    R = fit_res["R"]
    x_left, x_right = fit_res["x_left"], fit_res["x_right"]
    apex_y = yc - R

    # Full original image encoded as high-quality JPEG
    _, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    full_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

    full_edge_pts = [[round(float(pt[0]), 1), round(float(pt[1]), 1)] for pt in edge_pts]

    handle_left = [round(float(x_left), 1), round(float(baseline_y), 1)]
    handle_right = [round(float(x_right), 1), round(float(baseline_y), 1)]
    handle_apex = [round(float(xc), 1), round(float(apex_y), 1)]
    droplet_center = {"x": round(float(xc), 1), "y": round(float((baseline_y + apex_y) / 2.0), 1)}

    # High-quality thumbnail for session gallery
    c_y1 = max(0, int(apex_y - 60))
    c_y2 = min(h, int(baseline_y + 60))
    c_x1 = max(0, int(x_left - 80))
    c_x2 = min(w, int(x_right + 80))
    thumb_crop = img[c_y1:c_y2, c_x1:c_x2] if (c_y2 > c_y1 and c_x2 > c_x1) else img
    thumb_h = 70
    scale = thumb_h / float(max(1, thumb_crop.shape[0]))
    thumb_w = int(thumb_crop.shape[1] * scale)
    thumb = cv2.resize(thumb_crop, (max(1, thumb_w), thumb_h), interpolation=cv2.INTER_AREA)
    _, thumb_buf = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    thumb_b64 = "data:image/jpeg;base64," + base64.b64encode(thumb_buf).decode("utf-8")

    wetted_base_px = round(float(2 * np.sqrt(max(10.0, R**2 - (yc - baseline_y)**2))), 1)
    droplet_height_px = round(float(baseline_y - apex_y), 1)

    return {
        "success": True,
        "filename": filename,
        "full_image": full_b64,
        "crop_image": full_b64,
        "thumb_image": thumb_b64,
        "img_width": w,
        "img_height": h,
        "droplet_center": droplet_center,
        "edge_points": full_edge_pts,
        "baseline_handles": {"left": handle_left, "right": handle_right},
        "apex_handle": handle_apex,
        "initial_fit": {
            "xc": round(float(xc), 2),
            "yc": round(float(yc), 2),
            "R": round(float(R), 2),
            "wca_deg": round(float(fit_res["theta_deg"]), 2),
            "baseline_y": round(float(baseline_y), 1),
            "width_px": wetted_base_px,
            "height_px": droplet_height_px,
        },
    }


HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>WCA Studio &bull; Multi-Droplet Contact Angle &amp; Statistical Analysis</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-primary: #0a0e17;
      --bg-card: rgba(18, 24, 38, 0.88);
      --bg-card-hover: rgba(26, 35, 54, 0.95);
      --border-color: rgba(255, 255, 255, 0.08);
      --border-focus: rgba(56, 189, 248, 0.4);
      --accent: #38bdf8;
      --accent-glow: rgba(56, 189, 248, 0.25);
      --accent-grad: linear-gradient(135deg, #38bdf8 0%, #0284c7 100%);
      --emerald: #10b981;
      --emerald-grad: linear-gradient(135deg, #34d399 0%, #059669 100%);
      --amber: #f59e0b;
      --rose: #f43f5e;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --radius-sm: 8px;
      --radius-md: 12px;
      --radius-lg: 18px;
      --shadow-card: 0 10px 30px -10px rgba(0, 0, 0, 0.55);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg-primary);
      color: var(--text);
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      min-height: 100vh;
      padding: 20px 24px 60px;
      overflow-x: hidden;
      background-image:
        radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.08) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(14, 165, 233, 0.05) 0px, transparent 50%);
    }
    .app-shell { max-width: 1560px; margin: 0 auto; display: flex; flex-direction: column; gap: 18px; }

    /* Top Navigation Bar */
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 16px 24px;
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      backdrop-filter: blur(16px);
      box-shadow: var(--shadow-card);
    }
    .brand { display: flex; align-items: center; gap: 14px; }
    .brand-icon {
      width: 42px;
      height: 42px;
      background: var(--accent-grad);
      border-radius: var(--radius-md);
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 24px var(--accent-glow);
    }
    .brand-icon svg { width: 24px; height: 24px; fill: #0a0e17; }
    .brand h1 { font-size: 1.25rem; font-weight: 800; letter-spacing: -0.02em; }
    .brand-tag {
      font-size: 0.68rem;
      font-weight: 700;
      color: var(--accent);
      background: rgba(56, 189, 248, 0.12);
      border: 1px solid rgba(56, 189, 248, 0.28);
      padding: 3px 9px;
      border-radius: 20px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .header-actions { display: flex; gap: 10px; align-items: center; }

    /* Buttons */
    .btn {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      padding: 8px 15px;
      font-size: 0.8rem;
      font-weight: 600;
      border-radius: var(--radius-sm);
      border: 1px solid transparent;
      cursor: pointer;
      transition: all 0.18s cubic-bezier(0.4, 0, 0.2, 1);
      font-family: inherit;
    }
    .btn-primary {
      background: var(--accent-grad);
      color: #0a0e17;
      box-shadow: 0 4px 14px rgba(56, 189, 248, 0.3);
    }
    .btn-primary:hover {
      box-shadow: 0 6px 20px rgba(56, 189, 248, 0.5);
      transform: translateY(-1px);
    }
    .btn-success {
      background: var(--emerald-grad);
      color: #0a0e17;
      box-shadow: 0 4px 14px rgba(16, 185, 129, 0.25);
    }
    .btn-success:hover {
      box-shadow: 0 6px 20px rgba(16, 185, 129, 0.45);
      transform: translateY(-1px);
    }
    .btn-outline {
      background: rgba(255, 255, 255, 0.03);
      border-color: var(--border-color);
      color: var(--text);
    }
    .btn-outline:hover {
      background: rgba(255, 255, 255, 0.08);
      border-color: rgba(255, 255, 255, 0.2);
    }
    .btn-danger {
      background: rgba(244, 63, 94, 0.12);
      border-color: rgba(244, 63, 94, 0.3);
      color: var(--rose);
    }
    .btn-danger:hover {
      background: rgba(244, 63, 94, 0.25);
    }

    /* Aggregate Statistics Hero Banner */
    .stats-hero-banner {
      display: grid;
      grid-template-columns: 1.4fr 1.2fr 1fr 1fr;
      gap: 16px;
    }
    @media (max-width: 1200px) { .stats-hero-banner { grid-template-columns: 1fr 1fr; } }
    @media (max-width: 700px) { .stats-hero-banner { grid-template-columns: 1fr; } }

    .stat-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      padding: 18px 22px;
      backdrop-filter: blur(16px);
      box-shadow: var(--shadow-card);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      position: relative;
      overflow: hidden;
    }
    .stat-card.featured {
      border-color: rgba(56, 189, 248, 0.35);
      background: linear-gradient(135deg, rgba(56, 189, 248, 0.08) 0%, rgba(18, 24, 38, 0.95) 100%);
      box-shadow: 0 10px 30px -10px rgba(56, 189, 248, 0.25);
    }
    .stat-card.featured::after {
      content: "";
      position: absolute;
      top: 0; left: 0; right: 0; height: 3px;
      background: var(--accent-grad);
    }
    .stat-card.accent-green::after {
      content: "";
      position: absolute;
      top: 0; left: 0; right: 0; height: 3px;
      background: var(--emerald-grad);
    }
    .stat-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 6px;
    }
    .stat-label {
      font-size: 0.72rem;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .stat-pill {
      font-size: 0.68rem;
      font-weight: 700;
      padding: 2px 7px;
      border-radius: 12px;
      background: rgba(255, 255, 255, 0.06);
      color: var(--text-muted);
      font-family: 'JetBrains Mono', monospace;
    }
    .stat-pill.pill-cyan { background: rgba(56, 189, 248, 0.15); color: var(--accent); }
    .stat-pill.pill-green { background: rgba(16, 185, 129, 0.15); color: var(--emerald); }

    .stat-val-huge {
      font-size: 2.25rem;
      font-weight: 800;
      color: var(--accent);
      font-family: 'Plus Jakarta Sans', sans-serif;
      letter-spacing: -0.03em;
      line-height: 1.1;
      display: flex;
      align-items: baseline;
      gap: 6px;
    }
    .stat-val-huge .error-text {
      font-size: 1.25rem;
      font-weight: 700;
      color: #7dd3fc;
    }
    .stat-sub-details {
      font-size: 0.76rem;
      color: var(--text-dim);
      margin-top: 6px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    .stat-sub-details span strong { color: var(--text); }

    /* Multi-Image Filmstrip / Queue Section */
    .filmstrip-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      padding: 16px 20px;
      backdrop-filter: blur(16px);
      box-shadow: var(--shadow-card);
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .filmstrip-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
    }
    .filmstrip-title {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 0.82rem;
      font-weight: 700;
      letter-spacing: 0.04em;
      color: var(--text-muted);
      text-transform: uppercase;
    }
    .filmstrip-count-badge {
      font-size: 0.72rem;
      padding: 3px 9px;
      border-radius: 12px;
      background: rgba(56, 189, 248, 0.15);
      color: var(--accent);
      font-weight: 700;
      font-family: 'JetBrains Mono', monospace;
    }
    .filmstrip-actions {
      display: flex;
      gap: 8px;
      align-items: center;
    }

    /* Filmstrip Carousel */
    .filmstrip-track {
      display: flex;
      gap: 12px;
      overflow-x: auto;
      padding: 6px 2px 10px;
      scroll-behavior: smooth;
    }
    .filmstrip-track::-webkit-scrollbar {
      height: 6px;
    }
    .filmstrip-track::-webkit-scrollbar-thumb {
      background: rgba(255, 255, 255, 0.15);
      border-radius: 10px;
    }

    .thumb-item {
      flex: 0 0 160px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 8px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      cursor: pointer;
      position: relative;
      transition: all 0.18s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .thumb-item:hover {
      background: var(--bg-card-hover);
      border-color: rgba(56, 189, 248, 0.3);
      transform: translateY(-2px);
    }
    .thumb-item.active {
      border-color: var(--accent);
      background: rgba(56, 189, 248, 0.08);
      box-shadow: 0 0 16px rgba(56, 189, 248, 0.3);
    }
    .thumb-item.excluded {
      opacity: 0.45;
      filter: grayscale(0.7);
    }
    .thumb-img-wrapper {
      width: 100%;
      height: 62px;
      border-radius: var(--radius-sm);
      overflow: hidden;
      background: #000;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .thumb-img-wrapper img {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
    .thumb-footer {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .thumb-name {
      font-size: 0.7rem;
      font-weight: 700;
      color: var(--text);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 85px;
      font-family: 'JetBrains Mono', monospace;
    }
    .thumb-angle-badge {
      font-size: 0.72rem;
      font-weight: 800;
      color: var(--accent);
      font-family: 'JetBrains Mono', monospace;
      background: rgba(56, 189, 248, 0.14);
      padding: 1px 6px;
      border-radius: 4px;
    }
    .thumb-toggle-include {
      position: absolute;
      top: 5px;
      left: 5px;
      z-index: 2;
      background: rgba(0, 0, 0, 0.65);
      border-radius: 4px;
      padding: 2px 4px;
      display: flex;
      align-items: center;
      gap: 3px;
      font-size: 0.65rem;
      color: #e2e8f0;
      cursor: pointer;
    }
    .thumb-remove-btn {
      position: absolute;
      top: 5px;
      right: 5px;
      z-index: 2;
      background: rgba(0, 0, 0, 0.65);
      color: #94a3b8;
      border: none;
      border-radius: 50%;
      width: 18px;
      height: 18px;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      font-size: 11px;
      transition: all 0.15s ease;
    }
    .thumb-remove-btn:hover {
      background: var(--rose);
      color: #fff;
    }

    /* Main Grid Layout */
    .layout-grid {
      display: grid;
      grid-template-columns: 410px 1fr;
      gap: 18px;
      align-items: start;
    }
    @media (max-width: 1150px) { .layout-grid { grid-template-columns: 1fr; } }

    /* Cards */
    .card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      padding: 18px;
      backdrop-filter: blur(16px);
      box-shadow: var(--shadow-card);
      position: relative;
    }
    .card-title {
      font-size: 0.82rem;
      font-weight: 700;
      letter-spacing: 0.04em;
      color: var(--text-muted);
      text-transform: uppercase;
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    /* Drop Zone */
    .dropzone {
      border: 2px dashed rgba(255, 255, 255, 0.14);
      border-radius: var(--radius-md);
      padding: 16px 14px;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
      background: rgba(255, 255, 255, 0.02);
    }
    .dropzone:hover, .dropzone.dragover {
      border-color: var(--accent);
      background: rgba(56, 189, 248, 0.06);
    }
    .dropzone svg { width: 28px; height: 28px; fill: var(--accent); margin-bottom: 4px; }
    .dropzone p { font-size: 0.84rem; font-weight: 600; margin-bottom: 2px; }
    .dropzone span { font-size: 0.7rem; color: var(--text-dim); }
    input[type="file"] { display: none; }

    /* Control Rows */
    .control-row { margin-bottom: 11px; }
    .control-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.76rem;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 5px;
    }
    .control-val {
      color: var(--accent);
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.8rem;
    }
    input[type="range"] {
      width: 100%;
      height: 5px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 3px;
      outline: none;
      -webkit-appearance: none;
    }
    input[type="range"]::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 15px;
      height: 15px;
      background: var(--accent);
      border-radius: 50%;
      cursor: pointer;
      box-shadow: 0 0 10px var(--accent);
      transition: transform 0.1s ease;
    }
    input[type="range"]::-webkit-slider-thumb:hover { transform: scale(1.2); }

    .btn-nudge-row { display: flex; gap: 5px; margin-top: 5px; }
    .btn-nudge {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--border-color);
      color: var(--text-muted);
      font-size: 0.7rem;
      font-weight: 600;
      padding: 4px 6px;
      border-radius: var(--radius-sm);
      cursor: pointer;
      flex: 1;
      transition: all 0.15s ease;
      text-align: center;
    }
    .btn-nudge:hover { background: rgba(56, 189, 248, 0.15); color: var(--accent); }

    /* Color Pickers */
    .color-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 7px;
      margin-top: 6px;
    }
    .color-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-sm);
      padding: 5px 8px;
      font-size: 0.73rem;
      font-weight: 600;
      color: var(--text-muted);
    }
    .color-item input[type="color"] {
      -webkit-appearance: none;
      border: none;
      width: 20px;
      height: 20px;
      border-radius: 50%;
      cursor: pointer;
      background: none;
    }
    .color-item input[type="color"]::-webkit-color-swatch-wrapper { padding: 0; }
    .color-item input[type="color"]::-webkit-color-swatch {
      border: 1px solid rgba(255, 255, 255, 0.2);
      border-radius: 50%;
    }

    /* Switches */
    .switch-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 7px;
      font-size: 0.75rem;
      font-weight: 600;
      color: var(--text-muted);
    }
    .toggle-switch {
      position: relative;
      width: 32px;
      height: 17px;
    }
    .toggle-switch input { opacity: 0; width: 0; height: 0; }
    .slider-track {
      position: absolute;
      cursor: pointer;
      top: 0; left: 0; right: 0; bottom: 0;
      background-color: rgba(255, 255, 255, 0.1);
      transition: 0.2s;
      border-radius: 20px;
    }
    .slider-track:before {
      position: absolute;
      content: "";
      height: 11px;
      width: 11px;
      left: 3px;
      bottom: 3px;
      background-color: #fff;
      transition: 0.2s;
      border-radius: 50%;
    }
    input:checked + .slider-track { background-color: var(--accent); }
    input:checked + .slider-track:before { transform: translateX(15px); }

    /* Canvas Viewport */
    .canvas-viewport {
      position: relative;
      background: #000;
      border-radius: var(--radius-lg);
      border: 1px solid var(--border-color);
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6);
      min-height: 520px;
      max-height: 72vh;
    }
    canvas {
      display: block;
      touch-action: none;
      max-width: 100%;
      max-height: 72vh;
      object-fit: contain;
      cursor: move;
    }
    .canvas-overlay-hint {
      position: absolute;
      bottom: 12px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(10, 14, 23, 0.85);
      border: 1px solid var(--border-color);
      border-radius: 20px;
      padding: 5px 14px;
      font-size: 0.72rem;
      color: var(--text-muted);
      pointer-events: none;
      backdrop-filter: blur(8px);
      display: flex;
      align-items: center;
      gap: 6px;
      white-space: nowrap;
    }
    .canvas-overlay-hint span { color: var(--accent); font-weight: 700; }

    /* Comparison Table Section */
    .comparison-section {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      padding: 20px;
      backdrop-filter: blur(16px);
      box-shadow: var(--shadow-card);
    }
    .table-responsive {
      overflow-x: auto;
      margin-top: 10px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.82rem;
    }
    th {
      text-align: left;
      padding: 10px 12px;
      color: var(--text-dim);
      font-weight: 700;
      text-transform: uppercase;
      font-size: 0.69rem;
      letter-spacing: 0.05em;
      border-bottom: 1px solid var(--border-color);
    }
    td {
      padding: 10px 12px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.78rem;
      vertical-align: middle;
    }
    tr.table-row-active {
      background: rgba(56, 189, 248, 0.08);
    }
    tr:hover:not(.summary-row) {
      background: rgba(56, 189, 248, 0.04);
      cursor: pointer;
    }
    tr.row-excluded {
      opacity: 0.45;
    }
    td strong { color: var(--accent); }
    .badge-deg {
      display: inline-block;
      font-weight: 800;
      color: var(--accent);
      background: rgba(56, 189, 248, 0.12);
      padding: 2px 8px;
      border-radius: 6px;
      border: 1px solid rgba(56, 189, 248, 0.25);
    }

    /* Summary Row in Table */
    tr.summary-row {
      background: rgba(56, 189, 248, 0.1);
      border-top: 2px solid rgba(56, 189, 248, 0.4);
      font-weight: 700;
    }
    tr.summary-row td {
      color: #f8fafc;
      font-size: 0.84rem;
      padding: 14px 12px;
    }
    tr.summary-row td strong {
      color: #38bdf8;
      font-size: 0.95rem;
    }

    /* Notification Toast */
    .toast {
      position: fixed;
      bottom: 24px;
      right: 24px;
      background: rgba(18, 24, 38, 0.95);
      border: 1px solid var(--accent);
      color: var(--text);
      padding: 12px 20px;
      border-radius: var(--radius-md);
      font-size: 0.82rem;
      font-weight: 600;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
      backdrop-filter: blur(12px);
      z-index: 1000;
      display: none;
      align-items: center;
      gap: 10px;
      animation: fadeIn 0.2s ease-out;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
  </style>
</head>
<body>
  <div class="app-shell">

    <!-- Top Header -->
    <header>
      <div class="brand">
        <div class="brand-icon">
          <svg viewBox="0 0 24 24"><path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"></path></svg>
        </div>
        <div>
          <h1>WCA Studio</h1>
        </div>
        <span class="brand-tag">Multi-Droplet &amp; Asymmetry</span>
      </div>
      <div class="header-actions">
        <button class="btn btn-outline" onclick="copyStatsToClipboard()">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
          Copy Lab Report
        </button>
        <button class="btn btn-outline" onclick="downloadBatchCSV()">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3"/></svg>
          Export CSV (with Stats)
        </button>
        <button class="btn btn-primary" onclick="downloadCurrentCanvas()">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3"/></svg>
          Save Current Image
        </button>
      </div>
    </header>

    <!-- Aggregate Laboratory Statistics Banner (Mean & Error) -->
    <div class="stats-hero-banner">
      <!-- 1. Mean Contact Angle + SEM -->
      <div class="stat-card featured">
        <div class="stat-header">
          <span class="stat-label">Sample Mean Contact Angle (&theta;&#772;)</span>
          <span class="stat-pill pill-cyan" id="statSampleN">N = 0</span>
        </div>
        <div class="stat-val-huge">
          <span id="statMeanWCA">--&deg;</span>
          <span class="error-text" id="statSEM">&plusmn; --&deg;</span>
        </div>
        <div class="stat-sub-details">
          <span>Std Error (SEM): <strong id="statSEMDetail">--&deg;</strong></span>
          <span>95% CI: <strong id="stat95CI">[--, --]</strong></span>
        </div>
      </div>

      <!-- 2. Sample Standard Deviation & Dispersion -->
      <div class="stat-card accent-green">
        <div class="stat-header">
          <span class="stat-label">Dispersion &amp; Variability</span>
          <span class="stat-pill pill-green" id="statRSE">RSE: --%</span>
        </div>
        <div class="stat-val-huge" style="color: #34d399;">
          <span id="statSD">&plusmn; --&deg;</span>
          <span class="error-text" style="color: #6ee7b7; font-size: 0.95rem;">(SD, &sigma;)</span>
        </div>
        <div class="stat-sub-details">
          <span>Range: <strong id="statRange">--&deg; &ndash; --&deg;</strong></span>
          <span>Span: <strong id="statSpan">--&deg;</strong></span>
        </div>
      </div>

      <!-- 3. Current Selected Droplet Angle -->
      <div class="stat-card">
        <div class="stat-header">
          <span class="stat-label">Active Droplet Angle (&theta;)</span>
          <span class="stat-pill" id="statActiveFilename">No Image</span>
        </div>
        <div class="stat-val-huge" style="color: #f8fafc;">
          <span id="valWCA">--&deg;</span>
        </div>
        <div class="stat-sub-details">
          <span>&theta;L: <strong id="valLeft">--&deg;</strong></span>
          <span>&theta;R: <strong id="valRight">--&deg;</strong></span>
          <span>Tilt: <strong id="valTilt">0.0&deg;</strong></span>
        </div>
      </div>

      <!-- 4. Droplet Geometry & Asymmetry -->
      <div class="stat-card">
        <div class="stat-header">
          <span class="stat-label">Droplet Geometry &amp; Asymmetry</span>
          <span class="stat-pill" id="statAsymmetry">Asymmetry: 0.0&deg;</span>
        </div>
        <div class="stat-val-huge" style="font-size: 1.6rem; color: #f8fafc;">
          <span id="valHeight">Height: -- px</span>
        </div>
        <div class="stat-sub-details">
          <span>Base 2b: <strong id="valWidth">-- px</strong></span>
          <span>Apex Shift: <strong id="statApexShiftDisp">0 px</strong></span>
        </div>
      </div>
    </div>

    <!-- Multi-Image Filmstrip / Queue Section -->
    <div class="filmstrip-card">
      <div class="filmstrip-header">
        <div class="filmstrip-title">
          <span>Session Images Queue</span>
          <span class="filmstrip-count-badge" id="queueCount">0 Images</span>
        </div>
        <div class="filmstrip-actions">
          <button class="btn btn-primary" onclick="loadAllTestSamples()">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
            Load All Test Samples (6 Images)
          </button>
          <button class="btn btn-outline" onclick="document.getElementById('fileInputMultiple').click()">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 5v14M5 12h14"/></svg>
            Add Images
          </button>
          <button class="btn btn-outline" style="color: var(--rose);" onclick="clearAllQueue()">
            Clear All
          </button>
        </div>
      </div>

      <!-- Carousel Track -->
      <div class="filmstrip-track" id="filmstripTrack">
        <!-- Thumb items injected here -->
      </div>
    </div>

    <!-- Main Grid: Left Controls + Right Canvas -->
    <div class="layout-grid">

      <!-- Left Controls Sidebar -->
      <aside style="display: flex; flex-direction: column; gap: 16px;">

        <!-- Image Input Card -->
        <div class="card">
          <div class="card-title">Image Input</div>
          <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInputMultiple').click()">
            <svg viewBox="0 0 24 24"><path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM14 13v4h-4v-4H7l5-5 5 5h-3z"/></svg>
            <p>Upload Images (Single or Multiple)</p>
            <span>Click or drag &amp; drop photos here</span>
          </div>
          <input type="file" id="fileInputMultiple" accept="image/*" multiple onchange="handleMultipleFiles(event)">
        </div>

        <!-- Image Rotation, Center Reposition & Crop Card -->
        <div class="card">
          <div class="card-title">
            <span>Orientation &amp; Framing</span>
            <div style="display: flex; gap: 6px;">
              <button class="btn btn-outline" style="padding: 3px 8px; font-size: 0.7rem;" onclick="centerOnDroplet()">
                🎯 Center Droplet
              </button>
              <button class="btn btn-outline" style="padding: 3px 8px; font-size: 0.7rem;" onclick="autoLevelBaseline()">
                ⚡ Auto-Level
              </button>
            </div>
          </div>

          <!-- Rotation Control -->
          <div class="control-row">
            <div class="control-header">
              <span>Image Rotation:</span>
              <span class="control-val" id="valRotation">0.0&deg;</span>
            </div>
            <input type="range" id="rotSlider" min="-100" max="100" value="0" oninput="updateRotation()">
            <div class="btn-nudge-row">
              <button class="btn-nudge" onclick="nudgeRotation(-1.0)">-1.0&deg;</button>
              <button class="btn-nudge" onclick="nudgeRotation(-0.1)">-0.1&deg;</button>
              <button class="btn-nudge" onclick="setRotation(0.0)">Reset</button>
              <button class="btn-nudge" onclick="nudgeRotation(0.1)">+0.1&deg;</button>
              <button class="btn-nudge" onclick="nudgeRotation(1.0)">+1.0&deg;</button>
            </div>
          </div>

          <!-- Reposition Center X (Pan X) -->
          <div class="control-row">
            <div class="control-header">
              <span>Center X Position (Pan):</span>
              <span class="control-val" id="valCenterX">-- px</span>
            </div>
            <input type="range" id="centerXSlider" min="0" max="4032" value="2000" oninput="updateCenterFromSliders()">
            <div class="btn-nudge-row">
              <button class="btn-nudge" onclick="nudgeCenter(-100, 0)">-100px</button>
              <button class="btn-nudge" onclick="nudgeCenter(-20, 0)">-20px</button>
              <button class="btn-nudge" style="flex: 1.5; color: var(--accent);" onclick="centerOnDroplet()">Center Droplet</button>
              <button class="btn-nudge" onclick="nudgeCenter(20, 0)">+20px</button>
              <button class="btn-nudge" onclick="nudgeCenter(100, 0)">+100px</button>
            </div>
          </div>

          <!-- Reposition Center Y (Pan Y) -->
          <div class="control-row">
            <div class="control-header">
              <span>Center Y Position (Pan):</span>
              <span class="control-val" id="valCenterY">-- px</span>
            </div>
            <input type="range" id="centerYSlider" min="0" max="3024" value="1500" oninput="updateCenterFromSliders()">
            <div class="btn-nudge-row">
              <button class="btn-nudge" onclick="nudgeCenter(0, -100)">-100px</button>
              <button class="btn-nudge" onclick="nudgeCenter(0, -20)">-20px</button>
              <button class="btn-nudge" onclick="nudgeCenter(0, 20)">+20px</button>
              <button class="btn-nudge" onclick="nudgeCenter(0, 100)">+100px</button>
            </div>
          </div>

          <!-- Crop Width Slider (Expanded to Full Image Size) -->
          <div class="control-row">
            <div class="control-header">
              <span>Crop Width:</span>
              <span class="control-val" id="valCropW">1500 px</span>
            </div>
            <input type="range" id="cropWSlider" min="400" max="4032" value="1500" oninput="updateCropSize()">
            <div class="btn-nudge-row">
              <button class="btn-nudge" onclick="setCropDimensions(1000, null)">1000px</button>
              <button class="btn-nudge" onclick="setCropDimensions(1600, null)">1600px</button>
              <button class="btn-nudge" onclick="setCropDimensions(2600, null)">2600px</button>
              <button class="btn-nudge" style="color: var(--accent);" onclick="setCropDimensions('full', null)">Full Width</button>
            </div>
          </div>

          <!-- Crop Height Slider (Expanded to Full Image Size) -->
          <div class="control-row">
            <div class="control-header">
              <span>Crop Height:</span>
              <span class="control-val" id="valCropH">900 px</span>
            </div>
            <input type="range" id="cropHSlider" min="300" max="3024" value="900" oninput="updateCropSize()">
            <div class="btn-nudge-row">
              <button class="btn-nudge" onclick="setCropDimensions(null, 600)">600px</button>
              <button class="btn-nudge" onclick="setCropDimensions(null, 1000)">1000px</button>
              <button class="btn-nudge" onclick="setCropDimensions(null, 1600)">1600px</button>
              <button class="btn-nudge" style="color: var(--accent);" onclick="setCropDimensions(null, 'full')">Full Height</button>
            </div>
          </div>
        </div>

        <!-- Droplet Curve & Angle Optimizer Card -->
        <div class="card">
          <div class="card-title">
            <span>Droplet Curve &amp; Asymmetry</span>
            <button class="btn btn-outline" style="padding: 3px 8px; font-size: 0.7rem;" onclick="resetFitToDetected()">
              Reset Fit
            </button>
          </div>

          <!-- Direct Angle Shift Slider (Overall Angle) -->
          <div class="control-row">
            <div class="control-header">
              <span>Droplet Angle Fine-Tune (&plusmn;&theta;):</span>
              <span class="control-val" id="valAngleOffset">0.0&deg;</span>
            </div>
            <input type="range" id="angleOffsetSlider" min="-250" max="250" value="0" oninput="updateAngleOffset()">
            <div class="btn-nudge-row">
              <button class="btn-nudge" onclick="nudgeAngle(-5.0)">-5&deg;</button>
              <button class="btn-nudge" onclick="nudgeAngle(-1.0)">-1&deg;</button>
              <button class="btn-nudge" onclick="nudgeAngle(1.0)">+1&deg;</button>
              <button class="btn-nudge" onclick="nudgeAngle(5.0)">+5&deg;</button>
            </div>
          </div>

          <!-- Apex Lateral Shift Slider (Asymmetry Control) -->
          <div class="control-row">
            <div class="control-header">
              <span>Apex Lateral Shift (&Delta;x, Asymmetry):</span>
              <span class="control-val" id="valApexShift">0 px</span>
            </div>
            <input type="range" id="apexShiftSlider" min="-200" max="200" value="0" oninput="updateApexShift()">
            <div class="btn-nudge-row">
              <button class="btn-nudge" onclick="nudgeApexShift(-20)">-20px</button>
              <button class="btn-nudge" onclick="nudgeApexShift(-5)">-5px</button>
              <button class="btn-nudge" style="flex: 1.4; color: var(--accent);" onclick="setApexShift(0)">Center (0)</button>
              <button class="btn-nudge" onclick="nudgeApexShift(5)">+5px</button>
              <button class="btn-nudge" onclick="nudgeApexShift(20)">+20px</button>
            </div>
          </div>

          <!-- Height / Curvature Shift Slider -->
          <div class="control-row">
            <div class="control-header">
              <span>Apex Height Shift (&Delta;h):</span>
              <span class="control-val" id="valHeightOffset">0 px</span>
            </div>
            <input type="range" id="heightOffsetSlider" min="-80" max="80" value="0" oninput="updateHeightOffset()">
            <div class="btn-nudge-row">
              <button class="btn-nudge" onclick="nudgeHeight(-10)">-10px</button>
              <button class="btn-nudge" onclick="nudgeHeight(-2)">-2px</button>
              <button class="btn-nudge" onclick="nudgeHeight(2)">+2px</button>
              <button class="btn-nudge" onclick="nudgeHeight(10)">+10px</button>
            </div>
          </div>

          <!-- Baseline Meniscus Cutoff -->
          <div class="control-row">
            <div class="control-header">
              <span>Ignore Bottom Meniscus / Reflection:</span>
              <span class="control-val" id="valCutoff">8%</span>
            </div>
            <input type="range" id="cutoffSlider" min="0" max="25" value="8" oninput="recomputeAutoFit()">
          </div>

          <!-- Outlier Rejection Threshold -->
          <div class="control-row">
            <div class="control-header">
              <span>Outlier Rejection:</span>
              <span class="control-val" id="valRansac">3.0 px</span>
            </div>
            <input type="range" id="ransacSlider" min="10" max="80" value="30" oninput="recomputeAutoFit()">
          </div>
        </div>

        <!-- Annotation Styling Card -->
        <div class="card">
          <div class="card-title">Annotation Styling</div>
          <div class="color-grid">
            <div class="color-item">
              <span>Baseline</span>
              <input type="color" id="colBaseline" value="#ef4444" onchange="redrawCanvas()">
            </div>
            <div class="color-item">
              <span>Fit Curve</span>
              <input type="color" id="colFit" value="#38bdf8" onchange="redrawCanvas()">
            </div>
            <div class="color-item">
              <span>Tangents</span>
              <input type="color" id="colTangent" value="#ec4899" onchange="redrawCanvas()">
            </div>
            <div class="color-item">
              <span>Contour</span>
              <input type="color" id="colContour" value="#10b981" onchange="redrawCanvas()">
            </div>
            <div class="color-item">
              <span>Handles</span>
              <input type="color" id="colHandles" value="#f8fafc" onchange="redrawCanvas()">
            </div>
            <div class="color-item">
              <span>Overall Badge</span>
              <input type="color" id="colBadge" value="#ffffff" onchange="redrawCanvas()">
            </div>
            <div class="color-item">
              <span>Left/Right Angles</span>
              <input type="color" id="colSideAngles" value="#38bdf8" onchange="redrawCanvas()">
            </div>
          </div>

          <div class="switch-row" style="margin-top: 10px;">
            <span>Show Droplet Shape Arc</span>
            <label class="toggle-switch">
              <input type="checkbox" id="chkFitCurve" checked onchange="redrawCanvas()">
              <span class="slider-track"></span>
            </label>
          </div>
          <div class="switch-row">
            <span>Show Droplet Shaded Fill</span>
            <label class="toggle-switch">
              <input type="checkbox" id="chkFitFill" checked onchange="redrawCanvas()">
              <span class="slider-track"></span>
            </label>
          </div>
          <div class="switch-row">
            <span>Show Contour Edge Points</span>
            <label class="toggle-switch">
              <input type="checkbox" id="chkContour" checked onchange="redrawCanvas()">
              <span class="slider-track"></span>
            </label>
          </div>
          <div class="switch-row">
            <span>Show Tangent Vectors</span>
            <label class="toggle-switch">
              <input type="checkbox" id="chkTangents" checked onchange="redrawCanvas()">
              <span class="slider-track"></span>
            </label>
          </div>
          <div class="switch-row">
            <span>Show Overall Angle Badge</span>
            <label class="toggle-switch">
              <input type="checkbox" id="chkBadge" checked onchange="redrawCanvas()">
              <span class="slider-track"></span>
            </label>
          </div>
          <div class="switch-row">
            <span>Show Left &amp; Right Angles (&theta;L, &theta;R)</span>
            <label class="toggle-switch">
              <input type="checkbox" id="chkSideAngles" checked onchange="redrawCanvas()">
              <span class="slider-track"></span>
            </label>
          </div>
        </div>

      </aside>

      <!-- Right Main Viewport & Comparison Table -->
      <main style="display: flex; flex-direction: column; gap: 18px;">

        <!-- Main Interactive Viewport -->
        <div class="canvas-viewport" id="viewport">
          <canvas id="dropletCanvas"></canvas>
          <div class="canvas-overlay-hint">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg>
            Drag background to <span>Pan</span> &bull; Drag <span>Apex</span> in 2D (Height &amp; Left/Right Asymmetry) &bull; Drag <span>Baseline</span> to align
          </div>
        </div>

        <!-- Multi-Image Comparison Table Section -->
        <div class="comparison-section">
          <div class="card-title" style="margin-bottom: 0;">
            <span>Individual Droplet Measurements &amp; Statistics</span>
            <div style="display: flex; gap: 8px;">
              <button class="btn btn-outline" style="padding: 4px 10px; font-size: 0.72rem;" onclick="copyStatsToClipboard()">
                Copy Lab Summary
              </button>
              <button class="btn btn-primary" style="padding: 4px 10px; font-size: 0.72rem;" onclick="downloadBatchCSV()">
                Download CSV
              </button>
            </div>
          </div>

          <div class="table-responsive">
            <table id="batchTable">
              <thead>
                <tr>
                  <th style="width: 45px;">Include</th>
                  <th>Image Filename</th>
                  <th>Contact Angle (&theta;)</th>
                  <th>Left (&theta;L)</th>
                  <th>Right (&theta;R)</th>
                  <th>Asymmetry (&Delta;&theta;)</th>
                  <th>Apex Shift</th>
                  <th>Apex Height</th>
                  <th>Base Span (2b)</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody id="batchTableBody">
                <!-- Rows injected dynamically -->
              </tbody>
              <tfoot>
                <!-- Summary row injected dynamically -->
                <tr class="summary-row" id="summaryTableRow">
                  <td><strong>ALL</strong></td>
                  <td><strong id="tblSummaryCount">Mean (N=0)</strong></td>
                  <td><strong id="tblSummaryMean">--&deg; &plusmn; --&deg;</strong></td>
                  <td><span id="tblSummaryLeft">--&deg;</span></td>
                  <td><span id="tblSummaryRight">--&deg;</span></td>
                  <td><span id="tblSummaryAsym">--&deg;</span></td>
                  <td><span id="tblSummaryApexShift">-- px</span></td>
                  <td><span id="tblSummaryHeight">-- px</span></td>
                  <td><span id="tblSummaryWidth">-- px</span></td>
                  <td><strong id="tblSummarySD">SD: &plusmn;--&deg;</strong></td>
                </tr>
              </tfoot>
            </table>
          </div>
        </div>

      </main>

    </div>

  </div>

  <!-- Notification Toast -->
  <div class="toast" id="toast">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
    <span id="toastMsg">Action complete</span>
  </div>

  <script>
    // --- Application State ---
    let sessionImages = []; // List of image records
    let activeImageId = null; // ID of currently displayed/edited image

    let state = {
      image: new Image(),
      imageLoaded: false,
      filename: 'droplet.jpg',
      imgWidth: 4032,
      imgHeight: 3024,
      dropletCenter: { x: 2016, y: 1512 },
      cropCenter: { x: 2016, y: 1512 },
      cropW: 1500,
      cropH: 900,
      rotationDeg: 0.0,
      angleOffsetDeg: 0.0,
      heightOffsetPx: 0.0,
      apexShiftPx: 0.0, // Lateral shift along baseline (Left/Right asymmetry)
      edgePoints: [],
      bottomCutoffPct: 0.08,
      ransacTolPx: 3.0,
      baseAutoFit: null, // { h, b, thetaDeg }
      handles: {
        left: { x: 1700, y: 1600 },
        right: { x: 2300, y: 1600 },
        apex: { x: 2000, y: 1350 }
      },
      currentH: 250,
      currentB: 300,
      currentR: 350,
      wca: 60.0,
      leftWCA: 60.0,
      rightWCA: 60.0,
      activeDrag: null,
      panStartPos: { x: 0, y: 0 },
      panStartCenter: { x: 0, y: 0 }
    };

    const canvas = document.getElementById('dropletCanvas');
    const ctx = canvas.getContext('2d');

    // --- Toast Notifications ---
    function showToast(msg) {
      const toast = document.getElementById('toast');
      document.getElementById('toastMsg').innerText = msg;
      toast.style.display = 'flex';
      setTimeout(() => { toast.style.display = 'none'; }, 2600);
    }

    // --- Drag-and-drop Upload ---
    const dropzone = document.getElementById('dropzone');
    ['dragenter', 'dragover'].forEach(e => dropzone.addEventListener(e, (ev) => { ev.preventDefault(); dropzone.classList.add('dragover'); }));
    ['dragleave', 'drop'].forEach(e => dropzone.addEventListener(e, (ev) => { ev.preventDefault(); dropzone.classList.remove('dragover'); }));
    dropzone.addEventListener('drop', (ev) => {
      if (ev.dataTransfer.files.length > 0) {
        handleFileList(ev.dataTransfer.files);
      }
    });

    function handleMultipleFiles(e) {
      if (e.target.files.length > 0) {
        handleFileList(e.target.files);
      }
    }

    function handleFileList(files) {
      const arr = Array.from(files);
      let loaded = 0;
      arr.forEach(file => {
        const reader = new FileReader();
        reader.onload = function(e) {
          const b64 = e.target.result.split(',')[1];
          uploadAndAddToSession(b64, file.name, () => {
            loaded++;
            if (loaded === arr.length) {
              showToast(`Added ${arr.length} images to session.`);
            }
          });
        };
        reader.readAsDataURL(file);
      });
    }

    // --- Load All Samples from Local Test Directory ---
    function loadAllTestSamples() {
      fetch('/api/list_samples')
        .then(r => r.json())
        .then(data => {
          if (data.success && data.samples && data.samples.length > 0) {
            let processed = 0;
            data.samples.forEach(sampleName => {
              if (sessionImages.some(img => img.filename === sampleName)) {
                processed++;
                if (processed === data.samples.length) updateBatchStatisticsUI();
                return;
              }
              fetch('/api/load_sample?name=' + encodeURIComponent(sampleName))
                .then(r => r.json())
                .then(sampleData => {
                  if (sampleData.success) {
                    uploadAndAddToSession(sampleData.base64, sampleName, () => {
                      processed++;
                      if (processed === data.samples.length) {
                        showToast(`Loaded ${data.samples.length} test samples.`);
                      }
                    });
                  }
                });
            });
          }
        });
    }

    function uploadAndAddToSession(b64, filename, callback) {
      fetch('/api/analyze_raw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: b64, filename: filename })
      })
      .then(r => r.json())
      .then(data => {
        if (data.success) {
          const id = 'img_' + Date.now() + '_' + Math.random().toString(36).substr(2, 6);
          const init = data.initial_fit;
          const leftHandle = { x: data.baseline_handles.left[0], y: data.baseline_handles.left[1] };
          const rightHandle = { x: data.baseline_handles.right[0], y: data.baseline_handles.right[1] };
          const apexHandle = { x: data.apex_handle[0], y: data.apex_handle[1] };

          const record = {
            id: id,
            filename: filename,
            active: true, // included in stats
            full_image: data.full_image,
            thumb_image: data.thumb_image,
            img_width: data.img_width,
            img_height: data.img_height,
            droplet_center: { ...data.droplet_center },
            cropCenter: { ...data.droplet_center },
            cropW: Math.min(data.img_width, 1600),
            cropH: Math.min(data.img_height, 1000),
            edge_points: data.edge_points,
            baseline_handles: { left: leftHandle, right: rightHandle },
            apex_handle: apexHandle,
            handles: { left: { ...leftHandle }, right: { ...rightHandle }, apex: { ...apexHandle } },
            baseAutoFit: {
              h: init.height_px,
              b: init.width_px / 2.0,
              thetaDeg: init.wca_deg
            },
            rotationDeg: 0.0,
            angleOffsetDeg: 0.0,
            heightOffsetPx: 0.0,
            apexShiftPx: 0.0,
            wca: init.wca_deg,
            leftWCA: init.wca_deg,
            rightWCA: init.wca_deg,
            h: init.height_px,
            b: init.width_px / 2.0,
            R: init.R,
            baseline_y: init.baseline_y
          };

          sessionImages.push(record);

          if (!activeImageId) {
            selectImage(record.id);
          } else {
            renderFilmstrip();
            renderTable();
            updateBatchStatisticsUI();
          }

          if (callback) callback();
        } else {
          alert('Analysis error for ' + filename + ': ' + data.error);
        }
      });
    }

    function clearAllQueue() {
      if (confirm('Clear all images from the current session?')) {
        sessionImages = [];
        activeImageId = null;
        state.imageLoaded = false;
        renderFilmstrip();
        renderTable();
        updateBatchStatisticsUI();
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
    }

    // --- Switch Active Image in Canvas ---
    function selectImage(id) {
      const record = sessionImages.find(r => r.id === id);
      if (!record) return;

      activeImageId = id;

      state.filename = record.filename;
      state.imgWidth = record.img_width;
      state.imgHeight = record.img_height;
      state.dropletCenter = { ...record.droplet_center };
      state.cropCenter = record.cropCenter ? { ...record.cropCenter } : { ...record.droplet_center };
      state.cropW = record.cropW || Math.min(record.img_width, 1600);
      state.cropH = record.cropH || Math.min(record.img_height, 1000);
      state.edgePoints = record.edge_points;

      state.handles.left = { ...record.handles.left };
      state.handles.right = { ...record.handles.right };
      state.handles.apex = { ...record.handles.apex };

      state.rotationDeg = record.rotationDeg || 0.0;
      state.angleOffsetDeg = record.angleOffsetDeg || 0.0;
      state.heightOffsetPx = record.heightOffsetPx || 0.0;
      state.apexShiftPx = record.apexShiftPx || 0.0;
      state.baseAutoFit = record.baseAutoFit;

      // Update Center Sliders
      document.getElementById('centerXSlider').max = state.imgWidth;
      document.getElementById('centerXSlider').value = Math.round(state.cropCenter.x);
      document.getElementById('centerYSlider').max = state.imgHeight;
      document.getElementById('centerYSlider').value = Math.round(state.cropCenter.y);
      document.getElementById('valCenterX').innerText = Math.round(state.cropCenter.x) + ' px';
      document.getElementById('valCenterY').innerText = Math.round(state.cropCenter.y) + ' px';

      // Update Crop Sliders
      document.getElementById('cropWSlider').max = state.imgWidth;
      document.getElementById('cropWSlider').value = state.cropW;
      document.getElementById('valCropW').innerText = state.cropW + ' px';

      document.getElementById('cropHSlider').max = state.imgHeight;
      document.getElementById('cropHSlider').value = state.cropH;
      document.getElementById('valCropH').innerText = state.cropH + ' px';

      // Update Rotation and Curve Sliders
      document.getElementById('rotSlider').value = Math.round(state.rotationDeg * 10);
      document.getElementById('valRotation').innerText = state.rotationDeg.toFixed(1) + '°';
      document.getElementById('angleOffsetSlider').value = Math.round(state.angleOffsetDeg * 10);
      document.getElementById('valAngleOffset').innerText = (state.angleOffsetDeg >= 0 ? '+' : '') + state.angleOffsetDeg.toFixed(1) + '°';
      document.getElementById('heightOffsetSlider').value = state.heightOffsetPx;
      document.getElementById('valHeightOffset').innerText = (state.heightOffsetPx >= 0 ? '+' : '') + state.heightOffsetPx + ' px';

      document.getElementById('apexShiftSlider').value = Math.round(state.apexShiftPx);
      document.getElementById('valApexShift').innerText = (state.apexShiftPx >= 0 ? '+' : '') + Math.round(state.apexShiftPx) + ' px';

      document.getElementById('statActiveFilename').innerText = record.filename;

      state.image = new Image();
      state.image.onload = function() {
        state.imageLoaded = true;
        recalculateFromGeometry();
        redrawCanvas();
        renderFilmstrip();
        renderTable();
        updateBatchStatisticsUI();
      };
      state.image.src = record.full_image;
    }

    function toggleImageActive(id, event) {
      if (event) event.stopPropagation();
      const record = sessionImages.find(r => r.id === id);
      if (record) {
        record.active = !record.active;
        renderFilmstrip();
        renderTable();
        updateBatchStatisticsUI();
      }
    }

    function removeImage(id, event) {
      if (event) event.stopPropagation();
      const idx = sessionImages.findIndex(r => r.id === id);
      if (idx !== -1) {
        sessionImages.splice(idx, 1);
        if (activeImageId === id) {
          activeImageId = sessionImages.length > 0 ? sessionImages[0].id : null;
          if (activeImageId) selectImage(activeImageId);
          else {
            state.imageLoaded = false;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
          }
        }
        renderFilmstrip();
        renderTable();
        updateBatchStatisticsUI();
      }
    }

    // --- Statistics Computation (Mean, SD, SEM, 95% CI) ---
    function getStudentTCritical95(df) {
      const tTable = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447,  7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042
      };
      if (df in tTable) return tTable[df];
      if (df > 30) return 1.96 + (2.042 - 1.96) * (30 / df);
      return 12.706;
    }

    function computeStatistics() {
      const activeList = sessionImages.filter(x => x.active && !isNaN(x.wca));
      const N = activeList.length;

      if (N === 0) {
        return {
          N: 0,
          total: sessionImages.length,
          mean: 0,
          sd: 0,
          sem: 0,
          marginOfError: 0,
          ciLower: 0,
          ciUpper: 0,
          min: 0,
          max: 0,
          span: 0,
          rse: 0,
          meanLeft: 0,
          meanRight: 0,
          meanAsym: 0,
          meanApexShift: 0,
          meanH: 0,
          meanB: 0,
          meanR: 0
        };
      }

      if (N === 1) {
        const val = activeList[0].wca;
        const leftVal = activeList[0].leftWCA || val;
        const rightVal = activeList[0].rightWCA || val;
        return {
          N: 1,
          total: sessionImages.length,
          mean: val,
          sd: 0,
          sem: 0,
          marginOfError: 0,
          ciLower: val,
          ciUpper: val,
          min: val,
          max: val,
          span: 0,
          rse: 0,
          meanLeft: leftVal,
          meanRight: rightVal,
          meanAsym: Math.abs(leftVal - rightVal),
          meanApexShift: activeList[0].apexShiftPx || 0,
          meanH: activeList[0].h || 0,
          meanB: (activeList[0].b || 0) * 2,
          meanR: activeList[0].R || 0
        };
      }

      const values = activeList.map(x => x.wca);
      const sum = values.reduce((a, b) => a + b, 0);
      const mean = sum / N;

      const sumSqDiff = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0);
      const variance = sumSqDiff / (N - 1);
      const sd = Math.sqrt(variance);
      const sem = sd / Math.sqrt(N);

      const tCrit = getStudentTCritical95(N - 1);
      const marginOfError = tCrit * sem;
      const ciLower = mean - marginOfError;
      const ciUpper = mean + marginOfError;

      const min = Math.min(...values);
      const max = Math.max(...values);
      const span = max - min;
      const rse = mean > 0 ? (sem / mean) * 100 : 0;

      const meanLeft = activeList.map(x => x.leftWCA || x.wca).reduce((a, b) => a + b, 0) / N;
      const meanRight = activeList.map(x => x.rightWCA || x.wca).reduce((a, b) => a + b, 0) / N;
      const meanAsym = activeList.map(x => Math.abs((x.leftWCA || x.wca) - (x.rightWCA || x.wca))).reduce((a, b) => a + b, 0) / N;
      const meanApexShift = activeList.map(x => x.apexShiftPx || 0).reduce((a, b) => a + b, 0) / N;
      const meanH = activeList.map(x => x.h || 0).reduce((a, b) => a + b, 0) / N;
      const meanB = activeList.map(x => (x.b || 0) * 2).reduce((a, b) => a + b, 0) / N;
      const meanR = activeList.map(x => x.R || 0).reduce((a, b) => a + b, 0) / N;

      return {
        N,
        total: sessionImages.length,
        mean,
        sd,
        sem,
        marginOfError,
        ciLower,
        ciUpper,
        min,
        max,
        span,
        rse,
        meanLeft,
        meanRight,
        meanAsym,
        meanApexShift,
        meanH,
        meanB,
        meanR
      };
    }

    // --- Update Statistics UI Dashboard ---
    function updateBatchStatisticsUI() {
      const stats = computeStatistics();

      document.getElementById('statSampleN').innerText = `N = ${stats.N} / ${stats.total}`;
      document.getElementById('queueCount').innerText = `${sessionImages.length} Images`;

      if (stats.N > 0) {
        document.getElementById('statMeanWCA').innerText = stats.mean.toFixed(2) + '°';
        document.getElementById('statSEM').innerText = `± ${stats.sem.toFixed(2)}°`;
        document.getElementById('statSEMDetail').innerText = `± ${stats.sem.toFixed(2)}°`;
        document.getElementById('stat95CI').innerText = `[${stats.ciLower.toFixed(2)}°, ${stats.ciUpper.toFixed(2)}°]`;

        document.getElementById('statSD').innerText = `± ${stats.sd.toFixed(2)}°`;
        document.getElementById('statRSE').innerText = `RSE: ${stats.rse.toFixed(1)}%`;
        document.getElementById('statRange').innerText = `${stats.min.toFixed(1)}° – ${stats.max.toFixed(1)}°`;
        document.getElementById('statSpan').innerText = `Δ ${stats.span.toFixed(1)}°`;

        document.getElementById('statAsymmetry').innerText = `Mean Asym: Δθ = ${stats.meanAsym.toFixed(1)}°`;
        document.getElementById('statApexShiftDisp').innerText = `${stats.meanApexShift >= 0 ? '+' : ''}${stats.meanApexShift.toFixed(1)} px`;
      } else {
        document.getElementById('statMeanWCA').innerText = '--°';
        document.getElementById('statSEM').innerText = '± --°';
        document.getElementById('statSEMDetail').innerText = '--°';
        document.getElementById('stat95CI').innerText = '[--, --]';
        document.getElementById('statSD').innerText = '± --°';
        document.getElementById('statRSE').innerText = 'RSE: --%';
        document.getElementById('statRange').innerText = '--° – --°';
        document.getElementById('statSpan').innerText = '--°';
        document.getElementById('statAsymmetry').innerText = 'Asymmetry: 0.0°';
        document.getElementById('statApexShiftDisp').innerText = '0 px';
      }

      document.getElementById('tblSummaryCount').innerText = `Mean (N=${stats.N})`;
      if (stats.N > 0) {
        document.getElementById('tblSummaryMean').innerHTML = `<strong>${stats.mean.toFixed(2)}°</strong> <span style="font-size:0.75rem; color:#7dd3fc;">± ${stats.sem.toFixed(2)}° (SEM)</span>`;
        document.getElementById('tblSummaryLeft').innerText = stats.meanLeft.toFixed(2) + '°';
        document.getElementById('tblSummaryRight').innerText = stats.meanRight.toFixed(2) + '°';
        document.getElementById('tblSummaryAsym').innerText = stats.meanAsym.toFixed(2) + '°';
        document.getElementById('tblSummaryApexShift').innerText = (stats.meanApexShift >= 0 ? '+' : '') + stats.meanApexShift.toFixed(1) + ' px';
        document.getElementById('tblSummaryHeight').innerText = stats.meanH.toFixed(1) + ' px';
        document.getElementById('tblSummaryWidth').innerText = stats.meanB.toFixed(1) + ' px';
        document.getElementById('tblSummarySD').innerText = `SD: ±${stats.sd.toFixed(2)}°`;
      } else {
        document.getElementById('tblSummaryMean').innerText = '--°';
        document.getElementById('tblSummaryLeft').innerText = '--°';
        document.getElementById('tblSummaryRight').innerText = '--°';
        document.getElementById('tblSummaryAsym').innerText = '--°';
        document.getElementById('tblSummaryApexShift').innerText = '-- px';
        document.getElementById('tblSummaryHeight').innerText = '-- px';
        document.getElementById('tblSummaryWidth').innerText = '-- px';
        document.getElementById('tblSummarySD').innerText = 'SD: ±--°';
      }
    }

    // --- Render Filmstrip Carousel ---
    function renderFilmstrip() {
      const track = document.getElementById('filmstripTrack');
      track.innerHTML = '';

      if (sessionImages.length === 0) {
        track.innerHTML = '<div style="color: var(--text-dim); font-size: 0.8rem; padding: 12px;">No images loaded yet. Click "+ Add Images" or "Load All Test Samples".</div>';
        return;
      }

      sessionImages.forEach(img => {
        const item = document.createElement('div');
        item.className = 'thumb-item' + (img.id === activeImageId ? ' active' : '') + (!img.active ? ' excluded' : '');
        item.setAttribute('data-id', img.id);
        item.onclick = () => selectImage(img.id);

        item.innerHTML = `
          <div class="thumb-toggle-include" onclick="toggleImageActive('${img.id}', event)" title="Toggle inclusion in statistics">
            <input type="checkbox" ${img.active ? 'checked' : ''} style="cursor: pointer;">
            <span>${img.active ? 'Incl' : 'Excl'}</span>
          </div>
          <button class="thumb-remove-btn" onclick="removeImage('${img.id}', event)" title="Remove from queue">&times;</button>
          <div class="thumb-img-wrapper">
            <img src="${img.thumb_image}" alt="${img.filename}">
          </div>
          <div class="thumb-footer">
            <span class="thumb-name" title="${img.filename}">${img.filename}</span>
            <span class="thumb-angle-badge">${img.wca ? img.wca.toFixed(1) + '°' : '--'}</span>
          </div>
        `;
        track.appendChild(item);
      });
    }

    // --- Render Comparison Table ---
    function renderTable() {
      const tbody = document.getElementById('batchTableBody');
      tbody.innerHTML = '';

      sessionImages.forEach(img => {
        const tr = document.createElement('tr');
        if (img.id === activeImageId) tr.classList.add('table-row-active');
        if (!img.active) tr.classList.add('row-excluded');
        tr.setAttribute('data-id', img.id);

        tr.onclick = () => selectImage(img.id);

        const asym = img.leftWCA && img.rightWCA ? Math.abs(img.leftWCA - img.rightWCA).toFixed(1) + '°' : '--';
        const shift = img.apexShiftPx !== undefined ? (img.apexShiftPx >= 0 ? '+' : '') + Math.round(img.apexShiftPx) + ' px' : '0 px';

        tr.innerHTML = `
          <td style="text-align: center;" onclick="event.stopPropagation()">
            <input type="checkbox" ${img.active ? 'checked' : ''} onchange="toggleImageActive('${img.id}')" style="cursor: pointer; width: 15px; height: 15px;">
          </td>
          <td><strong>${img.filename}</strong></td>
          <td><span class="badge-deg">${img.wca ? img.wca.toFixed(1) + '°' : '--'}</span></td>
          <td>${img.leftWCA ? img.leftWCA.toFixed(1) + '°' : '--'}</td>
          <td>${img.rightWCA ? img.rightWCA.toFixed(1) + '°' : '--'}</td>
          <td>${asym}</td>
          <td>${shift}</td>
          <td>${img.h ? img.h.toFixed(1) + ' px' : '--'}</td>
          <td>${img.b ? (2 * img.b).toFixed(1) + ' px' : '--'}</td>
          <td onclick="event.stopPropagation()">
            <button class="btn btn-outline" style="padding: 2px 7px; font-size: 0.68rem;" onclick="selectImage('${img.id}')">Edit</button>
            <button class="btn btn-danger" style="padding: 2px 7px; font-size: 0.68rem;" onclick="removeImage('${img.id}', event)">&times;</button>
          </td>
        `;
        tbody.appendChild(tr);
      });
    }

    function updateActiveRowAndThumb() {
      if (!activeImageId) return;
      const record = sessionImages.find(r => r.id === activeImageId);
      if (!record) return;

      const activeThumb = document.querySelector(`.thumb-item[data-id="${activeImageId}"] .thumb-angle-badge`);
      if (activeThumb) activeThumb.innerText = record.wca.toFixed(1) + '°';

      const activeTr = document.querySelector(`tr[data-id="${activeImageId}"]`);
      if (activeTr) {
        const badge = activeTr.querySelector('.badge-deg');
        if (badge) badge.innerText = record.wca.toFixed(1) + '°';
        const cells = activeTr.querySelectorAll('td');
        if (cells.length >= 9) {
          cells[3].innerText = record.leftWCA.toFixed(1) + '°';
          cells[4].innerText = record.rightWCA.toFixed(1) + '°';
          cells[5].innerText = Math.abs(record.leftWCA - record.rightWCA).toFixed(1) + '°';
          cells[6].innerText = (record.apexShiftPx >= 0 ? '+' : '') + Math.round(record.apexShiftPx) + ' px';
          cells[7].innerText = record.h.toFixed(1) + ' px';
          cells[8].innerText = (2 * record.b).toFixed(1) + ' px';
        }
      }
    }

    // --- Core Canvas Geometry & Active Image Sync (Asymmetric Bi-Arc) ---
    function recalculateFromGeometry() {
      if (!state.baseAutoFit) return;

      const p1 = state.handles.left;
      const p2 = state.handles.right;

      const baseSpan = Math.hypot(p2.x - p1.x, p2.y - p1.y);
      const bHalf = baseSpan / 2.0;
      state.currentB = bHalf;

      const mx = (p1.x + p2.x) / 2.0;
      const my = (p1.y + p2.y) / 2.0;
      const ux = (p2.x - p1.x) / baseSpan;
      const uy = (p2.y - p1.y) / baseSpan;
      let nx = uy, ny = -ux;
      if (ny > 0) { nx = -nx; ny = -ny; }

      // Clamp lateral shift along baseline to within 44% of baseSpan
      const maxShift = baseSpan * 0.44;
      state.apexShiftPx = Math.max(-maxShift, Math.min(maxShift, state.apexShiftPx || 0.0));
      const shift = state.apexShiftPx;

      const bL = Math.max(15.0, bHalf + shift);
      const bR = Math.max(15.0, bHalf - shift);

      let targetTheta = state.baseAutoFit.thetaDeg + state.angleOffsetDeg;
      targetTheta = Math.max(5.0, Math.min(175.0, targetTheta));

      let h = bHalf * Math.tan((targetTheta * Math.PI) / 360.0) + state.heightOffsetPx;
      h = Math.max(10.0, h);
      state.currentH = h;

      // Contact angles for Left and Right sides
      const thetaL = 2 * Math.atan(h / bL) * (180 / Math.PI);
      const thetaR = 2 * Math.atan(h / bR) * (180 / Math.PI);
      const effectiveWCA = (thetaL + thetaR) / 2.0;

      state.wca = effectiveWCA;
      state.leftWCA = thetaL;
      state.rightWCA = thetaR;

      // Curvature radii for left and right circular arcs
      const RL = (bL * bL + h * h) / (2.0 * h);
      const RR = (bR * bR + h * h) / (2.0 * h);
      state.currentR = (RL + RR) / 2.0;

      // Baseline projection point P of apex:
      const Px = mx + shift * ux;
      const Py = my + shift * uy;

      // True Apex Handle Position in original image coordinates
      state.handles.apex = {
        x: Px + h * nx,
        y: Py + h * ny
      };

      state.asymFit = {
        Px, Py,
        ux, uy,
        nx, ny,
        bL, bR,
        h,
        RL, RR,
        thetaL, thetaR
      };

      const tiltDeg = Math.atan2(p2.y - p1.y, p2.x - p1.x) * (180 / Math.PI);
      const asymDeg = Math.abs(thetaL - thetaR);

      // Update Active Displays
      document.getElementById('valWCA').innerText = effectiveWCA.toFixed(1) + '°';
      document.getElementById('valLeft').innerText = thetaL.toFixed(1) + '°';
      document.getElementById('valRight').innerText = thetaR.toFixed(1) + '°';
      document.getElementById('valTilt').innerText = tiltDeg.toFixed(1) + '°';
      document.getElementById('valHeight').innerText = `Height: ${h.toFixed(1)} px`;
      document.getElementById('valWidth').innerText = (2 * bHalf).toFixed(1) + ' px';
      document.getElementById('statAsymmetry').innerText = `Asym: Δθ = ${asymDeg.toFixed(1)}°`;
      document.getElementById('statApexShiftDisp').innerText = `${shift >= 0 ? '+' : ''}${shift.toFixed(1)} px`;

      // Sync changes back to active image record in session
      if (activeImageId) {
        const record = sessionImages.find(r => r.id === activeImageId);
        if (record) {
          record.wca = effectiveWCA;
          record.leftWCA = thetaL;
          record.rightWCA = thetaR;
          record.h = h;
          record.b = bHalf;
          record.R = (RL + RR) / 2.0;
          record.handles = { left: { ...p1 }, right: { ...p2 }, apex: { ...state.handles.apex } };
          record.rotationDeg = state.rotationDeg;
          record.angleOffsetDeg = state.angleOffsetDeg;
          record.heightOffsetPx = state.heightOffsetPx;
          record.apexShiftPx = state.apexShiftPx;
          record.cropCenter = { ...state.cropCenter };
          record.cropW = state.cropW;
          record.cropH = state.cropH;

          updateActiveRowAndThumb();
          updateBatchStatisticsUI();
        }
      }
    }

    // Circle fitting algorithm
    function fitCirclePoints(pts) {
      const N = pts.length;
      if (N < 3) return null;
      let Sx = 0, Sy = 0, Sx2 = 0, Sy2 = 0, Sxy = 0, Sx3 = 0, Sy3 = 0, Sxy2 = 0, Sx2y = 0;
      for (let i = 0; i < N; i++) {
        const x = pts[i][0], y = pts[i][1];
        Sx += x; Sy += y;
        Sx2 += x * x; Sy2 += y * y; Sxy += x * y;
        Sx3 += x * x * x; Sy3 += y * y * y;
        Sxy2 += x * y * y; Sx2y += x * x * y;
      }
      const a1 = Sx2 - (Sx * Sx) / N;
      const b1 = Sxy - (Sx * Sy) / N;
      const c1 = 0.5 * ((Sx3 + Sxy2) - (Sx * (Sx2 + Sy2)) / N);
      const a2 = Sxy - (Sx * Sy) / N;
      const b2 = Sy2 - (Sy * Sy) / N;
      const c2 = 0.5 * ((Sx2y + Sy3) - (Sy * (Sx2 + Sy2)) / N);
      const det = a1 * b2 - a2 * b1;
      if (Math.abs(det) < 1e-6) return null;
      const xc = (c1 * b2 - c2 * b1) / det;
      const yc = (a1 * c2 - a2 * c1) / det;
      const R = Math.sqrt(Math.max(1, (Sx2 - 2 * xc * Sx + N * xc * xc + Sy2 - 2 * yc * Sy + N * yc * yc) / N));
      return { xc, yc, R };
    }

    function recomputeAutoFit() {
      if (!state.edgePoints || state.edgePoints.length < 15) return;

      state.bottomCutoffPct = parseInt(document.getElementById('cutoffSlider').value) / 100.0;
      state.ransacTolPx = parseInt(document.getElementById('ransacSlider').value) / 10.0;
      document.getElementById('valCutoff').innerText = Math.round(state.bottomCutoffPct * 100) + '%';
      document.getElementById('valRansac').innerText = state.ransacTolPx.toFixed(1) + ' px';

      const p1 = state.handles.left;
      const p2 = state.handles.right;
      const A = p2.y - p1.y;
      const B = p1.x - p2.x;
      const C = p2.x * p1.y - p1.x * p2.y;
      const norm = Math.hypot(A, B);

      let maxDist = -Infinity;
      for (let i = 0; i < state.edgePoints.length; i++) {
        const d = (A * state.edgePoints[i][0] + B * state.edgePoints[i][1] + C) / norm;
        if (d > maxDist) maxDist = d;
      }

      const cutoffDist = state.bottomCutoffPct * maxDist;
      let pts = [];
      for (let i = 0; i < state.edgePoints.length; i++) {
        const d = (A * state.edgePoints[i][0] + B * state.edgePoints[i][1] + C) / norm;
        if (d >= cutoffDist) pts.push(state.edgePoints[i]);
      }
      if (pts.length < 15) pts = state.edgePoints;

      const fit = fitCirclePoints(pts);
      if (!fit) return;

      const distCenter = (A * fit.xc + B * fit.yc + C) / norm;
      const absDist = Math.abs(distCenter);
      const bHalf = Math.sqrt(Math.max(10, fit.R * fit.R - absDist * absDist));
      const autoH = fit.R - absDist;

      state.baseAutoFit = {
        h: autoH,
        b: bHalf,
        thetaDeg: 2 * Math.atan(autoH / bHalf) * (180 / Math.PI)
      };

      recalculateFromGeometry();
      redrawCanvas();
    }

    // Sliders
    function resetSliders() {
      state.angleOffsetDeg = 0.0;
      state.heightOffsetPx = 0.0;
      state.apexShiftPx = 0.0;
      document.getElementById('angleOffsetSlider').value = 0;
      document.getElementById('heightOffsetSlider').value = 0;
      document.getElementById('apexShiftSlider').value = 0;
      document.getElementById('valAngleOffset').innerText = '0.0°';
      document.getElementById('valHeightOffset').innerText = '0 px';
      document.getElementById('valApexShift').innerText = '0 px';
    }

    function resetFitToDetected() {
      resetSliders();
      recomputeAutoFit();
    }

    function updateRotation() {
      const val = parseInt(document.getElementById('rotSlider').value) / 10.0;
      setRotation(val);
    }

    function nudgeRotation(delta) {
      const current = parseFloat(state.rotationDeg);
      const next = Math.max(-10.0, Math.min(10.0, Math.round((current + delta) * 10) / 10));
      document.getElementById('rotSlider').value = Math.round(next * 10);
      setRotation(next);
    }

    function setRotation(valDeg) {
      state.rotationDeg = valDeg;
      document.getElementById('valRotation').innerText = valDeg.toFixed(1) + '°';
      recalculateFromGeometry();
      redrawCanvas();
    }

    function autoLevelBaseline() {
      const dx = state.handles.right.x - state.handles.left.x;
      const dy = state.handles.right.y - state.handles.left.y;
      const tilt = Math.atan2(dy, dx) * (180 / Math.PI);
      const targetRot = Math.max(-10.0, Math.min(10.0, Math.round(-tilt * 10) / 10));
      document.getElementById('rotSlider').value = Math.round(targetRot * 10);
      setRotation(targetRot);
    }

    // --- Center Repositioning & Framing Controls ---
    function updateCenterFromSliders() {
      state.cropCenter.x = parseInt(document.getElementById('centerXSlider').value);
      state.cropCenter.y = parseInt(document.getElementById('centerYSlider').value);
      updateCenterDisplay();
      redrawCanvas();
    }

    function updateCenterDisplay() {
      document.getElementById('centerXSlider').value = Math.round(state.cropCenter.x);
      document.getElementById('centerYSlider').value = Math.round(state.cropCenter.y);
      document.getElementById('valCenterX').innerText = Math.round(state.cropCenter.x) + ' px';
      document.getElementById('valCenterY').innerText = Math.round(state.cropCenter.y) + ' px';
      if (activeImageId) {
        const record = sessionImages.find(r => r.id === activeImageId);
        if (record) record.cropCenter = { ...state.cropCenter };
      }
    }

    function centerOnDroplet() {
      if (state.dropletCenter) {
        state.cropCenter.x = state.dropletCenter.x;
        state.cropCenter.y = state.dropletCenter.y;
        updateCenterDisplay();
        redrawCanvas();
        showToast('Centered on droplet.');
      }
    }

    function nudgeCenter(dx, dy) {
      state.cropCenter.x = Math.max(0, Math.min(state.imgWidth, state.cropCenter.x + dx));
      state.cropCenter.y = Math.max(0, Math.min(state.imgHeight, state.cropCenter.y + dy));
      updateCenterDisplay();
      redrawCanvas();
    }

    function updateCropSize() {
      state.cropW = parseInt(document.getElementById('cropWSlider').value);
      state.cropH = parseInt(document.getElementById('cropHSlider').value);
      document.getElementById('valCropW').innerText = state.cropW + ' px';
      document.getElementById('valCropH').innerText = state.cropH + ' px';
      recalculateFromGeometry();
      redrawCanvas();
    }

    function setCropDimensions(w, h) {
      if (w === 'full') state.cropW = state.imgWidth;
      else if (w !== null) state.cropW = Math.min(state.imgWidth, w);

      if (h === 'full') state.cropH = state.imgHeight;
      else if (h !== null) state.cropH = Math.min(state.imgHeight, h);

      document.getElementById('cropWSlider').value = state.cropW;
      document.getElementById('cropHSlider').value = state.cropH;
      document.getElementById('valCropW').innerText = state.cropW + ' px';
      document.getElementById('valCropH').innerText = state.cropH + ' px';
      redrawCanvas();
    }

    function updateAngleOffset() {
      const val = parseInt(document.getElementById('angleOffsetSlider').value) / 10.0;
      state.angleOffsetDeg = val;
      document.getElementById('valAngleOffset').innerText = (val >= 0 ? '+' : '') + val.toFixed(1) + '°';
      recalculateFromGeometry();
      redrawCanvas();
    }

    function nudgeAngle(delta) {
      const next = Math.max(-25.0, Math.min(25.0, state.angleOffsetDeg + delta));
      document.getElementById('angleOffsetSlider').value = Math.round(next * 10);
      state.angleOffsetDeg = next;
      document.getElementById('valAngleOffset').innerText = (next >= 0 ? '+' : '') + next.toFixed(1) + '°';
      recalculateFromGeometry();
      redrawCanvas();
    }

    function updateApexShift() {
      const val = parseInt(document.getElementById('apexShiftSlider').value);
      state.apexShiftPx = val;
      document.getElementById('valApexShift').innerText = (val >= 0 ? '+' : '') + val + ' px';
      recalculateFromGeometry();
      redrawCanvas();
    }

    function nudgeApexShift(delta) {
      const current = state.apexShiftPx || 0;
      const p1 = state.handles.left;
      const p2 = state.handles.right;
      const baseSpan = Math.hypot(p2.x - p1.x, p2.y - p1.y);
      const maxShift = baseSpan * 0.44;
      const next = Math.max(-maxShift, Math.min(maxShift, current + delta));
      state.apexShiftPx = next;
      document.getElementById('apexShiftSlider').value = Math.round(next);
      document.getElementById('valApexShift').innerText = (next >= 0 ? '+' : '') + Math.round(next) + ' px';
      recalculateFromGeometry();
      redrawCanvas();
    }

    function setApexShift(val) {
      state.apexShiftPx = val;
      document.getElementById('apexShiftSlider').value = val;
      document.getElementById('valApexShift').innerText = val + ' px';
      recalculateFromGeometry();
      redrawCanvas();
    }

    function updateHeightOffset() {
      const val = parseInt(document.getElementById('heightOffsetSlider').value);
      state.heightOffsetPx = val;
      document.getElementById('valHeightOffset').innerText = (val >= 0 ? '+' : '') + val + ' px';
      recalculateFromGeometry();
      redrawCanvas();
    }

    function nudgeHeight(delta) {
      const next = Math.max(-80, Math.min(80, state.heightOffsetPx + delta));
      document.getElementById('heightOffsetSlider').value = next;
      state.heightOffsetPx = next;
      document.getElementById('valHeightOffset').innerText = (next >= 0 ? '+' : '') + next + ' px';
      recalculateFromGeometry();
      redrawCanvas();
    }

    // --- Draw Canvas ---
    function redrawCanvas() {
      if (!state.imageLoaded) return;

      const viewW = state.cropW;
      const viewH = state.cropH;
      canvas.width = viewW;
      canvas.height = viewH;
      ctx.clearRect(0, 0, viewW, viewH);

      const ox = viewW / 2 - state.cropCenter.x;
      const oy = viewH / 2 - state.cropCenter.y;
      const rotRad = (state.rotationDeg * Math.PI) / 180.0;

      // 1. Draw Rotated Source Photo
      ctx.save();
      ctx.translate(viewW / 2, viewH / 2);
      ctx.rotate(rotRad);
      ctx.translate(-viewW / 2, -viewH / 2);
      ctx.drawImage(state.image, ox, oy);
      ctx.restore();

      const colBaseline = document.getElementById('colBaseline').value;
      const colFit = document.getElementById('colFit').value;
      const colTangent = document.getElementById('colTangent').value;
      const colContour = document.getElementById('colContour').value;
      const colHandles = document.getElementById('colHandles').value;
      const colBadge = document.getElementById('colBadge').value;
      const colSideAngles = document.getElementById('colSideAngles') ? document.getElementById('colSideAngles').value : '#38bdf8';

      const showContour = document.getElementById('chkContour').checked;
      const showTangents = document.getElementById('chkTangents').checked;
      const showBadge = document.getElementById('chkBadge').checked;
      const showSideAngles = document.getElementById('chkSideAngles') ? document.getElementById('chkSideAngles').checked : true;
      const showFitCurve = document.getElementById('chkFitCurve') ? document.getElementById('chkFitCurve').checked : true;
      const showFitFill = document.getElementById('chkFitFill') ? document.getElementById('chkFitFill').checked : true;

      // Helper to transform coordinate from photo space into rotated canvas space
      function toScreen(px, py) {
        const x = px + ox;
        const y = py + oy;
        const rx = x - viewW / 2;
        const ry = y - viewH / 2;
        return {
          x: viewW / 2 + (rx * Math.cos(rotRad) - ry * Math.sin(rotRad)),
          y: viewH / 2 + (rx * Math.sin(rotRad) + ry * Math.cos(rotRad))
        };
      }

      // 2. Draw Detected Droplet Contour Edge Points
      if (showContour && state.edgePoints) {
        ctx.fillStyle = colContour;
        for (let i = 0; i < state.edgePoints.length; i++) {
          const pt = toScreen(state.edgePoints[i][0], state.edgePoints[i][1]);
          ctx.beginPath();
          ctx.arc(pt.x, pt.y, 2.0, 0, 2 * Math.PI);
          ctx.fill();
        }
      }

      // 3. Draw Substrate Baseline Line
      const p1 = toScreen(state.handles.left.x, state.handles.left.y);
      const p2 = toScreen(state.handles.right.x, state.handles.right.y);

      ctx.save();
      ctx.strokeStyle = colBaseline;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      const bdx = p2.x - p1.x;
      const bdy = p2.y - p1.y;
      ctx.moveTo(p1.x - bdx * 0.4, p1.y - bdy * 0.4);
      ctx.lineTo(p2.x + bdx * 0.4, p2.y + bdy * 0.4);
      ctx.stroke();
      ctx.restore();

      // 4. Draw Modeled Droplet Curve (Asymmetric Bi-Arc)
      if (state.asymFit && showFitCurve) {
        const { Px, Py, ux, uy, nx, ny, bL, bR, h, RL, RR, thetaL, thetaR } = state.asymFit;

        const numSteps = 80;
        const arcPoints = [];

        for (let i = 0; i <= numSteps; i++) {
          const t = i / numSteps;
          const s = -bL + t * (bL + bR);
          let yRel = 0;
          if (s <= 0) {
            yRel = (h - RL) + Math.sqrt(Math.max(0, RL * RL - s * s));
          } else {
            yRel = (h - RR) + Math.sqrt(Math.max(0, RR * RR - s * s));
          }
          const ptX = Px + s * ux + yRel * nx;
          const ptY = Py + s * uy + yRel * ny;
          arcPoints.push(toScreen(ptX, ptY));
        }

        // Draw Shaded Droplet Interior (if enabled)
        if (showFitFill && arcPoints.length > 0) {
          ctx.save();
          ctx.fillStyle = colFit + '25'; // subtle ~15% tint
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          for (let i = 0; i < arcPoints.length; i++) {
            ctx.lineTo(arcPoints[i].x, arcPoints[i].y);
          }
          ctx.lineTo(p2.x, p2.y);
          ctx.closePath();
          ctx.fill();
          ctx.restore();
        }

        // Draw Prominent Droplet Arc Line
        ctx.save();
        ctx.strokeStyle = colFit;
        ctx.lineWidth = 3.5;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.shadowColor = colFit;
        ctx.shadowBlur = 8;
        ctx.beginPath();
        for (let i = 0; i < arcPoints.length; i++) {
          if (i === 0) ctx.moveTo(arcPoints[i].x, arcPoints[i].y);
          else ctx.lineTo(arcPoints[i].x, arcPoints[i].y);
        }
        ctx.stroke();
        ctx.restore();

        // 5. Compute Tangent Lines (Left and Right with true thetaL and thetaR)
        const tLen = 140;
        const radL = (thetaL * Math.PI) / 180.0;
        const tgtLx = Math.cos(radL) * ux + Math.sin(radL) * nx;
        const tgtLy = Math.cos(radL) * uy + Math.sin(radL) * ny;
        const lTgtEnd = toScreen(state.handles.left.x + tgtLx * tLen, state.handles.left.y + tgtLy * tLen);

        const radR = (thetaR * Math.PI) / 180.0;
        const tgtRx = -Math.cos(radR) * ux + Math.sin(radR) * nx;
        const tgtRy = -Math.cos(radR) * uy + Math.sin(radR) * ny;
        const rTgtEnd = toScreen(state.handles.right.x + tgtRx * tLen, state.handles.right.y + tgtRy * tLen);

        // Draw Tangent Lines (if enabled)
        if (showTangents) {
          ctx.save();
          ctx.strokeStyle = colTangent;
          ctx.lineWidth = 2.5;

          // Left Tangent at p1 (A)
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(lTgtEnd.x, lTgtEnd.y);
          ctx.stroke();

          // Right Tangent at p2 (B)
          ctx.beginPath();
          ctx.moveTo(p2.x, p2.y);
          ctx.lineTo(rTgtEnd.x, rTgtEnd.y);
          ctx.stroke();
          ctx.restore();
        }

        // 5b. Draw Left & Right Angles on Screen (Text directly above the lines, no boxes)
        if (showSideAngles) {
          ctx.save();
          ctx.font = 'bold 14px "JetBrains Mono", monospace';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';

          // --- Left Side Angle Text (slightly above the left tangent line) ---
          const tPosL = 0.45;
          const posLx = p1.x + (lTgtEnd.x - p1.x) * tPosL;
          const posLy = p1.y + (lTgtEnd.y - p1.y) * tPosL;
          const ldx = lTgtEnd.x - p1.x;
          const ldy = lTgtEnd.y - p1.y;
          const lenL = Math.hypot(ldx, ldy) || 1;
          let upLx = -ldy / lenL;
          let upLy = ldx / lenL;
          if (upLy > 0) { upLx = -upLx; upLy = -upLy; }
          const textLx = posLx + upLx * 15;
          const textLy = posLy + upLy * 15;

          const textL = `${thetaL.toFixed(1)}\u00B0`;
          ctx.strokeStyle = '#0a0e17';
          ctx.lineWidth = 3.5;
          ctx.lineJoin = 'round';
          ctx.strokeText(textL, textLx, textLy);
          ctx.fillStyle = colSideAngles;
          ctx.fillText(textL, textLx, textLy);

          // --- Right Side Angle Text (slightly above the right tangent line) ---
          const tPosR = 0.45;
          const posRx = p2.x + (rTgtEnd.x - p2.x) * tPosR;
          const posRy = p2.y + (rTgtEnd.y - p2.y) * tPosR;
          const rdx = rTgtEnd.x - p2.x;
          const rdy = rTgtEnd.y - p2.y;
          const lenR = Math.hypot(rdx, rdy) || 1;
          let upRx = -rdy / lenR;
          let upRy = rdx / lenR;
          if (upRy > 0) { upRx = -upRx; upRy = -upRy; }
          const textRx = posRx + upRx * 15;
          const textRy = posRy + upRy * 15;

          const textR = `${thetaR.toFixed(1)}\u00B0`;
          ctx.strokeStyle = '#0a0e17';
          ctx.lineWidth = 3.5;
          ctx.lineJoin = 'round';
          ctx.strokeText(textR, textRx, textRy);
          ctx.fillStyle = colSideAngles;
          ctx.fillText(textR, textRx, textRy);

          ctx.restore();
        }
      }

      // 6. Draw Interactive Handles: Left, Right, and TOP APEX HANDLE
      function drawHandle(pt, label, col, isApex = false) {
        ctx.save();
        ctx.shadowColor = 'rgba(0,0,0,0.7)';
        ctx.shadowBlur = 8;

        ctx.fillStyle = col;
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, isApex ? 9 : 8, 0, 2 * Math.PI);
        ctx.fill();

        ctx.strokeStyle = isApex ? '#38bdf8' : '#0a0e17';
        ctx.lineWidth = 2.5;
        ctx.stroke();

        ctx.font = 'bold 11px "JetBrains Mono", monospace';
        ctx.fillStyle = col;
        ctx.fillText(label, pt.x + 12, pt.y - 10);
        ctx.restore();
      }

      const pA = toScreen(state.handles.apex.x, state.handles.apex.y);
      drawHandle(p1, 'A (Left)', colHandles);
      drawHandle(p2, 'B (Right)', colHandles);
      drawHandle(pA, 'H (Apex 2D)', colFit, true);

      // 7. Clean Floating Angle Badge
      if (showBadge && state.wca) {
        const asym = Math.abs(state.leftWCA - state.rightWCA);
        const badgeTitle = `${state.wca.toFixed(1)}°`;
        const showSub = showSideAngles || asym >= 0.2;
        const badgeSub = showSub ? `L: ${state.leftWCA.toFixed(1)}°  R: ${state.rightWCA.toFixed(1)}°` : '';

        ctx.save();
        ctx.font = 'bold 22px "Plus Jakarta Sans", sans-serif';
        const txtWidth = Math.max(ctx.measureText(badgeTitle).width, badgeSub ? 130 : 0);
        const bHeight = badgeSub ? 52 : 36;
        const bx = pA.x - txtWidth / 2 - 14;
        const by = pA.y - bHeight - 14;

        ctx.fillStyle = 'rgba(10, 14, 23, 0.88)';
        ctx.strokeStyle = colFit;
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        ctx.roundRect(bx, by, txtWidth + 28, bHeight, 10);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = colBadge;
        ctx.fillText(badgeTitle, bx + (txtWidth + 28 - ctx.measureText(badgeTitle).width) / 2, by + 25);

        if (badgeSub) {
          ctx.font = 'bold 11px "JetBrains Mono", monospace';
          ctx.fillStyle = colSideAngles;
          ctx.fillText(badgeSub, bx + (txtWidth + 28 - ctx.measureText(badgeSub).width) / 2, by + 43);
        }
        ctx.restore();
      }
    }

    // --- Interactive Mouse / Pointer Dragging (Handles & Background Pan) ---
    canvas.addEventListener('pointerdown', (e) => {
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const pos = {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
      };

      const viewW = state.cropW;
      const viewH = state.cropH;
      const ox = viewW / 2 - state.cropCenter.x;
      const oy = viewH / 2 - state.cropCenter.y;
      const rotRad = (state.rotationDeg * Math.PI) / 180.0;

      function toScreen(px, py) {
        const x = px + ox;
        const y = py + oy;
        const rx = x - viewW / 2;
        const ry = y - viewH / 2;
        return {
          x: viewW / 2 + (rx * Math.cos(rotRad) - ry * Math.sin(rotRad)),
          y: viewH / 2 + (rx * Math.sin(rotRad) + ry * Math.cos(rotRad))
        };
      }

      const sL = toScreen(state.handles.left.x, state.handles.left.y);
      const sR = toScreen(state.handles.right.x, state.handles.right.y);
      const sA = toScreen(state.handles.apex.x, state.handles.apex.y);

      const hitRadius = 26;
      const dL = Math.hypot(pos.x - sL.x, pos.y - sL.y);
      const dR = Math.hypot(pos.x - sR.x, pos.y - sR.y);
      const dA = Math.hypot(pos.x - sA.x, pos.y - sA.y);

      if (dA < hitRadius) {
        state.activeDrag = 'apex';
      } else if (dL < hitRadius) {
        state.activeDrag = 'left';
      } else if (dR < hitRadius) {
        state.activeDrag = 'right';
      } else {
        // Background click -> Reposition / Pan Droplet!
        state.activeDrag = 'pan';
        state.panStartPos = { x: pos.x, y: pos.y };
        state.panStartCenter = { ...state.cropCenter };
        canvas.style.cursor = 'grabbing';
      }

      canvas.setPointerCapture(e.pointerId);
    });

    canvas.addEventListener('pointermove', (e) => {
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const pos = {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
      };

      const viewW = state.cropW;
      const viewH = state.cropH;
      const ox = viewW / 2 - state.cropCenter.x;
      const oy = viewH / 2 - state.cropCenter.y;
      const rotRad = (state.rotationDeg * Math.PI) / 180.0;

      if (state.activeDrag === 'pan') {
        const dx = pos.x - state.panStartPos.x;
        const dy = pos.y - state.panStartPos.y;
        const unRotDx = dx * Math.cos(-rotRad) - dy * Math.sin(-rotRad);
        const unRotDy = dx * Math.sin(-rotRad) + dy * Math.cos(-rotRad);
        state.cropCenter.x = Math.max(0, Math.min(state.imgWidth, Math.round(state.panStartCenter.x - unRotDx)));
        state.cropCenter.y = Math.max(0, Math.min(state.imgHeight, Math.round(state.panStartCenter.y - unRotDy)));
        updateCenterDisplay();
        redrawCanvas();
        return;
      }

      function fromScreen(sx, sy) {
        const rx = sx - viewW / 2;
        const ry = sy - viewH / 2;
        const unRotX = rx * Math.cos(-rotRad) - ry * Math.sin(-rotRad);
        const unRotY = rx * Math.sin(-rotRad) + ry * Math.cos(-rotRad);
        return {
          x: unRotX + viewW / 2 - ox,
          y: unRotY + viewH / 2 - oy
        };
      }

      if (state.activeDrag) {
        const unRot = fromScreen(pos.x, pos.y);

        if (state.activeDrag === 'left') {
          state.handles.left.x = Math.round(unRot.x);
          state.handles.left.y = Math.round(unRot.y);
          recalculateFromGeometry();
          redrawCanvas();
        } else if (state.activeDrag === 'right') {
          state.handles.right.x = Math.round(unRot.x);
          state.handles.right.y = Math.round(unRot.y);
          recalculateFromGeometry();
          redrawCanvas();
        } else if (state.activeDrag === 'apex') {
          const p1 = state.handles.left;
          const p2 = state.handles.right;
          const baseSpan = Math.hypot(p2.x - p1.x, p2.y - p1.y);
          const mx = (p1.x + p2.x) / 2.0;
          const my = (p1.y + p2.y) / 2.0;
          const ux = (p2.x - p1.x) / baseSpan;
          const uy = (p2.y - p1.y) / baseSpan;
          let nx = uy, ny = -ux;
          if (ny > 0) { nx = -nx; ny = -ny; }

          const vx = unRot.x - mx;
          const vy = unRot.y - my;

          // Lateral shift along baseline (Left / Right asymmetry)
          const projShift = vx * ux + vy * uy;
          const maxShift = baseSpan * 0.44;
          state.apexShiftPx = Math.max(-maxShift, Math.min(maxShift, projShift));

          // Vertical height along normal
          const projH = vx * nx + vy * ny;
          const newH = Math.max(12, projH);

          if (state.baseAutoFit) {
            const baseB = baseSpan / 2.0;
            const autoH = baseB * Math.tan((state.baseAutoFit.thetaDeg * Math.PI) / 360.0);
            state.heightOffsetPx = Math.round(newH - autoH);
            document.getElementById('heightOffsetSlider').value = state.heightOffsetPx;
            document.getElementById('valHeightOffset').innerText = (state.heightOffsetPx >= 0 ? '+' : '') + state.heightOffsetPx + ' px';
          }

          document.getElementById('apexShiftSlider').value = Math.round(state.apexShiftPx);
          document.getElementById('valApexShift').innerText = (state.apexShiftPx >= 0 ? '+' : '') + Math.round(state.apexShiftPx) + ' px';

          recalculateFromGeometry();
          redrawCanvas();
        }
      } else {
        function toScreen(px, py) {
          const x = px + ox;
          const y = py + oy;
          const rx = x - viewW / 2;
          const ry = y - viewH / 2;
          return {
            x: viewW / 2 + (rx * Math.cos(rotRad) - ry * Math.sin(rotRad)),
            y: viewH / 2 + (rx * Math.sin(rotRad) + ry * Math.cos(rotRad))
          };
        }

        const sL = toScreen(state.handles.left.x, state.handles.left.y);
        const sR = toScreen(state.handles.right.x, state.handles.right.y);
        const sA = toScreen(state.handles.apex.x, state.handles.apex.y);

        const hitRadius = 26;
        const dL = Math.hypot(pos.x - sL.x, pos.y - sL.y);
        const dR = Math.hypot(pos.x - sR.x, pos.y - sR.y);
        const dA = Math.hypot(pos.x - sA.x, pos.y - sA.y);

        if (dL < hitRadius || dR < hitRadius || dA < hitRadius) {
          canvas.style.cursor = 'grab';
        } else {
          canvas.style.cursor = 'move';
        }
      }
    });

    canvas.addEventListener('pointerup', (e) => {
      state.activeDrag = null;
      canvas.style.cursor = 'move';
      try { canvas.releasePointerCapture(e.pointerId); } catch(ex){}
      if (activeImageId) {
        const record = sessionImages.find(r => r.id === activeImageId);
        if (record) {
          record.cropCenter = { ...state.cropCenter };
          record.cropW = state.cropW;
          record.cropH = state.cropH;
          record.apexShiftPx = state.apexShiftPx;
        }
      }
    });

    // --- Export Functions ---
    function downloadCurrentCanvas() {
      const link = document.createElement('a');
      link.download = state.filename.replace(/\\.[^/.]+$/, '') + '_wca_measurement.png';
      link.href = canvas.toDataURL('image/png');
      link.click();
      showToast('Exported clean image.');
    }

    function downloadBatchCSV() {
      if (sessionImages.length === 0) {
        alert('No image measurements to export.');
        return;
      }
      const stats = computeStatistics();

      let csv = 'Image Filename,Status,Contact Angle (deg),Left Angle (deg),Right Angle (deg),Asymmetry (deg),Apex Lateral Shift (px),Apex Height (px),Base Span (px),Curvature Radius (px),Image Rotation (deg)\\n';
      sessionImages.forEach(r => {
        const asym = Math.abs(r.leftWCA - r.rightWCA).toFixed(2);
        const shift = (r.apexShiftPx || 0).toFixed(1);
        csv += `"${r.filename}","${r.active ? 'Included' : 'Excluded'}",${r.wca.toFixed(2)},${r.leftWCA.toFixed(2)},${r.rightWCA.toFixed(2)},${asym},${shift},${r.h.toFixed(1)},${(2*r.b).toFixed(1)},${r.R.toFixed(1)},${r.rotationDeg.toFixed(1)}\\n`;
      });

      csv += '\\n--- LABORATORY SUMMARY & STATISTICS ---\\n';
      csv += `Valid Sample Count (N),${stats.N}\\n`;
      csv += `Total Loaded Images,${stats.total}\\n`;
      csv += `Sample Mean Contact Angle (deg),${stats.mean.toFixed(3)}\\n`;
      csv += `Sample Standard Deviation (SD deg),${stats.sd.toFixed(3)}\\n`;
      csv += `Standard Error of Mean (SEM deg),${stats.sem.toFixed(3)}\\n`;
      csv += `95% Confidence Interval (deg),"[${stats.ciLower.toFixed(3)}, ${stats.ciUpper.toFixed(3)}]"\\n`;
      csv += `Relative Standard Error (RSE %),${stats.rse.toFixed(2)}%\\n`;
      csv += `Mean Left Contact Angle (deg),${stats.meanLeft.toFixed(3)}\\n`;
      csv += `Mean Right Contact Angle (deg),${stats.meanRight.toFixed(3)}\\n`;
      csv += `Mean Asymmetry (deg),${stats.meanAsym.toFixed(3)}\\n`;
      csv += `Mean Apex Lateral Shift (px),${stats.meanApexShift.toFixed(2)}\\n`;

      const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'wca_droplet_statistics_summary.csv';
      a.click();
      showToast('Exported CSV with complete statistics.');
    }

    function copyStatsToClipboard() {
      const stats = computeStatistics();
      if (stats.N === 0) {
        alert('No measurements available.');
        return;
      }
      const text = `Water Contact Angle: ${stats.mean.toFixed(2)}° ± ${stats.sem.toFixed(2)}° (SEM) [SD: ±${stats.sd.toFixed(2)}°, 95% CI: ${stats.ciLower.toFixed(2)}° - ${stats.ciUpper.toFixed(2)}°, Mean Asymmetry: ${stats.meanAsym.toFixed(2)}°, N = ${stats.N}]`;
      navigator.clipboard.writeText(text).then(() => {
        showToast('Copied summary to clipboard!');
      });
    }

    // --- Onload Initialization ---
    window.onload = function() {
      loadAllTestSamples();
    };
  </script>
</body>
</html>
"""


class ModernWCAHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode("utf-8"))

        elif path == "/api/list_samples":
            folder = os.path.join(BASE_DIR, "test images")
            samples = []
            if os.path.isdir(folder):
                for fname in sorted(os.listdir(folder)):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        samples.append(fname)
            self.send_json({"success": True, "samples": samples})

        elif path == "/api/load_sample":
            query = urllib.parse.parse_qs(parsed.query)
            sample_name = query.get("name", [""])[0]
            sample_path = os.path.join(BASE_DIR, "test images", sample_name)
            if os.path.isfile(sample_path):
                with open(sample_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                self.send_json({"success": True, "base64": b64, "name": sample_name})
            else:
                self.send_json({"success": False, "error": f"Sample not found: {sample_name}"})

        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path == "/api/analyze_raw":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
                img_b64 = data.get("image", "")
                filename = data.get("filename", "upload.jpg")
                img_bytes = base64.b64decode(img_b64)
                result = process_image_raw(img_bytes, filename=filename)
                self.send_json(result)
            except Exception as e:
                self.send_json({"success": False, "error": str(e)})
        else:
            self.send_error(404, "Not Found")

    def send_json(self, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def start_server():
    server = HTTPServer(("127.0.0.1", PORT), ModernWCAHandler)
    url = f"http://127.0.0.1:{PORT}"
    print("=" * 65)
    print("  WATER CONTACT ANGLE (WCA) STUDIO - MULTI-DROPLET & ASYMMETRY")
    print("=" * 65)
    print(f"Server live at: {url}")
    print("=" * 65)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    start_server()
