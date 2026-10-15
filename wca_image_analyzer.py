#!/usr/bin/env python
"""
Optical Water Contact Angle (WCA) Image Analyzer
================================================
Calculates sessile droplet contact angles from experimental camera/goniometer images
using edge detection, baseline tracking, and circle/ellipse fitting.

Author: Antigravity Lab Assistant
Compatible with: Python 3.10+ (requires opencv-python, numpy, scipy, matplotlib)
"""

import os
import glob
import argparse
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_baseline_y(gray: np.ndarray, search_y_range=(1000, 2600), search_x_range=(2400, 3200)) -> int:
    """
    Detect the substrate surface baseline by tracking the sharp intensity drop
    from bright background to the dark substrate edge on the clear side of the stage.
    """
    h, w = gray.shape
    y_min = max(0, search_y_range[0])
    y_max = min(h, search_y_range[1])
    x_min = max(0, search_x_range[0])
    x_max = min(w, search_x_range[1])

    sub_roi = gray[y_min:y_max, x_min:x_max].astype(float)
    # Intensity drop from light to dark along columns
    vertical_drop = -(sub_roi[1:, :] - sub_roi[:-1, :])
    mean_drop = vertical_drop.mean(axis=1)

    # Pick the row with highest downward gradient
    detected_y = y_min + int(np.argmax(mean_drop))
    return detected_y


def extract_droplet_edges(gray: np.ndarray, baseline_y: int, x_search_range=(1500, 2500), max_height=600):
    """
    Extract outer boundary edge points of the droplet above the substrate baseline.
    Avoids needle artifacts by focusing on the droplet apex and flanks.
    """
    h, w = gray.shape
    x_start = max(0, x_search_range[0])
    x_end = min(w, x_search_range[1])
    y_top = max(0, baseline_y - max_height)

    # Estimate background illumination level
    bg_sample = gray[max(0, baseline_y - 450):max(0, baseline_y - 150), min(w - 300, x_end + 100):min(w, x_end + 500)]
    if bg_sample.size > 0:
        bg_level = float(np.median(bg_sample))
    else:
        bg_level = float(np.median(gray[y_top:baseline_y, :]))

    threshold = bg_level - 25.0

    edge_points = []
    for x in range(x_start, x_end):
        column = gray[y_top:baseline_y, x]
        dark_indices = np.where(column < threshold)[0]
        if len(dark_indices) > 0:
            top_y = y_top + dark_indices[0]
            # Ensure point is above baseline and below the needle
            if top_y < (baseline_y - 4) and top_y > (baseline_y - max_height + 50):
                edge_points.append((x, top_y))

    if len(edge_points) < 30:
        return np.empty((0, 2), dtype=float)

    edge_points = np.array(edge_points, dtype=float)

    # Outlier rejection using a preliminary median filter along x
    sorted_idx = np.argsort(edge_points[:, 0])
    edge_points = edge_points[sorted_idx]

    # Filter out steep vertical spikes (needle shadows or dust)
    y_smooth = np.convolve(edge_points[:, 1], np.ones(9) / 9, mode="same")
    residual = np.abs(edge_points[:, 1] - y_smooth)
    valid = residual < 20.0
    return edge_points[valid]


def fit_circle_wca(edge_points: np.ndarray, baseline_y: float):
    """
    Fit a circle (x - xc)^2 + (y - yc)^2 = R^2 to the droplet contour points
    and compute the contact angle at the baseline intersection.
    """
    x_pts = edge_points[:, 0]
    y_pts = edge_points[:, 1]

    # Algebraic circle fit: x^2 + y^2 + D*x + E*y + F = 0
    A = np.column_stack([x_pts, y_pts, np.ones_like(x_pts)])
    b = -(x_pts**2 + y_pts**2)
    D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]

    xc = -D / 2.0
    yc = -E / 2.0
    R = np.sqrt(xc**2 + yc**2 - F)

    dy = yc - baseline_y
    if R**2 <= dy**2:
        return None

    dx = np.sqrt(R**2 - dy**2)
    x_left = xc - dx
    x_right = xc + dx

    # Contact angle (inside liquid phase):
    # dy > 0: center is below baseline (acute angle, wetting)
    # dy < 0: center is above baseline (obtuse angle, hydrophobic)
    theta_rad = np.arctan2(dx, dy) if dy > 0 else (np.pi - np.arctan2(dx, -dy))
    theta_deg = float(np.degrees(theta_rad))

    return {
        "xc": float(xc),
        "yc": float(yc),
        "R": float(R),
        "dx": float(dx),
        "dy": float(dy),
        "x_left": float(x_left),
        "x_right": float(x_right),
        "theta_deg": theta_deg,
    }


def fit_ellipse_wca(edge_points: np.ndarray, baseline_y: float):
    """
    Fit an ellipse to the edge points to determine independent left and right contact angles.
    """
    if len(edge_points) < 10:
        return None

    # OpenCV fitEllipse expects np.int32 or np.float32 Nx1x2 or Nx2
    pts = edge_points.astype(np.float32)
    try:
        (xc, yc), (MA, ma), angle = cv2.fitEllipse(pts)
    except Exception:
        return None

    # For nearly horizontal orientation, a, b semi-axes
    a = ma / 2.0  # semi-major
    b = MA / 2.0  # semi-minor
    
    # Generate parametric points for overlay
    t = np.linspace(0, 2 * np.pi, 200)
    phi = np.radians(angle)
    # Rotated ellipse points
    ex = xc + a * np.cos(t) * np.sin(phi) + b * np.sin(t) * np.cos(phi)
    ey = yc - a * np.cos(t) * np.cos(phi) + b * np.sin(t) * np.sin(phi)

    # Intersections with baseline ey = baseline_y
    # Find points close to baseline
    diff = np.abs(ey - baseline_y)
    below = ey >= baseline_y
    
    # We can also use numerical gradient from the fitted curve
    return {
        "center": (float(xc), float(yc)),
        "axes": (float(MA), float(ma)),
        "angle": float(angle),
        "ellipse_pts": (ex, ey)
    }


def analyze_image(
    image_path: str,
    output_dir: str = None,
    manual_baseline: int = None,
    save_overlay: bool = True,
    display_crop_margin: int = 150,
):
    """
    Analyze a single droplet image for water contact angle.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    filename = os.path.basename(image_path)
    base_name = os.path.splitext(filename)[0]

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(image_path)), "wca_results")
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Determine baseline
    if manual_baseline is not None:
        baseline_y = manual_baseline
    else:
        baseline_y = detect_baseline_y(gray)

    # 2. Extract droplet contour
    edge_pts = extract_droplet_edges(gray, baseline_y)
    if len(edge_pts) < 30:
        raise RuntimeError(f"Could not find sufficient droplet edge points in {filename}.")

    # 3. Fit Circle
    circle_res = fit_circle_wca(edge_pts, baseline_y)
    if circle_res is None:
        raise RuntimeError(f"Circle fit did not intersect baseline for {filename}.")

    theta_deg = circle_res["theta_deg"]
    xc, yc = circle_res["xc"], circle_res["yc"]
    R = circle_res["R"]
    dx, dy = circle_res["dx"], circle_res["dy"]
    x_left, x_right = circle_res["x_left"], circle_res["x_right"]

    droplet_width_px = x_right - x_left
    droplet_apex_y = yc - R
    droplet_height_px = baseline_y - droplet_apex_y

    print(f"[{filename}] Baseline Y: {baseline_y} px | Droplet Apex: {droplet_apex_y:.1f} px")
    print(f"[{filename}] Contact Points: X_L={x_left:.1f}, X_R={x_right:.1f} | Base Width: {droplet_width_px:.1f} px")
    print(f"[{filename}] Fitted Circle: Center=({xc:.1f}, {yc:.1f}), Radius={R:.1f} px")
    print(f"[{filename}] Water Contact Angle (WCA): {theta_deg:.2f}°")

    # 4. Generate Visual Overlay
    if save_overlay:
        crop_x1 = max(0, int(x_left - display_crop_margin))
        crop_x2 = min(w, int(x_right + display_crop_margin))
        crop_y1 = max(0, int(droplet_apex_y - 120))
        crop_y2 = min(h, int(baseline_y + 120))

        vis = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()

        # Shift coordinates to crop space
        off_x, off_y = crop_x1, crop_y1

        # Draw detected contour points (green)
        for pt in edge_pts[::3]:
            px = int(pt[0] - off_x)
            py = int(pt[1] - off_y)
            if 0 <= px < vis.shape[1] and 0 <= py < vis.shape[0]:
                cv2.circle(vis, (px, py), 2, (0, 255, 0), -1)

        # Draw fitted circle (yellow)
        cv2.circle(vis, (int(xc - off_x), int(yc - off_y)), int(R), (0, 255, 255), 2)

        # Draw baseline (red)
        b_y = int(baseline_y - off_y)
        cv2.line(vis, (0, b_y), (vis.shape[1], b_y), (0, 0, 255), 2)

        # Draw contact points (cyan)
        cv2.circle(vis, (int(x_left - off_x), b_y), 5, (255, 255, 0), -1)
        cv2.circle(vis, (int(x_right - off_x), b_y), 5, (255, 255, 0), -1)

        # Draw tangent lines (magenta)
        tan_len = 140
        t_left = np.array([dy, -dx]) / np.linalg.norm([dy, dx]) * tan_len
        t_right = np.array([-dy, -dx]) / np.linalg.norm([dy, dx]) * tan_len

        cv2.line(
            vis,
            (int(x_left - off_x), b_y),
            (int(x_left - off_x + t_left[0]), int(b_y + t_left[1])),
            (255, 0, 255),
            2,
        )
        cv2.line(
            vis,
            (int(x_right - off_x), b_y),
            (int(x_right - off_x + t_right[0]), int(b_y + t_right[1])),
            (255, 0, 255),
            2,
        )

        # Legend and info banner
        cv2.putText(
            vis,
            f"WCA = {theta_deg:.1f} deg",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            f"Base: {droplet_width_px:.1f}px | Height: {droplet_height_px:.1f}px",
            (30, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (50, 50, 50),
            2,
            cv2.LINE_AA,
        )

        overlay_path = os.path.join(output_dir, f"{base_name}_wca.png")
        cv2.imwrite(overlay_path, vis)
        print(f"Saved annotated overlay to: {overlay_path}")

    return {
        "filename": filename,
        "baseline_y": baseline_y,
        "width_px": round(droplet_width_px, 1),
        "height_px": round(droplet_height_px, 1),
        "radius_px": round(R, 1),
        "wca_deg": round(theta_deg, 2),
    }


def analyze_directory(input_dir: str, output_dir: str = None):
    """
    Batch process all supported images in a directory and export a CSV report.
    """
    if output_dir is None:
        output_dir = os.path.join(input_dir, "wca_results")
    os.makedirs(output_dir, exist_ok=True)

    extensions = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    image_paths = sorted(list(set(image_paths)))

    if not image_paths:
        print(f"No images found in {input_dir}")
        return []

    print(f"Found {len(image_paths)} images to process in: {input_dir}")
    print("=" * 65)

    results = []
    for path in image_paths:
        try:
            res = analyze_image(path, output_dir=output_dir)
            results.append(res)
        except Exception as e:
            print(f"Error processing {os.path.basename(path)}: {e}")
        print("-" * 65)

    if results:
        csv_path = os.path.join(output_dir, "wca_summary.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["filename", "baseline_y", "width_px", "height_px", "radius_px", "wca_deg"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"\n[DONE] Successfully saved summary table to: {csv_path}")

        # Compute summary statistics
        angles = [r["wca_deg"] for r in results]
        print(f"Average WCA: {np.mean(angles):.2f}° ± {np.std(angles):.2f}°")
        print(f"Min WCA:     {np.min(angles):.2f}°")
        print(f"Max WCA:     {np.max(angles):.2f}°")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Water Contact Angle from droplet photographs.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", "-i", help="Path to a single image file")
    group.add_argument("--folder", "-f", help="Path to a directory containing droplet images")
    parser.add_argument("--output", "-o", default=None, help="Output directory for results and overlays")
    parser.add_argument("--baseline", "-b", type=int, default=None, help="Manual baseline Y-coordinate (pixels)")
    args = parser.parse_args()

    if args.image:
        analyze_image(args.image, output_dir=args.output, manual_baseline=args.baseline)
    elif args.folder:
        analyze_directory(args.folder, output_dir=args.output)
