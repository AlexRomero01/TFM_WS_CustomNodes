#!/usr/bin/env python3
"""
classical_vs_gt_eval.py
========================
Evaluates classical segmentation algorithms (ExG and HSV) against
ground-truth polygon annotations exported from Roboflow in YOLO v8
Segmentation format.

Usage:
    python3 classical_vs_gt_eval.py --dataset_dir /path/to/dataset/test

Expected directory structure (YOLO v8 Seg export from Roboflow):
    dataset/
        test/
            images/   ← JPG/PNG files
            labels/   ← .txt files with normalised polygon coords

Output:
    Prints a per-class and aggregate metric table to stdout.
    Saves a qualitative comparison grid (optional, --save_vis).
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Classical segmentation algorithms
# ─────────────────────────────────────────────────────────────────────────────

def segment_exg(img_bgr: np.ndarray) -> np.ndarray:
    """Excess Green Index: ExG = 2G - R - B, Otsu binarisation.
    Returns a binary mask (uint8, 0/255)."""
    img = img_bgr.astype(np.float32)
    exg = 2.0 * img[:, :, 1] - img[:, :, 0] - img[:, :, 2]
    # Normalise to 0-255 for Otsu
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = _morphological_refine(mask)
    return mask


def segment_hsv(img_bgr: np.ndarray,
                h_lo: int = 35, h_hi: int = 85,
                s_lo: int = 40, s_hi: int = 255,
                v_lo: int = 30, v_hi: int = 255) -> np.ndarray:
    """HSV green-channel thresholding.
    Hue range 35°–85° isolates green pigments independently of luminance.
    Returns a binary mask (uint8, 0/255)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
    upper = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = _morphological_refine(mask)
    return mask


def _morphological_refine(mask: np.ndarray,
                           kernel_size: int = 5) -> np.ndarray:
    """Closing then Opening with an elliptical kernel — same as the ROS2 node."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth annotation parser (YOLO v8 Segmentation format)
# ─────────────────────────────────────────────────────────────────────────────

def load_gt_mask(label_path: Path, img_h: int, img_w: int) -> np.ndarray:
    """Parse a YOLO-seg .txt file and rasterise all polygons into a binary mask.

    Each line: class_id  x1 y1 x2 y2 ... xN yN   (all values normalised 0-1)
    Returns a uint8 mask (0 = background, 255 = plant).
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if not label_path.exists():
        return mask  # no annotations → pure background image

    with label_path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:       # need class_id + at least 3 (x,y) pairs
                continue
            coords = list(map(float, parts[1:]))   # drop class_id
            # Reshape to (N, 2) and denormalise
            pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] *= img_w
            pts[:, 1] *= img_h
            pts = pts.astype(np.int32)
            cv2.fillPoly(mask, [pts], color=255)

    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Metric calculation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(pred_mask: np.ndarray,
                    gt_mask:   np.ndarray) -> dict:
    """Binary pixel-level metrics between two uint8 masks (0/255).

    Returns:
        tp, fp, fn, tn, iou, precision, recall, f1, dice
    """
    pred_bin = pred_mask > 0
    gt_bin   = gt_mask   > 0

    tp = int(np.logical_and( pred_bin,  gt_bin).sum())
    fp = int(np.logical_and( pred_bin, ~gt_bin).sum())
    fn = int(np.logical_and(~pred_bin,  gt_bin).sum())
    tn = int(np.logical_and(~pred_bin, ~gt_bin).sum())

    iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp)       if (tp + fp)      > 0 else 0.0
    recall    = tp / (tp + fn)       if (tp + fn)      > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    dice      = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                iou=iou, precision=precision, recall=recall,
                f1=f1, dice=dice)


def aggregate_metrics(per_image: list[dict]) -> dict:
    """Micro-average across all images (sum TP/FP/FN, then recompute)."""
    tp = sum(m['tp'] for m in per_image)
    fp = sum(m['fp'] for m in per_image)
    fn = sum(m['fn'] for m in per_image)

    iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp)       if (tp + fp)      > 0 else 0.0
    recall    = tp / (tp + fn)       if (tp + fn)      > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    dice      = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    return dict(tp=tp, fp=fp, fn=fn,
                iou=iou, precision=precision, recall=recall,
                f1=f1, dice=dice)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helper
# ─────────────────────────────────────────────────────────────────────────────

def make_comparison_tile(img_bgr: np.ndarray,
                         gt_mask:  np.ndarray,
                         exg_mask: np.ndarray,
                         hsv_mask: np.ndarray,
                         title: str = "") -> np.ndarray:
    """Build a 1×4 tile: Original | GT | ExG | HSV."""
    def overlay(base, mask, color=(0, 255, 0), alpha=0.45):
        vis = base.copy()
        vis[mask > 0] = (
            (1 - alpha) * vis[mask > 0] + alpha * np.array(color)
        ).astype(np.uint8)
        return vis

    gt_vis  = overlay(img_bgr, gt_mask,  color=(0, 200, 0))
    exg_vis = overlay(img_bgr, exg_mask, color=(0, 180, 255))
    hsv_vis = overlay(img_bgr, hsv_mask, color=(255, 100, 0))

    def label(img, txt):
        out = img.copy()
        cv2.putText(out, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return out

    row = np.hstack([
        label(img_bgr, "Original"),
        label(gt_vis,  "Ground Truth"),
        label(exg_vis, "ExG"),
        label(hsv_vis, "HSV"),
    ])

    if title:
        cv2.putText(row, title, (8, row.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(dataset_dir: str,
             save_vis: bool = False,
             vis_dir: str = "eval_vis",
             max_vis: int = 10) -> None:

    img_dir   = Path(dataset_dir) / "images"
    label_dir = Path(dataset_dir) / "labels"

    if not img_dir.exists():
        sys.exit(f"[ERROR] images/ not found under: {dataset_dir}")
    if not label_dir.exists():
        sys.exit(f"[ERROR] labels/ not found under: {dataset_dir}")

    img_paths = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
    )
    if not img_paths:
        sys.exit(f"[ERROR] No images found in {img_dir}")

    print(f"\n{'─'*60}")
    print(f"  Dataset : {dataset_dir}")
    print(f"  Images  : {len(img_paths)}")
    print(f"{'─'*60}\n")

    exg_metrics_all = []
    hsv_metrics_all = []

    if save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    for i, img_path in enumerate(img_paths):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[WARN] Cannot read: {img_path.name}  — skipping")
            continue

        h, w = img_bgr.shape[:2]
        label_path = label_dir / (img_path.stem + ".txt")

        gt_mask  = load_gt_mask(label_path, h, w)
        exg_mask = segment_exg(img_bgr)
        hsv_mask = segment_hsv(img_bgr)

        exg_m = compute_metrics(exg_mask, gt_mask)
        hsv_m = compute_metrics(hsv_mask, gt_mask)
        exg_metrics_all.append(exg_m)
        hsv_metrics_all.append(hsv_m)

        if save_vis and i < max_vis:
            tile = make_comparison_tile(img_bgr, gt_mask, exg_mask, hsv_mask,
                                        title=img_path.name)
            out_path = Path(vis_dir) / f"vis_{img_path.stem}.jpg"
            cv2.imwrite(str(out_path), tile)

        # Progress
        print(f"  [{i+1:>4}/{len(img_paths)}] {img_path.name:40s}"
              f"  ExG IoU={exg_m['iou']:.3f}  HSV IoU={hsv_m['iou']:.3f}")

    # ── Aggregate ──────────────────────────────────────────────────────────
    exg_agg = aggregate_metrics(exg_metrics_all)
    hsv_agg = aggregate_metrics(hsv_metrics_all)

    # ── Pretty print ───────────────────────────────────────────────────────
    header = f"\n{'─'*62}\n  AGGREGATE RESULTS  (micro-average over {len(img_paths)} images)\n{'─'*62}"
    print(header)

    col_w = 20
    print(f"\n  {'Metric':<{col_w}} {'ExG':>12} {'HSV':>12}")
    print(f"  {'─' * col_w} {'─'*12} {'─'*12}")

    metrics_to_show = [
        ("IoU (Jaccard)",   "iou"),
        ("Precision",       "precision"),
        ("Recall",          "recall"),
        ("F1 Score",        "f1"),
        ("Dice Coefficient","dice"),
    ]
    for label, key in metrics_to_show:
        ev = exg_agg[key]
        hv = hsv_agg[key]
        print(f"  {label:<{col_w}} {ev:>12.4f} {hv:>12.4f}")

    print(f"\n  {'Pixel Counts (aggregated)'}")
    for key in ('tp', 'fp', 'fn'):
        print(f"  {key.upper():<{col_w}} {exg_agg[key]:>12,} {hsv_agg[key]:>12,}")

    print(f"\n{'─'*62}")
    print("  LATEX TABLE (copy into your report):")
    print(f"{'─'*62}")

    latex = r"""
\begin{table}[htbp]
\centering
\caption{Pixel-level segmentation performance of classical methods on the test set ($N$=XX images), compared to YOLO polygon ground-truth masks.}
\label{tab:classical_benchmark}
\begin{tabular}{lccc}
\toprule
\textbf{Metric} & \textbf{ExG} & \textbf{HSV} \\
\midrule"""

    for label, key in metrics_to_show:
        ev = exg_agg[key]
        hv = hsv_agg[key]
        latex += f"\n{label} & {ev:.4f} & {hv:.4f} \\\\"

    latex += r"""
\bottomrule
\end{tabular}
\end{table}"""
    print(latex)
    print(f"\n{'─'*62}\n")

    if save_vis:
        print(f"  Visualisations saved to: {vis_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ExG and HSV vs. YOLO-seg ground-truth labels.")
    parser.add_argument(
        "--dataset_dir", required=True,
        help="Path to the test split, e.g. /path/to/dataset/test")
    parser.add_argument(
        "--save_vis", action="store_true",
        help="Save side-by-side comparison images (slow for large datasets)")
    parser.add_argument(
        "--vis_dir", default="eval_vis",
        help="Output folder for visualisation tiles (default: eval_vis)")
    parser.add_argument(
        "--max_vis", type=int, default=10,
        help="Maximum number of visualisation tiles to save (default: 10)")
    args = parser.parse_args()

    evaluate(
        dataset_dir=args.dataset_dir,
        save_vis=args.save_vis,
        vis_dir=args.vis_dir,
        max_vis=args.max_vis,
    )


if __name__ == "__main__":
    main()
