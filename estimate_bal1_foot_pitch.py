import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def line_angle_deg(x1: int, y1: int, x2: int, y2: int) -> float:
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate foot-support frame pitch from BAL-1 image")
    parser.add_argument("--image", type=str, default="bal1.jpg")
    parser.add_argument("--out-json", type=str, default="output_data/bal1_image_pitch_estimate.json")
    parser.add_argument("--out-overlay", type=str, default="output_data/bal1_line_candidates.jpg")
    parser.add_argument("--dims-json", type=str, default="wheelchair_bal1_dims.json")
    parser.add_argument("--update-dims", action="store_true", help="Update dims JSON with estimated pitch")
    args = parser.parse_args()

    image_path = Path(args.image)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 70, 180)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=80, maxLineGap=15)
    if lines is None:
        raise RuntimeError("No lines detected")

    candidates = []

    # ROI: front-lower area where foot-support frame likely appears
    x_min = int(w * 0.45)
    y_min = int(h * 0.40)

    for l in lines[:, 0, :]:
        x1, y1, x2, y2 = map(int, l)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        if cx < x_min or cy < y_min:
            continue
        dx = x2 - x1
        dy = y2 - y1
        length = float(math.hypot(dx, dy))
        if length < 90:
            continue
        if abs(dx) < 10:
            continue

        ang = line_angle_deg(x1, y1, x2, y2)

        # keep oblique lines, reject near horizontal/vertical
        if not (12 <= abs(ang) <= 75):
            continue

        # preference around expected |pitch| ~ 35 deg
        closeness = max(0.0, 1.0 - abs(abs(ang) - 35.0) / 40.0)
        score = length * (0.6 + 0.4 * closeness)

        candidates.append({
            "line": [x1, y1, x2, y2],
            "angle_deg": ang,
            "length": length,
            "score": score,
            "center": [cx, cy],
        })

    if not candidates:
        raise RuntimeError("No suitable candidate lines found in ROI")

    candidates.sort(key=lambda c: c["score"], reverse=True)
    top = candidates[:20]

    # robust estimate from top-N weighted median-like selection
    angles = np.array([c["angle_deg"] for c in top], dtype=float)
    weights = np.array([c["score"] for c in top], dtype=float)

    # choose sign cluster by higher total weight near +/-
    pos_w = weights[angles > 0].sum()
    neg_w = weights[angles < 0].sum()
    if pos_w >= neg_w:
        sel = angles > 0
        selected_sign = "+"
    else:
        sel = angles < 0
        selected_sign = "-"

    sel_angles = angles[sel]
    sel_weights = weights[sel]
    if len(sel_angles) < 3:
        sel_angles = angles
        sel_weights = weights
        selected_sign = "mixed"

    est_angle = float(np.average(sel_angles, weights=sel_weights))
    spread = float(np.sqrt(np.average((sel_angles - est_angle) ** 2, weights=sel_weights)))

    # Convert image pitch to model pitch assumption (x forward, z up):
    # image y-down positive angle means physically downward-forward -> negative z slope
    model_pitch_deg = -abs(est_angle)

    # confidence heuristic
    confidence = float(max(0.0, min(1.0, 1.0 - spread / 25.0)))

    # draw overlay
    overlay = img.copy()
    cv2.rectangle(overlay, (x_min, y_min), (w - 1, h - 1), (80, 80, 255), 2)
    for i, c in enumerate(top[:10]):
        x1, y1, x2, y2 = c["line"]
        color = (0, 255, 0) if i == 0 else (0, 180, 255)
        cv2.line(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            overlay,
            f"{c['angle_deg']:.1f}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        overlay,
        f"est(img)={est_angle:.2f} deg, model={model_pitch_deg:.2f} deg, conf={confidence:.2f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_overlay = Path(args.out_overlay)
    out_overlay.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "image": str(image_path),
        "image_size": [int(w), int(h)],
        "roi": {"x_min": x_min, "y_min": y_min},
        "top_candidates": top[:10],
        "selected_sign_cluster": selected_sign,
        "estimated_image_angle_deg": est_angle,
        "estimated_model_pitch_deg": model_pitch_deg,
        "spread_deg": spread,
        "confidence": confidence,
        "note": "model pitch uses x-forward,z-up convention and assumes downward-forward support frame",
    }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    cv2.imwrite(str(out_overlay), overlay)

    if args.update_dims:
        dims_path = Path(args.dims_json)
        with dims_path.open("r", encoding="utf-8") as f:
            dims = json.load(f)
        dims.setdefault("assumptions", {})
        dims["assumptions"]["foot_support_frame_pitch_deg"] = float(round(model_pitch_deg, 2))
        dims["assumptions"]["foot_support_frame_pitch_confidence"] = float(round(confidence, 3))
        dims["assumptions"]["foot_support_frame_pitch_source"] = "image_bal1_auto"
        with dims_path.open("w", encoding="utf-8") as f:
            json.dump(dims, f, ensure_ascii=False, indent=2)

    print(f"saved: {out_json}")
    print(f"saved: {out_overlay}")
    print(f"estimated_model_pitch_deg={model_pitch_deg:.3f}, confidence={confidence:.3f}")


if __name__ == "__main__":
    main()
