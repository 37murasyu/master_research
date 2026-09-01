import argparse
import csv
import json
import math
from pathlib import Path


def inch_to_mm(value: float) -> float:
    return value * 25.4


def build_points(dims: dict) -> tuple[dict, list[tuple[str, str]], dict]:
    rear_d = inch_to_mm(float(dims["rear_wheel_diameter_inch"]))
    front_d = inch_to_mm(float(dims["front_wheel_diameter_inch"]))
    rear_r = rear_d * 0.5
    front_r = front_d * 0.5

    width = float(dims["overall_width"])
    seat_w = float(dims["seat_back_width"])
    wheelbase = float(dims["wheelbase_front_center_to_rear_center"])
    seat_rear_h = float(dims["seat_rear_height_from_ground"])
    seat_front_h = float(dims["seat_front_height_from_ground"])
    seat_depth = float(dims["armrest_length_from_back"])
    armrest_h = float(dims["armrest_height_above_seat"])
    overall_h = float(dims["overall_height"])

    assumptions = dims.get("assumptions", {})
    armrest_y = float(assumptions.get("armrest_y_offset_from_center", seat_w * 0.5))
    foot_pitch_deg = float(assumptions.get("foot_support_frame_pitch_deg", -35.0))
    foot_len = float(dims["foot_support_al_frame_length"])

    # Coordinate: origin at rear axle midpoint
    # x: forward, y: left, z: up
    rear_axle_mid = (0.0, 0.0, rear_r)
    front_axle_mid = (wheelbase, 0.0, front_r)

    points = {
        "rear_axle_mid": rear_axle_mid,
        "rear_wheel_center_R": (0.0, -width * 0.5, rear_r),
        "rear_wheel_center_L": (0.0, width * 0.5, rear_r),
        "front_wheel_center_R": (wheelbase, -width * 0.5, front_r),
        "front_wheel_center_L": (wheelbase, width * 0.5, front_r),
        "seat_rear_mid": (0.0, 0.0, seat_rear_h),
        "seat_front_mid": (seat_depth, 0.0, seat_front_h),
        "seat_rear_R": (0.0, -seat_w * 0.5, seat_rear_h),
        "seat_rear_L": (0.0, seat_w * 0.5, seat_rear_h),
        "seat_front_R": (seat_depth, -seat_w * 0.5, seat_front_h),
        "seat_front_L": (seat_depth, seat_w * 0.5, seat_front_h),
        "backrest_top_mid": (0.0, 0.0, overall_h),
    }

    # Armrest endpoints (using seat rear/front heights + armrest offset)
    points["armrest_rear_R"] = (0.0, -armrest_y, seat_rear_h + armrest_h)
    points["armrest_rear_L"] = (0.0, armrest_y, seat_rear_h + armrest_h)
    points["armrest_front_R"] = (seat_depth, -armrest_y, seat_front_h + armrest_h)
    points["armrest_front_L"] = (seat_depth, armrest_y, seat_front_h + armrest_h)

    # Foot-support aluminum frame from armrest front toward forward/downward at assumed pitch
    pitch = math.radians(foot_pitch_deg)
    dx = foot_len * math.cos(pitch)
    dz = foot_len * math.sin(pitch)

    af_r = points["armrest_front_R"]
    af_l = points["armrest_front_L"]
    points["foot_frame_tip_R"] = (af_r[0] + dx, af_r[1], af_r[2] + dz)
    points["foot_frame_tip_L"] = (af_l[0] + dx, af_l[1], af_l[2] + dz)

    edges = [
        ("rear_wheel_center_R", "front_wheel_center_R"),
        ("rear_wheel_center_L", "front_wheel_center_L"),
        ("rear_wheel_center_R", "rear_wheel_center_L"),
        ("front_wheel_center_R", "front_wheel_center_L"),
        ("seat_rear_R", "seat_front_R"),
        ("seat_rear_L", "seat_front_L"),
        ("seat_rear_R", "seat_rear_L"),
        ("seat_front_R", "seat_front_L"),
        ("seat_rear_mid", "backrest_top_mid"),
        ("armrest_rear_R", "armrest_front_R"),
        ("armrest_rear_L", "armrest_front_L"),
        ("armrest_front_R", "foot_frame_tip_R"),
        ("armrest_front_L", "foot_frame_tip_L"),
    ]

    derived = {
        "rear_wheel_diameter_mm": rear_d,
        "rear_wheel_radius_mm": rear_r,
        "front_wheel_diameter_mm": front_d,
        "front_wheel_radius_mm": front_r,
        "seat_pitch_deg": math.degrees(math.atan2(seat_front_h - seat_rear_h, max(1e-9, seat_depth))),
        "wheel_center_height_diff_mm": rear_r - front_r,
        "foot_support_frame_pitch_deg_assumed": foot_pitch_deg,
    }

    return points, edges, derived


def save_points_csv(points: dict, out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "x_mm", "y_mm", "z_mm"])
        for name, xyz in points.items():
            writer.writerow([name, f"{xyz[0]:.3f}", f"{xyz[1]:.3f}", f"{xyz[2]:.3f}"])


def save_edges_csv(edges: list[tuple[str, str]], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["from", "to"])
        for start, end in edges:
            writer.writerow([start, end])


def save_obj(points: dict, edges: list[tuple[str, str]], out_path: Path) -> None:
    names = list(points.keys())
    index_map = {name: i + 1 for i, name in enumerate(names)}
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Approx wheelchair wireframe generated from dimensions\n")
        for name in names:
            x, y, z = points[name]
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for start, end in edges:
            f.write(f"l {index_map[start]} {index_map[end]}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build approximate 3D wheelchair wireframe from dimensions")
    parser.add_argument("--dims", type=str, default="wheelchair_bal1_dims.json", help="Input dimensions JSON")
    parser.add_argument("--out-prefix", type=str, default="output_data/wheelchair_bal1_approx", help="Output prefix")
    args = parser.parse_args()

    dims_path = Path(args.dims)
    if not dims_path.exists():
        raise FileNotFoundError(f"Dimensions file not found: {dims_path}")

    with dims_path.open("r", encoding="utf-8") as f:
        dims = json.load(f)

    points, edges, derived = build_points(dims)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    points_csv = out_prefix.with_name(out_prefix.name + "_points.csv")
    edges_csv = out_prefix.with_name(out_prefix.name + "_edges.csv")
    obj_path = out_prefix.with_name(out_prefix.name + ".obj")
    meta_json = out_prefix.with_name(out_prefix.name + "_derived.json")

    save_points_csv(points, points_csv)
    save_edges_csv(edges, edges_csv)
    save_obj(points, edges, obj_path)

    with meta_json.open("w", encoding="utf-8") as f:
        json.dump(derived, f, ensure_ascii=False, indent=2)

    print("Generated files:")
    print(f"  {points_csv}")
    print(f"  {edges_csv}")
    print(f"  {obj_path}")
    print(f"  {meta_json}")
    print("Derived:")
    for key, value in derived.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")


if __name__ == "__main__":
    main()
