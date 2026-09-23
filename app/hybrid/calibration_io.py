"""Hybrid calibration files. All translations and object points are in cm."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
import numpy as np
from app.hybrid.checkerboard import Intrinsics
from app.hybrid.paths import calibration_root

FILES = ("c0.dat", "c1.dat", "rot_trans_c0.dat", "rot_trans_c1.dat")


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def mac_identity(index):
    """Mac のカメラの識別子（内部パラメータのキャッシュの鍵に使う）。

    OpenCV のカメラ番号は安定しない（Camo や iPhone の連係カメラで入れ替わる）ので、
    この鍵が合っても別のカメラのものでありうる。校正ランナーは、キャッシュを今回の画像に
    当てはめて誤差が大きければ捨てる（``hybrid_calibrate.CACHE_TOLERANCE_PX``）。
    """
    model = platform.machine()
    if platform.system() == "Darwin":
        model = subprocess.check_output(["sysctl", "-n", "hw.model"], text=True).strip()
    return f"{model}:camera{index}"


def cache_key(kind, identity, size):
    if not identity:
        raise ValueError("端末識別子がないため内部パラメータを共有できません")
    return hashlib.sha256(
        json.dumps([kind, identity, list(size)]).encode()
    ).hexdigest()[:24]


def save_intrinsics(key, value, *, root=None):
    directory = Path(root or calibration_root()) / "intrinsics"
    directory.mkdir(parents=True, exist_ok=True)
    write_json(
        directory / f"{key}.json",
        dict(
            K=value.K.tolist(),
            distortion=value.distortion.tolist(),
            size=value.size,
            rms=value.rms,
        ),
    )


def load_intrinsics(key, *, root=None):
    path = Path(root or calibration_root()) / "intrinsics" / f"{key}.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    return Intrinsics(
        np.array(value["K"]),
        np.array(value["distortion"]),
        tuple(value["size"]),
        value["rms"],
    )


def save_calibration(i0, i1, stereo, board, *, cameras, root=None):
    root = Path(root or calibration_root())
    directory = root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True)
    for index, (intrinsic, r, t) in enumerate(
        ((i0, np.eye(3), np.zeros((3, 1))), (i1, stereo.R, stereo.T))
    ):
        with (directory / f"c{index}.dat").open("w") as stream:
            stream.write("intrinsic:\n")
            np.savetxt(stream, intrinsic.K)
            stream.write("distortion:\n")
            np.savetxt(stream, intrinsic.distortion.reshape(1, -1))
        with (directory / f"rot_trans_c{index}.dat").open("w") as stream:
            stream.write("R:\n")
            np.savetxt(stream, r)
            stream.write("T:\n")
            np.savetxt(stream, np.asarray(t).reshape(3, 1))
    metadata = dict(
        units="cm",
        board=asdict(board),
        created_at=datetime.now(timezone.utc).isoformat(),
        cameras=[
            dict(
                c,
                width=i.size[0],
                height=i.size[1],
                rms=i.rms,
                intrinsics_source=c.get("intrinsics_source", "new"),
            )
            for c, i in zip(cameras, (i0, i1))
        ],
        stereo=dict(
            rms=stereo.rms,
            pairs=len(stereo.used_indices),
            baseline_cm=float(np.linalg.norm(stereo.T)),
            square_error_mm=stereo.square_error_mm,
        ),
    )
    write_json(directory / "meta.json", metadata)
    write_json(root / "latest.json", {"directory": directory.name})
    return directory


@dataclass
class Calibration:
    directory: Path
    meta: dict
    intrinsics: tuple[Intrinsics, Intrinsics]
    projections: tuple[np.ndarray, np.ndarray]


def load_calibration(directory="latest", *, root=None):
    root = Path(root or calibration_root())
    directory = Path(directory)
    if str(directory) == "latest":
        directory = root / json.loads((root / "latest.json").read_text())["directory"]
    meta = json.loads((directory / "meta.json").read_text())
    return _calibration_from(directory, meta)


def load_session_calibration(session):
    """計測フォルダ（Recorder が校正ファイル 4 つと ``calibration_meta`` を写したもの）から校正を読む。

    計測に使った校正そのものなので、校正をやり直した後や別の PC でも、記録から三角測量し直せる。
    """
    session = Path(session)
    meta = json.loads((session / "meta.json").read_text(encoding="utf-8"))
    return _calibration_from(session, meta["calibration_meta"])


def _calibration_from(directory, meta):
    from utils import read_camera_parameters, get_projection_matrix

    if meta.get("units") != "cm":
        raise ValueError("混成校正の T は cm で保存されている必要があります")
    intrinsics = []
    for i, camera in enumerate(meta["cameras"]):
        k, d = read_camera_parameters(i, directory)
        intrinsics.append(
            Intrinsics(k, d, (camera["width"], camera["height"]), camera["rms"])
        )
    if len(intrinsics) != 2:
        raise ValueError("校正は2カメラ分必要です")
    return Calibration(
        directory,
        meta,
        tuple(intrinsics),
        tuple(get_projection_matrix(i, False, base_dir=directory) for i in (0, 1)),
    )
