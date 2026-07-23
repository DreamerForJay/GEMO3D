"""Configuration, KITTI I/O, and lightweight shared utilities."""
from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

VEHICLE_TYPES = {"compact", "sedan", "suv", "van"}

# KITTI dimension order: (height, width, length).
VEHICLE_FIXED_DIMS_HWL = {
    "compact": (1.47, 1.72, 3.95),
    "sedan": (1.52, 1.64, 3.86),
    "suv": (1.65, 1.78, 4.40),
    "van": (2.05, 1.92, 5.10),
}

# Repository defaults. Prefer overriding these through CLI options.
DEFAULT_VEHICLE_COMP_PKLS = {
    "compact": "/home/e114/Desktop/mono3d/comp/15_1.65_minicooper.pkl",
    "sedan": "/home/e114/Desktop/mono3d/comp/14_1.65_skoda_superb.pkl",
    "suv": "/home/e114/Desktop/mono3d/comp/16_1.65_bmw_x5m_SUV.pkl",
    "van": "/home/e114/Desktop/mono3d/comp/3_vellfire.pkl",
}

# Backward-compatible aliases used by the original implementation.
VT5 = VEHICLE_TYPES
VT5_FIXED_DIMS_HWL = VEHICLE_FIXED_DIMS_HWL
VT5_COMP_PKL = DEFAULT_VEHICLE_COMP_PKLS

def _veh_label_of_k(pred_by_k: dict, k: int) -> str:
    v = pred_by_k.get(k, {})
    lab = str(v.get("veh_label", "")).strip().lower()
    return lab if lab in VT5 else "unknown"


def resolve_fixed_dims_hwl_for_k(pred_by_k: dict, k: int, default_hwl: tuple) -> tuple:
    lab = _veh_label_of_k(pred_by_k, k)
    if lab in VT5_FIXED_DIMS_HWL:
        return VT5_FIXED_DIMS_HWL[lab]
    return default_hwl


def _numpy_pickle_compat_shim():
    """
    Allow loading pickles created with NumPy 2.x (which may reference numpy._core.*)
    in an environment that only has NumPy 1.x (numpy.core.*).
    """
    import sys
    import importlib
    try:
        import numpy.core as _np_core
        sys.modules.setdefault("numpy._core", _np_core)

        subs = (
            "multiarray",
            "_multiarray_umath",
            "numeric",
            "umath",
            "overrides",
            "_methods",
            "_dtype",
            "_exceptions",
            "fromnumeric",
            "shape_base",
        )
        for s in subs:
            try:
                m = importlib.import_module(f"numpy.core.{s}")
            except Exception:
                continue
            sys.modules.setdefault(f"numpy._core.{s}", m)
    except Exception:
        # If numpy itself is broken/unavailable, do nothing here.
        pass


def _parse_vt_comp_pkl_map(map_text: str) -> dict:
    """
    Parse a per-vehicle compensation-model map.

    Format examples:
      compact=/path/min.pkl,sedan=/path/sed.pkl,suv=/path/suv.pkl,van=/path/van.pkl
      compact:/path/min.pkl; sedan:/path/sed.pkl

    Values "", "none", "null", "-" disable that type and make it fall back to --comp_pkl.
    """
    out = {}
    txt = str(map_text or "").strip()
    if not txt:
        return out
    for item in txt.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            k, v = item.split("=", 1)
        elif ":" in item:
            k, v = item.split(":", 1)
        else:
            print(f"[vehclf] WARN: ignore bad --vt_comp_pkl_map item: {item}")
            continue
        lab = k.strip().lower()
        path = v.strip()
        if lab not in VT5:
            print(f"[vehclf] WARN: unknown vehicle type in --vt_comp_pkl_map: {lab}")
            continue
        out[lab] = "" if path.lower() in ("", "none", "null", "nil", "-") else path
    return out


def build_comp_models_by_vt5(comp_pkl_paths: Optional[dict] = None) -> dict:
    import pickle
    from pathlib import Path
    d = {}

    # Backward compatible default: keep using the original hard-coded VT5_COMP_PKL.
    # CLI overrides only replace the specified labels.
    paths = dict(VT5_COMP_PKL)
    if comp_pkl_paths:
        for lab, p in comp_pkl_paths.items():
            if lab in VT5:
                paths[lab] = p

    for lab, p in paths.items():
        if not p:
            continue
        fp = Path(str(p)).expanduser()
        if not fp.is_file():
            print(f"[vehclf] WARN: comp pkl not found for {lab}: {p}")
            continue
        try:
            with open(fp, "rb") as f:
                _numpy_pickle_compat_shim()
                d[lab] = pickle.load(f)
            from .compensation import model_type_name
            mt = model_type_name(d[lab]) if isinstance(d[lab], dict) else type(d[lab]).__name__
            print(f"[vehclf] loaded comp pkl for {lab}: {p} ({mt})")
        except Exception as e:
            print(f"[vehclf] WARN: failed to load comp pkl for {lab}: {p} ({e})")
    return d


def resolve_comp_model_for_k(pred_by_k: dict, k: int, comp_models_by_type: dict, default_comp_model):
    lab = _veh_label_of_k(pred_by_k, k)
    if lab in comp_models_by_type:
        return comp_models_by_type[lab]
    return default_comp_model


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def read_split_ids(split_path: Path) -> List[str]:
    ids = []
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # accept "000123" or "000123.png"
            s = Path(s).stem
            ids.append(s)
    return ids


def read_kitti_calib_P2(calib_path: Path) -> np.ndarray:
    """
    KITTI calib txt contains lines like:
      P2: fx 0 cx tx  0 fy cy ty  0 0 1 0
    Return P2 as 3x4 float64.
    """
    P2 = None
    with open(calib_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("P2:") or line.startswith("P2 "):
                vals = line.split(":", 1)[1].strip().split()
                arr = np.array([float(v) for v in vals], dtype=np.float64).reshape(3, 4)
                P2 = arr
                break
    if P2 is None:
        raise ValueError(f"Cannot find P2 in calib file: {calib_path}")
    return P2


def K_from_P2(P2: np.ndarray) -> Dict[str, float]:
    fx = float(P2[0, 0])
    fy = float(P2[1, 1])
    cx = float(P2[0, 2])
    cy = float(P2[1, 2])
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}


def backproject_uv_depth(u: float, v: float, z: float, P2: np.ndarray) -> Tuple[float, float, float]:
    """
    Back-project pixel (u,v) with depth z into KITTI camera coords, using P2 (3x4).
    Camera coords: X right, Y down, Z forward.
    """
    fx = float(P2[0, 0]); fy = float(P2[1, 1])
    cx = float(P2[0, 2]); cy = float(P2[1, 2])
    Tx = float(P2[0, 3]); Ty = float(P2[1, 3])

    x = (u * z - cx * z - Tx) / fx
    y = (v * z - cy * z - Ty) / fy
    return float(x), float(y), float(z)


def alpha_from_xz_ry(x: float, z: float, ry: float) -> float:
    theta_ray = math.atan2(x, z)
    return wrap_pi(ry - theta_ray)


def theta_ray_from_u(u: float, K: Dict[str, float]) -> float:
    # ray angle in XZ plane, consistent with KITTI definition
    return math.atan2((u - K["cx"]) / K["fx"], 1.0)


def resolve_dims(p, dims_source: str, fixed_dims_hwl):
    if dims_source in ("egonet", "deep3d") and p is not None and ("dims" in p):
        # p["dims"] is expected to be (h,w,l) in this pipeline
        h, w, l = p["dims"]
        return float(h), float(w), float(l)
    else:
        return fixed_dims_hwl


def resolve_ry(p, theta_ray: float, allow_alpha_fallback: bool = True) -> float:
    """
    Resolve KITTI rotation_y (ry) from a prediction dict.

    Rule:
      1) If alpha exists, ALWAYS recompute ry = alpha + current theta_ray.
         This keeps ry consistent with the ray actually used by this pipeline.
      2) Only fallback to raw ry / yaw if alpha is unavailable.
    """
    if p is None:
        return 0.0

    if isinstance(p, dict):
        if "alpha" in p:
            return wrap_pi(float(p["alpha"]) + float(theta_ray))
        if "ry" in p:
            return wrap_pi(float(p["ry"]))
        if "yaw" in p:
            return wrap_pi(float(p["yaw"]))

    return 0.0


def apply_ry_offset(ry: float, ry_offset_deg: float) -> float:
    """Add a global offset (deg) to ry and wrap to [-pi, pi)."""
    return wrap_pi(float(ry) + math.radians(float(ry_offset_deg)))

from dataclasses import dataclass


@dataclass
class KittiObj:
    cls: str
    trunc: float
    occ: int
    alpha: float

    x1: float
    y1: float
    x2: float
    y2: float

    h: float
    w: float
    l: float

    x: float
    y: float
    z: float
    ry: float

def read_kitti_label_2(label_path: Path, keep_cls: str = "Car") -> List[KittiObj]:
    objs: List[KittiObj] = []
    if (label_path is None) or (not label_path.is_file()):
        return objs
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 15:
                continue
            cls = parts[0]
            if cls != keep_cls:
                continue
            trunc = float(parts[1]); occ = int(float(parts[2]))
            alpha = float(parts[3])
            x1, y1, x2, y2 = map(float, parts[4:8])
            h, w, l = map(float, parts[8:11])
            x, y, z = map(float, parts[11:14])
            ry = float(parts[14])
            objs.append(KittiObj(cls, trunc, occ, alpha, x1, y1, x2, y2, h, w, l, x, y, z, ry))
    return objs


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    area_b = max(0.0, bx2-bx1) * max(0.0, by2-by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 1e-9 else 0.0
