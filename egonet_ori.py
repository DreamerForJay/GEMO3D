"""
egonet_ori.py

Adapter for using the official EgoNet implementation (https://github.com/Nicholasli1995/EgoNet)
as an orientation head inside the mono3d nearest_time_kitti_infer pipeline.

This module exposes two functions that are used by nearest_time_kitti_infer.py:

    - load_head(weights_dir: str) -> None
        Initialise EgoNet with pre-trained weights.

    - infer_batch(image_bgr: np.ndarray,
                  dets_xyxy_cls: list[tuple[float,float,float,float,str,float]],
                  P_or_K: np.ndarray) -> list[dict]
        Run EgoNet on one RGB image and a list of 2D detections and
        return per-detection orientation predictions.

By default, only orientation (ry / alpha) is exported.

Optional (NO tight-fit):
    If env EGONET_EXPORT_DIMS=1, this adapter will also export dims=[h,w,l]
    estimated from EgoNet's predicted 3D cuboid keypoints, following the
    official EgoNet get_template() logic (measure 12 edges; average groups).

    - If dims are invalid, returns [0,0,0] unless EGONET_DIMS_FALLBACK_HWL is set.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

# Global state (initialised in load_head)
_EGONET_MODEL = None          # type: ignore[assignment]
_EGONET_CFG: Dict[str, Any] | None = None
_EGONET_DEVICE: torch.device | None = None
_EGONET_ALPHA_MODE: str | None = None

# Simple in-memory image cache used by the monkey-patched EgoNet.load_cv2
_EGONET_IMG_CACHE: Dict[str, np.ndarray] = {}


# ---------------------------------------------------------------------------
# Utility: resolve EgoNet repo on sys.path
# ---------------------------------------------------------------------------

def _ensure_egonet_on_path() -> None:
    """
    Make sure 'libs.model.egonet' from the official EgoNet repo is importable.

    Priority:
        1) Respect EGONET_ROOT environment variable.
        2) Look for EgoNet-master / EgoNet next to this file.
        3) If 'libs' is already importable, assume it already points to EgoNet.
    """
    try:
        import libs.model.egonet  # type: ignore[unused-import]
        return
    except Exception:
        pass

    here = Path(__file__).resolve().parent

    candidates: List[Path] = []

    env_root = os.environ.get("EGONET_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    for name in ("EgoNet-master", "EgoNet", "egonet"):
        candidates.append(here / name)
        candidates.append(here.parent / name)

    for root in candidates:
        if not root:
            continue
        root = root.expanduser().resolve()
        if (root / "libs" / "model" / "egonet.py").is_file():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            try:
                import libs.model.egonet  # type: ignore[unused-import]
                return
            except Exception:
                continue

    try:
        import libs.model.egonet  # type: ignore[unused-import]  # noqa: F401
        return
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Cannot import 'libs.model.egonet'. "
            "Please install the official EgoNet repo as a package, "
            "or set EGONET_ROOT to the EgoNet repository root."
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_head(weights_dir: str) -> None:
    """
    Initialise the official EgoNet model.

    Parameters
    ----------
    weights_dir:
        Directory containing the EgoNet checkpoints, i.e. files like
        'HC.pth', 'L.pth', 'LS.npy'. This is typically the directory that
        was previously passed as --egonet_weights.
    """
    global _EGONET_MODEL, _EGONET_CFG, _EGONET_DEVICE, _EGONET_ALPHA_MODE

    _ensure_egonet_on_path()

    # Delayed imports that depend on EgoNet being on sys.path
    from libs.model.egonet import EgoNet  # type: ignore[import]
    from libs.arguments.parse import read_yaml_file  # type: ignore[import]
    import types

    weights_dir = os.path.abspath(os.path.expanduser(weights_dir))

    # ------------------------------------------------------------------
    # Locate a config YAML
    #   1) Respect EGONET_CFG env var.
    #   2) Try EGONET_ROOT/configs/...
    #   3) Fallback to libs.__file__ / current path guessing.
    # ------------------------------------------------------------------
    cfg_path: str | None = os.environ.get("EGONET_CFG")

    if cfg_path is None:
        repo_root: Path | None = None

        egonet_root_env = os.environ.get("EGONET_ROOT")
        if egonet_root_env:
            rr = Path(egonet_root_env).expanduser()
            if (rr / "configs").is_dir():
                repo_root = rr.resolve()

        if repo_root is None:
            try:
                import libs  # type: ignore[import]
                lf = getattr(libs, "__file__", None)
                if lf:
                    libs_dir = Path(lf).resolve().parent
                    repo_root = libs_dir.parent
            except Exception:
                repo_root = None

        if repo_root is None:
            repo_root = Path(__file__).resolve().parents[1]

        for name in ("KITTI_inference:test_submission.yml", "KITTI_inference:demo.yml"):
            candidate = repo_root / "configs" / name
            if candidate.is_file():
                cfg_path = str(candidate)
                break

    if cfg_path is None:
        raise FileNotFoundError(
            "Could not find EgoNet config YAML. "
            "Set EGONET_CFG to something like "
            "'/path/to/EgoNet-master/configs/KITTI_inference:test_submission.yml'."
        )

    cfgs: Dict[str, Any] = read_yaml_file(cfg_path)  # type: ignore[assignment]

    cfgs.setdefault("dirs", {})
    cfgs["dirs"]["ckpt"] = weights_dir

    out_dir = cfgs["dirs"].get("output")
    if not out_dir or "YOUR_OUTPUT_DIR" in str(out_dir):
        cfgs["dirs"]["output"] = os.path.join(weights_dir, "egonet_results")

    use_gpu = torch.cuda.is_available()
    cfgs["use_gpu"] = bool(use_gpu)
    device = torch.device("cuda" if use_gpu else "cpu")

    model = EgoNet(cfgs, pre_trained=True)
    model.eval()
    model.to(device)

    testing_settings = cfgs.get("testing_settings", {}) or {}
    alpha_mode = testing_settings.get("alpha_mode", "proj")

    try:
        import torchvision.transforms as T  # type: ignore[import]

        ds_cfg = cfgs.get("dataset", {}) or {}
        pth_t = ds_cfg.get("pth_transform", {}) or {}
        mean = pth_t.get("mean", [0.485, 0.456, 0.406])
        std = pth_t.get("std", [0.229, 0.224, 0.225])

        model.pth_trans = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )
    except Exception:
        model.pth_trans = None  # type: ignore[attr-defined]

    # Monkey-patch load_cv2 so that EgoNet reads from our in-memory images
    def _load_cv2_inmem(self, path: str, rgb: bool = True) -> np.ndarray:  # type: ignore[override]
        import cv2

        arr = _EGONET_IMG_CACHE.get(path)
        if arr is None:
            if rgb:
                data = cv2.cvtColor(
                    cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
                )
            else:
                data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            return data

        if rgb:
            if arr.ndim == 3 and arr.shape[2] == 3:
                return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            return arr
        else:
            return arr

    model.load_cv2 = types.MethodType(_load_cv2_inmem, model)  # type: ignore[assignment]

    _EGONET_MODEL = model
    _EGONET_CFG = cfgs
    _EGONET_DEVICE = device
    _EGONET_ALPHA_MODE = alpha_mode

    print(f"[egonet_ori] EgoNet initialised on {device}, cfg='{cfg_path}'")


def _as_numpy(mat: Any) -> np.ndarray:
    return np.asarray(mat, dtype=np.float32)


def _extract_K(P_or_K: Any) -> np.ndarray:
    """
    Convert the given camera matrix to a 3x3 intrinsic matrix K.
    Accepts either:
        - 3x4 projection matrix P
        - 3x3 intrinsic matrix K
    """
    K = _as_numpy(P_or_K)
    if K.shape == (3, 4):
        K = K[:, :3]
    if K.shape != (3, 3):
        raise ValueError(f"P_or_K must be 3x4 or 3x3, got shape {K.shape}")
    return K


def _parse_fallback_dims_hwl() -> List[float] | None:
    """
    Optional fallback dims if EgoNet dims are invalid.
    Env: EGONET_DIMS_FALLBACK_HWL="1.52,1.64,3.86"
    """
    s = os.environ.get("EGONET_DIMS_FALLBACK_HWL")
    if not s:
        return None
    try:
        parts = [float(x.strip()) for x in s.split(",")]
        if len(parts) != 3:
            return None
        h, w, l = parts
        if h > 0 and w > 0 and l > 0:
            return [h, w, l]
        return None
    except Exception:
        return None


def _estimate_dims_from_record(record: Dict[str, Any], n: int) -> List[List[float]] | None:
    """
    Estimate dims from EgoNet predicted 3D keypoints (NO tight-fit).
    Follows official EgoNet get_template():
        - Use interp_dict['bbox12'] to pick 12 edges
        - First 4 edges -> height
        - Next 4 edges -> length
        - Last 4 edges -> width
    Returns dims in [h,w,l] order (to match KITTI label and your fixed_dims_hwl arg).
    """
    if "kpts_3d_pred" not in record:
        return None

    kpts = np.asarray(record["kpts_3d_pred"], dtype=np.float32)
    if kpts.ndim == 3 and kpts.shape[-1] == 3:
        kpts = kpts.reshape(kpts.shape[0], -1)  # (N, 3P)

    if kpts.ndim != 2:
        return None

    if kpts.shape[0] < n:
        n = kpts.shape[0]

    # Import official interp_dict (same as EgoNet does internally)
    try:
        from libs.dataset.KITTI.car_instance import interp_dict  # type: ignore[import]
    except Exception:
        return None

    try:
        pidx_1based, cidx_1based = interp_dict["bbox12"]  # arrays/lists, 1-based
        pidx = np.asarray(pidx_1based, dtype=np.int64) - 1
        cidx = np.asarray(cidx_1based, dtype=np.int64) - 1
    except Exception:
        return None

    dims_out: List[List[float]] = []
    for i in range(n):
        flat = kpts[i]
        if flat.size % 3 != 0:
            dims_out.append([0.0, 0.0, 0.0])
            continue

        pts = flat.reshape(-1, 3)  # (P,3)
        P = pts.shape[0]

        # Ensure indices are in range
        if pidx.max(initial=-1) >= P or cidx.max(initial=-1) >= P:
            dims_out.append([0.0, 0.0, 0.0])
            continue

        parents = pts[pidx, :]    # (12,3)
        children = pts[cidx, :]   # (12,3)
        lines = parents - children
        lines = np.sqrt(np.sum(lines * lines, axis=1))  # (12,)

        h = float(np.sum(lines[:4]) / 4.0)     # height
        l = float(np.sum(lines[4:8]) / 4.0)    # length
        w = float(np.sum(lines[8:12]) / 4.0)   # width


        # Sanity
        if not (np.isfinite(h) and np.isfinite(w) and np.isfinite(l)):
            dims_out.append([0.0, 0.0, 0.0])
            continue
        if h <= 0 or w <= 0 or l <= 0:
            dims_out.append([0.0, 0.0, 0.0])
            continue

        dims_out.append([h, w, l])

    return dims_out


def infer_batch(
    image_bgr: np.ndarray,
    dets_xyxy_cls: Sequence[Tuple[float, float, float, float, str, float]],
    P_or_K: Any,
) -> List[Dict[str, Any]]:
    """
    Run EgoNet orientation prediction on a single image.

    Returns list of dicts per detection:
        - 'alpha', 'yaw', 'ry'
        - optional 'dims'=[h,w,l] if EGONET_EXPORT_DIMS=1 and available
    """
    if _EGONET_MODEL is None or _EGONET_DEVICE is None:
        raise RuntimeError("EgoNet head not initialised. Call load_head() first.")

    model = _EGONET_MODEL
    device = _EGONET_DEVICE
    alpha_mode = _EGONET_ALPHA_MODE or "proj"

    dets = list(dets_xyxy_cls)
    if len(dets) == 0:
        return []

    export_dims = os.environ.get("EGONET_EXPORT_DIMS", "0").strip() == "1"
    fallback_dims = _parse_fallback_dims_hwl()

    K = _extract_K(P_or_K)

    # Build annot_dict for EgoNet.forward()
    img_key = "egonet_inmem_0"
    _EGONET_IMG_CACHE.clear()
    _EGONET_IMG_CACHE[img_key] = np.asarray(image_bgr, dtype=np.uint8)

    boxes_list: List[List[float]] = []
    scores_list: List[float] = []
    labels_list: List[int] = []

    for (x1, y1, x2, y2, cls_name, score) in dets:
        boxes_list.append([float(x1), float(y1), float(x2), float(y2)])
        scores_list.append(float(score))
        labels_list.append(0)  # EgoNet car label

    boxes = np.asarray(boxes_list, dtype=np.float32)
    scores = np.asarray(scores_list, dtype=np.float32)
    labels = np.asarray(labels_list, dtype=np.int32)

    annot_dict: Dict[str, Any] = {
        "path": [img_key],
        "boxes": [boxes],
        "scores": [scores],
        "labels": [labels],
        "K": [K.astype(np.float32)],
    }

    # Forward + post_process (official)
    records = model(annot_dict)  # type: ignore[call-arg]

    color_dict = {
        "bbox_2d": "y",
        "bbox_3d": "y",
        "kpts": ["yx", "y"],
    }
    save_dict = {"flag": False, "save_dir": None}

    records = model.post_process(  # type: ignore[call-arg]
        records,
        save_dict=save_dict,
        color_dict=color_dict,
        alpha_mode=alpha_mode,
        visualize=False,
    )

    record = records[img_key]

    euler_angles = np.asarray(record["euler_angles"], dtype=np.float32)  # (N, 3)
    alphas = np.asarray(record["alphas"], dtype=np.float32).reshape(-1)

    # Optional dims from kpts_3d_pred (NO tight-fit)
    dims_est: List[List[float]] | None = None
    if export_dims:
        dims_est = _estimate_dims_from_record(record, n=len(dets))
        # fallback if needed
        if dims_est is not None and fallback_dims is not None:
            for i in range(min(len(dims_est), len(dets))):
                h, w, l = dims_est[i]
                if h <= 0 or w <= 0 or l <= 0:
                    dims_est[i] = fallback_dims

    preds: List[Dict[str, Any]] = []
    for idx, (x1, y1, x2, y2, cls_name, score) in enumerate(dets):
        if idx >= euler_angles.shape[0]:
            break

        yaw = float(euler_angles[idx, 1])
        alpha = float(alphas[idx])

        u_center = 0.5 * (float(x1) + float(x2))
        theta_ray = math.atan2(u_center - float(K[0, 2]), float(K[0, 0]))

        if export_dims and dims_est is not None and idx < len(dims_est):
            dims_hwl = dims_est[idx]
        else:
            dims_hwl = [0.0, 0.0, 0.0]

        preds.append(
            {
                "cls": cls_name,
                "score": float(score),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "alpha": alpha,
                "theta_ray": theta_ray,
                "yaw": yaw,
                "ry": yaw,
                "dims": [float(dims_hwl[0]), float(dims_hwl[1]), float(dims_hwl[2])],
            }
        )

    return preds
