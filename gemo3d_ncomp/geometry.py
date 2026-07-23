"""Camera projection and 3D-box geometry helpers."""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.special import logsumexp

from .config import KittiObj,iou_xyxy


def _smooth_min(vals: np.ndarray, k: float = 50.0) -> float:
    vals = np.asarray(vals, dtype=np.float64)
    return float(-logsumexp(-float(k) * vals) / float(k))


def _smooth_max(vals: np.ndarray, k: float = 50.0) -> float:
    vals = np.asarray(vals, dtype=np.float64)
    return float(logsumexp(float(k) * vals) / float(k))


def _bbox_from_proj_uv_smooth(uv: np.ndarray, k: float = 50.0) -> tuple[float, float, float, float]:
    xs = uv[:, 0]
    ys = uv[:, 1]
    return _smooth_min(xs, k), _smooth_min(ys, k), _smooth_max(xs, k), _smooth_max(ys, k)

def box3d_corners_cam(h: float, w: float, l: float, x: float, y: float, z: float, ry: float) -> np.ndarray:
    """
    KITTI: (x,y,z) is bottom center in camera coords.
    Camera coords: X right, Y down, Z forward.
    """
    # 8 corners in object coord (bottom at y=0, top at y=-h because y is down)
    # KITTI/Devkit convention: length l along +X, width w along +Z
    x_c = np.array([ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2], dtype=np.float64)
    y_c = np.array([ 0.0,  0.0,  0.0,  0.0, -h,  -h,  -h,  -h ], dtype=np.float64)
    z_c = np.array([ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2], dtype=np.float64)


    c = math.cos(ry); s = math.sin(ry)
    R = np.array([[ c, 0.0,  s],
                  [0.0, 1.0, 0.0],
                  [-s, 0.0,  c]], dtype=np.float64)
    corners = np.stack([x_c, y_c, z_c], axis=0)  # (3,8)
    corners = (R @ corners).T  # (8,3)
    corners += np.array([x, y, z], dtype=np.float64)[None, :]
    return corners


def project_points(P2: np.ndarray, pts3d: np.ndarray) -> np.ndarray:
    """pts3d: (N,3) -> pts2d: (N,2), invalid depth -> NaN"""
    n = pts3d.shape[0]
    pts_h = np.hstack([pts3d, np.ones((n, 1), dtype=np.float64)])  # (N,4)
    proj = (P2 @ pts_h.T).T  # (N,3)
    z = proj[:, 2]

    u = np.full(n, np.nan, dtype=np.float64)
    v = np.full(n, np.nan, dtype=np.float64)

    valid = z > 1e-3  # depth 必須在相機前方
    u[valid] = proj[valid, 0] / z[valid]
    v[valid] = proj[valid, 1] / z[valid]
    return np.stack([u, v], axis=1)


def project_gt_center_uv(best_gt: "KittiObj", P2: np.ndarray, mode: str = "gt_bottom"):
    """
    mode:
      - gt_bottom : project KITTI label location (x,y,z), i.e. 3D box bottom center
      - gt_geom   : project geometric center (x, y-h/2, z)
    return:
      (u, v) or None
    """
    if best_gt is None:
        return None

    mode = str(mode).strip().lower()
    if mode == "gt_geom":
        pts3d = np.array([[best_gt.x, best_gt.y - 0.5 * best_gt.h, best_gt.z]], dtype=np.float64)
    else:
        # default: KITTI location = bottom center
        pts3d = np.array([[best_gt.x, best_gt.y, best_gt.z]], dtype=np.float64)

    uv = project_points(P2, pts3d)[0]
    if not np.all(np.isfinite(uv)):
        return None
    return float(uv[0]), float(uv[1])


def project_gt_heightline_info(best_gt: "KittiObj", P2: np.ndarray):
    """
    Build the GT-centered vertical height line used in vis/2dbbox_h.

    Definition:
      1) project GT geometric center -> (u_gt, v_gt)
      2) use the GT top-center and bottom-center projections, which lie on the same
         image x = u_gt for a standard pinhole camera, to form the vertical height line.

    This is the image-space height that truly passes through the GT 3D geometric-center
    projection, rather than the projected 3D-box outer extent.
    """
    if best_gt is None:
        return None

    pts3d = np.array([
        [best_gt.x, best_gt.y - 0.5 * best_gt.h, best_gt.z],  # geometric center
        [best_gt.x, best_gt.y - 1.0 * best_gt.h, best_gt.z],  # top center
        [best_gt.x, best_gt.y,                  best_gt.z],   # bottom center
    ], dtype=np.float64)
    uv = project_points(P2, pts3d)
    if not np.all(np.isfinite(uv)):
        return None

    u_gt, v_gt = uv[0]
    u_top, v_top = uv[1]
    u_bot, v_bot = uv[2]

    # keep the displayed line strictly on the GT geometric-center x
    u_line = float(u_gt)
    h_px = float(v_bot - v_top)

    return {
        "u_gt": float(u_gt),
        "v_gt": float(v_gt),
        "u_top": float(u_top),
        "v_top": float(v_top),
        "u_bottom": float(u_bot),
        "v_bottom": float(v_bot),
        "u_line": u_line,
        "h_px": h_px,
    }


def find_best_gt_by_iou(pred_bbox: Tuple[float, float, float, float], gt_objs: List["KittiObj"]):
    best_iou_2d = 0.0
    best_gt_2d: Optional[KittiObj] = None
    for g in gt_objs:
        iou = iou_xyxy(pred_bbox, (g.x1, g.y1, g.x2, g.y2))
        if iou > best_iou_2d:
            best_iou_2d = iou
            best_gt_2d = g
    return best_gt_2d, float(best_iou_2d)


def choose_proj_uv(
    mode: str,
    x1: float, y1: float, x2: float, y2: float,
    P2: np.ndarray,
    best_gt: Optional["KittiObj"] = None,
    best_iou: float = 0.0,
    iou_thres: float = 0.5,
):
    """
    Select (u,v) used for theta_ray / backproject_uv_depth.
    """
    mode = str(mode).strip().lower()

    if mode == "bbox_center":
        return 0.5 * (x1 + x2), 0.5 * (y1 + y2), "bbox_center"

    if mode == "gt_bottom":
        if (best_gt is not None) and (float(best_iou) >= float(iou_thres)):
            uv = project_gt_center_uv(best_gt, P2, mode="gt_bottom")
            if uv is not None:
                return uv[0], uv[1], "gt_bottom"
        return 0.5 * (x1 + x2), y2, "bbox_bottom"

    if mode == "gt_geom":
        if (best_gt is not None) and (float(best_iou) >= float(iou_thres)):
            uv = project_gt_center_uv(best_gt, P2, mode="gt_geom")
            if uv is not None:
                return uv[0], uv[1], "gt_geom"
        return 0.5 * (x1 + x2), y2, "bbox_bottom"

    # default: bbox bottom
    return 0.5 * (x1 + x2), y2, "bbox_bottom"


def project_pred_bottom_uv(P2: np.ndarray, x: float, y: float, z: float):
    pts3d = np.array([[float(x), float(y), float(z)]], dtype=np.float64)
    uv = project_points(P2, pts3d)[0]
    if not np.all(np.isfinite(uv)):
        return None
    return float(uv[0]), float(uv[1])


def project_pred_geom_uv(P2: np.ndarray, x: float, y: float, z: float, h: float):
    """
    Project predicted 3D geometric center:
      (x, y - h/2, z)
    """
    pts3d = np.array([[float(x), float(y) - 0.5 * float(h), float(z)]], dtype=np.float64)
    uv = project_points(P2, pts3d)[0]
    if not np.all(np.isfinite(uv)):
        return None
    return float(uv[0]), float(uv[1])


def project_pred_boxctr_uv(
    P2: np.ndarray,
    x: float,
    y: float,
    z: float,
    h: float,
    w: float,
    l: float,
    ry: float,
    k_smooth: float = 50.0,
):
    """
    Project the CURRENT predicted 3D box, then take the center of its projected 2D extent.
    This quantity can change in x with yaw / size / depth, unlike geom-center projection.
    """
    corners3d = box3d_corners_cam(
        float(h), float(w), float(l),
        float(x), float(y), float(z), float(ry)
    )
    uv = project_points(P2, corners3d)
    if not np.all(np.isfinite(uv)):
        return None

    px1, py1, px2, py2 = _bbox_from_proj_uv_smooth(uv, k=float(k_smooth))
    if not all(math.isfinite(v) for v in [px1, py1, px2, py2]):
        return None

    return float(0.5 * (px1 + px2)), float(0.5 * (py1 + py2))


def project_pred_proj_bbox_info(
    P2: np.ndarray,
    x: float,
    y: float,
    z: float,
    h: float,
    w: float,
    l: float,
    ry: float,
    k_smooth: float = 50.0,
):
    """
    Project the predicted 3D box and return its smooth 2D extent.

    This is used for the new vis/2dbbox_h overlay and CSV export:
      - pred_proj_h_px  : projected 3D-box height in image pixels
      - pred_bbox_h_px  : raw 2D bbox height in pixels
      - pred_proj_h_diff_px = pred_proj_h_px - pred_bbox_h_px
    """
    corners3d = box3d_corners_cam(
        float(h), float(w), float(l),
        float(x), float(y), float(z), float(ry)
    )
    uv = project_points(P2, corners3d)
    if not np.all(np.isfinite(uv)):
        return None

    px1, py1, px2, py2 = _bbox_from_proj_uv_smooth(uv, k=float(k_smooth))
    vals = [px1, py1, px2, py2]
    if not all(math.isfinite(v) for v in vals):
        return None

    return {
        "x1": float(px1),
        "y1": float(py1),
        "x2": float(px2),
        "y2": float(py2),
        "cx": float(0.5 * (px1 + px2)),
        "cy": float(0.5 * (py1 + py2)),
        "w_px": float(px2 - px1),
        "h_px": float(py2 - py1),
    }


def project_pred_feedback_uv(
    mode: str,
    P2: np.ndarray,
    x: float,
    y: float,
    z: float,
    h: float,
    w: float,
    l: float,
    ry: float,
    k_smooth: float = 50.0,
):
    """
    mode:
      - pred_reproj_bottom : feedback by predicted bottom-center reprojection
      - pred_reproj_geom_x : feedback u from CURRENT projected 3D-box 2D-center,
                             while keeping v from predicted bottom-center projection.

    Note:
      The old implementation used geometric-center projection u.
      That does NOT really change x because only y was changed.
    """
    mode = str(mode).strip().lower()

    if mode == "pred_reproj_bottom":
        uvb = project_pred_bottom_uv(P2, x, y, z)
        if uvb is None:
            return None
        return float(uvb[0]), float(uvb[1]), "pred_reproj_bottom"

    if mode == "pred_reproj_geom_x":
        uvb = project_pred_bottom_uv(P2, x, y, z)
        uvc = project_pred_boxctr_uv(
            P2=P2,
            x=x, y=y, z=z,
            h=h, w=w, l=l, ry=ry,
            k_smooth=float(k_smooth),
        )
        if (uvb is None) or (uvc is None):
            return None
        # x 用 projected 3D-box center；y 仍用 bottom-center projection
        return float(uvc[0]), float(uvb[1]), "pred_reproj_geom_x"

    return None
