"""OpenCV visualizations and BEV/3D IoU utilities."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import KittiObj, iou_xyxy, wrap_pi
from .geometry import box3d_corners_cam, project_gt_center_uv, project_points


_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

def draw_box3d(img: np.ndarray, P2: np.ndarray, h: float, w: float, l: float,
              x: float, y: float, z: float, ry: float,
              color: tuple[int, int, int], thickness: int = 1):

    corners3d = box3d_corners_cam(h, w, l, x, y, z, ry)
    corners2d = project_points(P2, corners3d)  # float

    # 任何角點無效就不要畫（最保守，保證不炸）
    if (not np.all(np.isfinite(corners2d))):
        return
    # 避免極端值溢位（多半來自不合理深度）
    if np.any(np.abs(corners2d) > 1e6):
        return

    corners2d = corners2d.astype(np.int32)
    for i, j in _EDGES:
        p1 = (int(corners2d[i, 0]), int(corners2d[i, 1]))
        p2 = (int(corners2d[j, 0]), int(corners2d[j, 1]))
        cv2.line(img, p1, p2, color, thickness, lineType=cv2.LINE_AA)


def draw_text_lines(img: np.ndarray, x: int, y: int, lines: List[str],
                    color: Tuple[int,int,int] = (255,255,255),
                    bg: Tuple[int,int,int] = (0,0,0),
                    font_scale: float = 0.4, thickness: int = 1, pad: int = 4):
    font = cv2.FONT_HERSHEY_SIMPLEX
    sizes = [cv2.getTextSize(t, font, font_scale, thickness)[0] for t in lines]
    w = max(s[0] for s in sizes) if sizes else 0
    h = sum(s[1] for s in sizes) + (len(lines)-1)*2
    x1, y1 = x, y
    x2, y2 = x + w + 2*pad, y + h + 2*pad
    cv2.rectangle(img, (x1, y1-90), (x2, y2-90), bg, -1)
    yy = y-90 + pad
    for t, (tw, th) in zip(lines, sizes):
        yy += th
        cv2.putText(img, t, (x+pad, yy), font, font_scale, color, thickness, cv2.LINE_AA)
        yy += 2


def draw_text_panel_top_right(img: np.ndarray, lines: List[str],
                              color: Tuple[int, int, int] = (255, 255, 255),
                              bg: Tuple[int, int, int] = (0, 0, 0),
                              font_scale: float = 0.55, thickness: int = 1,
                              pad: int = 6, margin: int = 12):
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    sizes = [cv2.getTextSize(t, font, font_scale, thickness)[0] for t in lines]
    w = max(s[0] for s in sizes) if sizes else 0
    h = sum(s[1] for s in sizes) + max(0, len(lines) - 1) * 4
    x1 = max(0, int(img.shape[1] - w - 2 * pad - margin))
    y1 = max(0, int(margin))
    x2 = min(int(img.shape[1] - 1), x1 + w + 2 * pad)
    y2 = min(int(img.shape[0] - 1), y1 + h + 2 * pad)
    cv2.rectangle(img, (x1, y1), (x2, y2), bg, -1)
    yy = y1 + pad
    for t, (_tw, th) in zip(lines, sizes):
        yy += th
        cv2.putText(img, t, (x1 + pad, yy), font, font_scale, color, thickness, cv2.LINE_AA)
        yy += 4


def build_pair_overlay_lines(pred_id: int,
                             pred_z: float,
                             comp: float,
                             ry: float,
                             comp_angle_deg: Optional[float] = None,
                             best_gt: Optional["KittiObj"] = None,
                             best_iou: float = 0.0,
                             vis_iou_thres: float = 0.5,
                             pair_bev_iou: Optional[float] = None,
                             pair_3d_iou: Optional[float] = None,
                             vmeta: Optional[dict] = None,
                             center_src: str = "") -> List[str]:
    lines: List[str] = []
    matched = (best_gt is not None) and (float(best_iou) >= float(vis_iou_thres))

    if matched:
        dz = float(pred_z) - float(best_gt.z)
        dry = wrap_pi(float(ry) - float(best_gt.ry))
        lines.append(f"GTz={float(best_gt.z):.2f}m")
        lines.append(f"Predz={float(pred_z):.2f}m")
        lines.append(f"dZ={dz:+.2f}m")
        lines.append(f"cmp={float(comp):+.2f}")
        if comp_angle_deg is not None:
            lines.append(f"compAng={float(comp_angle_deg):+.1f}deg")
        lines.append(f"dRy={math.degrees(dry):+.1f}deg")
        if pair_bev_iou is not None:
            lines.append(f"BEV={float(pair_bev_iou):.3f}")
        if pair_3d_iou is not None:
            lines.append(f"3DIoU={float(pair_3d_iou):.3f}")
    else:
        lines.append("GTz=n/a")
        lines.append(f"Predz={float(pred_z):.2f}m")
        if comp_angle_deg is not None:
            lines.append(f"compAng={float(comp_angle_deg):+.1f}deg")
        lines.append("no mat")

    if center_src:
        lines.append(f"ctr={str(center_src).upper()}")

    if isinstance(vmeta, dict) and ("veh_label" in vmeta):
        raw_lab = str(vmeta.get("veh_raw", "")).strip()
        if raw_lab:
            lines.append(raw_lab)
    return lines


def choose_pred_center_xyz(center_mode: str,
                           x: float,
                           y: float,
                           z: float,
                           best_gt: Optional["KittiObj"] = None,
                           best_iou: float = 0.0,
                           iou_thres: float = 0.5) -> Tuple[float, float, float, str]:
    """Select predicted 3D box center source.

    center_mode:
      - "bbox": keep the center estimated from 2D box backprojection / tight-fit
      - "gt"  : when a GT match exists and IoU>=threshold, overwrite center with GT (x,y,z)
    """
    mode = str(center_mode).strip().lower()
    if mode == "gt" and (best_gt is not None) and (float(best_iou) >= float(iou_thres)):
        return float(best_gt.x), float(best_gt.y), float(best_gt.z), "gt"
    return float(x), float(y), float(z), "bbox"


def _bev_xz_to_pix(x: float, z: float, xlim, zlim, ppm: float) -> tuple[int, int]:
    """Map (x,z) in meters to BEV image pixel (px,py). x->right, z->up."""
    xmin, xmax = float(xlim[0]), float(xlim[1])
    zmin, zmax = float(zlim[0]), float(zlim[1])

    px = int(round((x - xmin) * ppm))
    py = int(round((zmax - z) * ppm))  # z forward is up in BEV
    return px, py


def _bev_box_corners_xz(x: float, z: float, ry: float, L: float, W: float):
    """
    2D corners on X-Z plane for a KITTI box.
    - L along local +X, W along local +Z
    - ry: rotation about camera Y (KITTI)
    """
    dx = 0.5 * float(L)  # length
    dz = 0.5 * float(W)  # width

    # local corners (x,z)
    local = [(+dx, +dz), (+dx, -dz), (-dx, -dz), (-dx, +dz)]

    c = math.cos(float(ry))
    s = math.sin(float(ry))

    corners = []
    for lx, lz in local:
        X = c * lx + s * lz + float(x)
        Z = -s * lx + c * lz + float(z)
        corners.append((X, Z))
    return corners


def draw_bev_local(
    objs: list[dict],
    xlim=(-30.0, 30.0),
    zlim=(0.0, 80.0),
    ppm: float = 10.0,
    img_w: Optional[int] = None,
    fx: Optional[float] = None,
    draw_fov: bool = True,
) -> np.ndarray:
    """
    Draw a simple BEV canvas (no point cloud), overlay rotated rectangles.
    objs item format:
      { "x":..., "z":..., "ry":..., "L":..., "W":..., "cls":"pred"/"gt" }
    """
    xmin, xmax = float(xlim[0]), float(xlim[1])
    zmin, zmax = float(zlim[0]), float(zlim[1])

    Wpx = int(round((xmax - xmin) * ppm))
    Hpx = int(round((zmax - zmin) * ppm))
    bev = np.full((Hpx, Wpx, 3), 255, dtype=np.uint8)

    # grid every 5m
    step = 5.0
    x = math.ceil(xmin / step) * step
    while x <= xmax:
        p1 = _bev_xz_to_pix(x, zmin, xlim, zlim, ppm)
        p2 = _bev_xz_to_pix(x, zmax, xlim, zlim, ppm)
        cv2.line(bev, p1, p2, (230, 230, 230), 1, lineType=cv2.LINE_AA)
        x += step
    z = math.ceil(zmin / step) * step
    while z <= zmax:
        p1 = _bev_xz_to_pix(xmin, z, xlim, zlim, ppm)
        p2 = _bev_xz_to_pix(xmax, z, xlim, zlim, ppm)
        cv2.line(bev, p1, p2, (230, 230, 230), 1, lineType=cv2.LINE_AA)
        z += step

    # ego/camera at (0,0)
    ego = _bev_xz_to_pix(0.0, 0.0, xlim, zlim, ppm)
    cv2.circle(bev, ego, 4, (0, 0, 0), -1, lineType=cv2.LINE_AA)

    # approximate camera FOV wedge using fx and image width
    if draw_fov and (img_w is not None) and (fx is not None) and (fx > 1e-6):
        fov = 2.0 * math.atan((0.5 * float(img_w)) / float(fx))
        a = 0.5 * fov
        far = zmax
        x1 = math.tan(-a) * far
        x2 = math.tan(+a) * far
        p1 = _bev_xz_to_pix(x1, far, xlim, zlim, ppm)
        p2 = _bev_xz_to_pix(x2, far, xlim, zlim, ppm)
        cv2.line(bev, ego, p1, (200, 200, 200), 1, lineType=cv2.LINE_AA)
        cv2.line(bev, ego, p2, (200, 200, 200), 1, lineType=cv2.LINE_AA)

    # draw boxes
    for o in objs:
        cls = str(o.get("cls", "pred"))
        color = (0, 255, 0) if cls == "pred" else (0, 0, 255)

        x = float(o["x"]); z = float(o["z"])
        ry = float(o["ry"])
        L = float(o["L"]); Wm = float(o["W"])

        corners = _bev_box_corners_xz(x, z, ry, L=L, W=Wm)
        pts = np.array([_bev_xz_to_pix(X, Z, xlim, zlim, ppm) for (X, Z) in corners], dtype=np.int32).reshape(-1, 1, 2)

        cv2.polylines(bev, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

        # heading line (front direction: local +X, i.e., length axis)
        c = math.cos(ry); s = math.sin(ry)
        hx = c * (0.5 * L)        # R @ [L/2, 0] -> x
        hz = -s * (0.5 * L)       # R @ [L/2, 0] -> z
        p0 = _bev_xz_to_pix(x, z, xlim, zlim, ppm)
        p1 = _bev_xz_to_pix(x + hx, z + hz, xlim, zlim, ppm)
        cv2.line(bev, p0, p1, color, 2, lineType=cv2.LINE_AA)

    return bev


def bev_iou_rotated_rects(pred_box: dict, gt_box: dict) -> tuple[float, float, float, Optional[np.ndarray]]:
    """Compute BEV IoU on X-Z plane for two rotated rectangles.
    Returns (iou, inter_area, union_area, inter_poly_xy[optional]).
    """
    pred = np.asarray(_bev_box_corners_xz(pred_box["x"], pred_box["z"], pred_box["ry"], pred_box["L"], pred_box["W"]), dtype=np.float32)
    gt = np.asarray(_bev_box_corners_xz(gt_box["x"], gt_box["z"], gt_box["ry"], gt_box["L"], gt_box["W"]), dtype=np.float32)
    area_pred = float(abs(cv2.contourArea(pred)))
    area_gt = float(abs(cv2.contourArea(gt)))
    inter_area = 0.0
    inter_poly = None
    try:
        inter_area, inter_poly = cv2.intersectConvexConvex(pred, gt)
        inter_area = float(inter_area)
    except Exception:
        inter_area = 0.0
        inter_poly = None
    union_area = float(area_pred + area_gt - inter_area)
    iou = float(inter_area / union_area) if union_area > 1e-9 else 0.0
    return iou, inter_area, union_area, inter_poly


def iou3d_kitti_boxes(pred_box: dict, gt_box: dict) -> float:
    pred_poly = np.asarray(_bev_box_corners_xz(pred_box["x"], pred_box["z"], pred_box["ry"], pred_box["l"], pred_box["w"]), dtype=np.float32)
    gt_poly = np.asarray(_bev_box_corners_xz(gt_box["x"], gt_box["z"], gt_box["ry"], gt_box["l"], gt_box["w"]), dtype=np.float32)
    a_pred = float(abs(cv2.contourArea(pred_poly)))
    a_gt = float(abs(cv2.contourArea(gt_poly)))
    inter_area = 0.0
    try:
        inter_area, _ = cv2.intersectConvexConvex(pred_poly, gt_poly)
        inter_area = float(inter_area)
    except Exception:
        inter_area = 0.0
    if inter_area <= 0.0:
        return 0.0

    pred_bottom = float(pred_box["y"])
    pred_top = float(pred_box["y"]) - float(pred_box["h"])
    gt_bottom = float(gt_box["y"])
    gt_top = float(gt_box["y"]) - float(gt_box["h"])
    h_ov = max(0.0, min(pred_bottom, gt_bottom) - max(pred_top, gt_top))
    if h_ov <= 0.0:
        return 0.0

    inter_vol = inter_area * h_ov
    v_pred = float(pred_box["h"]) * float(pred_box["w"]) * float(pred_box["l"])
    v_gt = float(gt_box["h"]) * float(gt_box["w"]) * float(gt_box["l"])
    union = max(v_pred + v_gt - inter_vol, 1e-12)
    return float(inter_vol / union)


def draw_bev_follow_pair(
    pred_box: dict,
    gt_box: dict,
    ppm: float = 28.0,
    pad_m: float = 5.0,
    grid_step_m: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Draw a local top-view that follows the matched GT box center.
    The canvas is centered on the GT box so it behaves like a simple follow camera.
    """
    pred_corners = _bev_box_corners_xz(pred_box["x"], pred_box["z"], pred_box["ry"], pred_box["L"], pred_box["W"])
    gt_corners = _bev_box_corners_xz(gt_box["x"], gt_box["z"], gt_box["ry"], gt_box["L"], gt_box["W"])
    all_x = [p[0] for p in pred_corners + gt_corners]
    all_z = [p[1] for p in pred_corners + gt_corners]

    cx = float(gt_box["x"])
    cz = float(gt_box["z"])
    half_x = max(max(abs(v - cx) for v in all_x) + pad_m, 6.0)
    half_z = max(max(abs(v - cz) for v in all_z) + pad_m, 8.0)

    xlim = (cx - half_x, cx + half_x)
    zlim = (cz - half_z, cz + half_z)
    xmin, xmax = xlim
    zmin, zmax = zlim

    Wpx = max(64, int(round((xmax - xmin) * ppm)))
    Hpx = max(64, int(round((zmax - zmin) * ppm)))
    bev = np.full((Hpx, Wpx, 3), 255, dtype=np.uint8)

    # local grid around tracked GT car
    if grid_step_m > 1e-6:
        x = math.ceil(xmin / grid_step_m) * grid_step_m
        while x <= xmax:
            p1 = _bev_xz_to_pix(x, zmin, xlim, zlim, ppm)
            p2 = _bev_xz_to_pix(x, zmax, xlim, zlim, ppm)
            cv2.line(bev, p1, p2, (236, 236, 236), 1, lineType=cv2.LINE_AA)
            x += grid_step_m
        z = math.ceil(zmin / grid_step_m) * grid_step_m
        while z <= zmax:
            p1 = _bev_xz_to_pix(xmin, z, xlim, zlim, ppm)
            p2 = _bev_xz_to_pix(xmax, z, xlim, zlim, ppm)
            cv2.line(bev, p1, p2, (236, 236, 236), 1, lineType=cv2.LINE_AA)
            z += grid_step_m

    pred_pts_m = np.asarray(pred_corners, dtype=np.float32)
    gt_pts_m = np.asarray(gt_corners, dtype=np.float32)
    pred_pts = np.asarray([_bev_xz_to_pix(X, Z, xlim, zlim, ppm) for (X, Z) in pred_corners], dtype=np.int32).reshape(-1, 1, 2)
    gt_pts = np.asarray([_bev_xz_to_pix(X, Z, xlim, zlim, ppm) for (X, Z) in gt_corners], dtype=np.int32).reshape(-1, 1, 2)

    iou, inter_area, union_area, inter_poly_m = bev_iou_rotated_rects(pred_box, gt_box)

    overlay = bev.copy()
    cv2.fillPoly(overlay, [gt_pts], color=(0, 0, 255), lineType=cv2.LINE_AA)
    cv2.fillPoly(overlay, [pred_pts], color=(0, 255, 0), lineType=cv2.LINE_AA)
    if inter_poly_m is not None and len(inter_poly_m) >= 3:
        inter_poly_m = np.asarray(inter_poly_m, dtype=np.float32).reshape(-1, 2)
        inter_pts = np.asarray([_bev_xz_to_pix(float(X), float(Z), xlim, zlim, ppm) for (X, Z) in inter_poly_m], dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [inter_pts], color=(0, 165, 255), lineType=cv2.LINE_AA)
    bev = cv2.addWeighted(overlay, 0.22, bev, 0.78, 0.0)

    cv2.polylines(bev, [gt_pts], isClosed=True, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.polylines(bev, [pred_pts], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    for box, color in ((gt_box, (0, 0, 255)), (pred_box, (0, 255, 0))):
        c = math.cos(float(box["ry"])); s = math.sin(float(box["ry"]))
        hx = c * (0.5 * float(box["L"]))
        hz = -s * (0.5 * float(box["L"]))
        p0 = _bev_xz_to_pix(float(box["x"]), float(box["z"]), xlim, zlim, ppm)
        p1 = _bev_xz_to_pix(float(box["x"]) + hx, float(box["z"]) + hz, xlim, zlim, ppm)
        cv2.circle(bev, p0, 3, color, -1, lineType=cv2.LINE_AA)
        cv2.line(bev, p0, p1, color, 2, lineType=cv2.LINE_AA)

    tracked = _bev_xz_to_pix(cx, cz, xlim, zlim, ppm)
    cv2.drawMarker(bev, tracked, (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)

    lines = [
        f"BEV IoU={iou:.4f}",
        f"Inter={inter_area:.3f}  Union={union_area:.3f}",
        f"GT xz=({gt_box['x']:+.2f},{gt_box['z']:+.2f}) ry={math.degrees(float(gt_box['ry'])):+.1f}",
        f"PD xz=({pred_box['x']:+.2f},{pred_box['z']:+.2f}) ry={math.degrees(float(pred_box['ry'])):+.1f}",
    ]
    draw_text_lines(bev, 8, 58, lines, color=(255,255,255), bg=(24,24,24), font_scale=0.5, thickness=1)
    return bev, iou


def _kitti_cam_pts_to_carla_sensor_pts(pts_cam: np.ndarray) -> np.ndarray:
    pts_cam = np.asarray(pts_cam, dtype=np.float64)
    xs = pts_cam[:, 2]   # KITTI Z -> CARLA sensor X (forward)
    ys = pts_cam[:, 0]   # KITTI X -> CARLA sensor Y (right)
    zs = -pts_cam[:, 1]  # KITTI Y(down) -> CARLA sensor Z(up)
    return np.stack([xs, ys, zs], axis=1)


def _sensor_pts_to_world_pts(pts_sensor: np.ndarray, W2S: np.ndarray) -> np.ndarray:
    pts_sensor = np.asarray(pts_sensor, dtype=np.float64)
    S2W = np.linalg.inv(np.asarray(W2S, dtype=np.float64))
    pts4 = np.concatenate([pts_sensor, np.ones((pts_sensor.shape[0], 1), dtype=np.float64)], axis=1)
    world = (S2W @ pts4.T).T
    return world[:, :3]


def _world_pts_to_sensor_pts(pts_world: np.ndarray, W2S: np.ndarray) -> np.ndarray:
    pts_world = np.asarray(pts_world, dtype=np.float64)
    pts4 = np.concatenate([pts_world, np.ones((pts_world.shape[0], 1), dtype=np.float64)], axis=1)
    sens = (np.asarray(W2S, dtype=np.float64) @ pts4.T).T
    return sens[:, :3]


def _ground_plane_sensor_from_w2s(front_w2s: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Build ground plane in CARLA sensor frame from world->sensor extrinsic.

    Assumption:
      CARLA world ground plane is Z_world = 0

    Plane in sensor frame:
      n_s^T X_s + d_s = 0
    """
    W2S = np.asarray(front_w2s, dtype=np.float64).reshape(4, 4)
    R = W2S[:3, :3]
    t = W2S[:3, 3]

    n_w = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # world up axis
    d_w = 0.0

    # X_s = R X_w + t
    n_s = R @ n_w
    d_s = float(d_w - (n_s @ t))
    return n_s, d_s


def _ground_y_from_kitti_xz(x: float, z: float, front_w2s: np.ndarray) -> Optional[float]:
    """
    Solve KITTI Y (down-positive) from ground plane, given KITTI X,Z.

    Mapping already used in this file:
      sensor = [X_forward, Y_right, Z_up]
      KITTI  = [X_right,   Y_down,  Z_forward]

    So a KITTI point (x, y, z) corresponds to sensor point:
      p_s = [z, x, -y]

    If ground plane in sensor frame is:
      n_s^T p_s + d_s = 0
    then:
      n_x * z + n_y * x - n_z * y + d = 0
      => y = (n_x*z + n_y*x + d) / n_z
    """
    n_s, d_s = _ground_plane_sensor_from_w2s(front_w2s)

    nz = float(n_s[2])
    if abs(nz) < 1e-9:
        return None

    y = (float(n_s[0]) * float(z) + float(n_s[1]) * float(x) + float(d_s)) / nz

    if not math.isfinite(y):
        return None
    return float(y)


def world_corners_from_kitti_box(h: float, w: float, l: float, x: float, y: float, z: float, ry: float, front_w2s: np.ndarray) -> np.ndarray:
    corners_cam = box3d_corners_cam(h, w, l, x, y, z, ry)
    corners_sensor = _kitti_cam_pts_to_carla_sensor_pts(corners_cam)
    return _sensor_pts_to_world_pts(corners_sensor, front_w2s)


def bev_iou_world_corners(pred_world: np.ndarray, gt_world: np.ndarray) -> tuple[float, float, float, Optional[np.ndarray]]:
    pred = np.asarray(pred_world[:4, :2], dtype=np.float32)
    gt = np.asarray(gt_world[:4, :2], dtype=np.float32)
    area_pred = float(abs(cv2.contourArea(pred)))
    area_gt = float(abs(cv2.contourArea(gt)))
    inter_area = 0.0
    inter_poly = None
    try:
        inter_area, inter_poly = cv2.intersectConvexConvex(pred, gt)
        inter_area = float(inter_area)
    except Exception:
        inter_area = 0.0
        inter_poly = None
    union_area = float(area_pred + area_gt - inter_area)
    iou = float(inter_area / union_area) if union_area > 1e-9 else 0.0
    return iou, inter_area, union_area, inter_poly


def _project_world_pts_to_top_image(pts_world: np.ndarray, top_cam_meta: dict) -> Optional[np.ndarray]:
    if not isinstance(top_cam_meta, dict):
        return None
    K = top_cam_meta.get("intrinsics", None)
    W2S = top_cam_meta.get("extrinsics", None)
    if K is None or W2S is None:
        return None
    K = {k: float(v) for k, v in dict(K).items()}
    W2S = np.asarray(W2S, dtype=np.float64).reshape(4, 4)
    pts_sensor = _world_pts_to_sensor_pts(pts_world, W2S)
    z = pts_sensor[:, 0]
    valid = z > 1e-5
    if not np.all(valid):
        return None
    x_img = pts_sensor[:, 1]
    y_img = -pts_sensor[:, 2]
    u = K["fx"] * (x_img / z) + K["cx"]
    v = K["fy"] * (y_img / z) + K["cy"]
    return np.stack([u, v], axis=1)


def _project_world_pts_to_carla_image(pts_world: np.ndarray, cam_meta: dict) -> Optional[np.ndarray]:
    if not isinstance(cam_meta, dict):
        return None
    K = cam_meta.get("intrinsics", None)
    W2S = cam_meta.get("extrinsics", None)
    if K is None or W2S is None:
        return None
    K = {k: float(v) for k, v in dict(K).items()}
    W2S = np.asarray(W2S, dtype=np.float64).reshape(4, 4)
    pts_sensor = _world_pts_to_sensor_pts(pts_world, W2S)
    z = pts_sensor[:, 0]
    valid = z > 1e-5
    if not np.all(valid):
        return None
    x_img = pts_sensor[:, 1]
    y_img = -pts_sensor[:, 2]
    u = K["fx"] * (x_img / z) + K["cx"]
    v = K["fy"] * (y_img / z) + K["cy"]
    return np.stack([u, v], axis=1)


def _convex_hull_from_uv(uv: np.ndarray) -> Optional[np.ndarray]:
    if uv is None:
        return None
    arr = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
    if arr.shape[0] < 3 or (not np.all(np.isfinite(arr))):
        return None
    hull = cv2.convexHull(arr.reshape(-1, 1, 2))
    if hull is None or len(hull) < 3:
        return None
    return hull.astype(np.float32)


def _camera_hull_iou(pred_hull: np.ndarray, gt_hull: np.ndarray) -> tuple[float, float, float, Optional[np.ndarray]]:
    area_pred = float(abs(cv2.contourArea(pred_hull)))
    area_gt = float(abs(cv2.contourArea(gt_hull)))
    inter_area = 0.0
    inter_poly = None
    try:
        inter_area, inter_poly = cv2.intersectConvexConvex(pred_hull, gt_hull)
        inter_area = float(inter_area)
    except Exception:
        inter_area = 0.0
        inter_poly = None
    union_area = float(area_pred + area_gt - inter_area)
    iou = float(inter_area / union_area) if union_area > 1e-9 else 0.0
    return iou, inter_area, union_area, inter_poly


def _crop_from_polys(img: np.ndarray, polys: List[np.ndarray], pad_px: int = 36, min_size_px: int = 240) -> np.ndarray:
    pts_all = []
    for p in polys:
        if p is None:
            continue
        arr = np.asarray(p, dtype=np.float32).reshape(-1, 2)
        if arr.size > 0 and np.all(np.isfinite(arr)):
            pts_all.append(arr)
    if not pts_all:
        return img.copy()
    pts = np.concatenate(pts_all, axis=0)
    xmin = int(max(0, math.floor(float(np.min(pts[:, 0]))) - pad_px))
    ymin = int(max(0, math.floor(float(np.min(pts[:, 1]))) - pad_px))
    xmax = int(min(img.shape[1], math.ceil(float(np.max(pts[:, 0]))) + pad_px))
    ymax = int(min(img.shape[0], math.ceil(float(np.max(pts[:, 1]))) + pad_px))
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half_w = max(0.5 * (xmax - xmin), 0.5 * float(min_size_px))
    half_h = max(0.5 * (ymax - ymin), 0.5 * float(min_size_px))
    xmin = max(0, int(round(cx - half_w)))
    xmax = min(int(img.shape[1]), int(round(cx + half_w)))
    ymin = max(0, int(round(cy - half_h)))
    ymax = min(int(img.shape[0]), int(round(cy + half_h)))
    if xmax <= xmin or ymax <= ymin:
        return img.copy()
    return img[ymin:ymax, xmin:xmax].copy()


def draw_camera_iou_pair(base_img: np.ndarray,
                         cam_meta: dict,
                         pred_world: np.ndarray,
                         gt_world: np.ndarray,
                         title: str = "IoU") -> Optional[tuple[np.ndarray, float]]:
    pred_uv = _project_world_pts_to_carla_image(pred_world, cam_meta)
    gt_uv = _project_world_pts_to_carla_image(gt_world, cam_meta)
    pred_hull = _convex_hull_from_uv(pred_uv)
    gt_hull = _convex_hull_from_uv(gt_uv)
    if pred_hull is None or gt_hull is None:
        return None

    out = base_img.copy()
    iou, inter_area, union_area, inter_poly = _camera_hull_iou(pred_hull, gt_hull)

    overlay = out.copy()
    cv2.fillPoly(overlay, [gt_hull.astype(np.int32)], color=(0, 0, 255), lineType=cv2.LINE_AA)
    cv2.fillPoly(overlay, [pred_hull.astype(np.int32)], color=(0, 255, 0), lineType=cv2.LINE_AA)
    if inter_poly is not None and len(inter_poly) >= 3:
        inter_poly = np.asarray(inter_poly, dtype=np.float32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [inter_poly.astype(np.int32)], color=(0, 165, 255), lineType=cv2.LINE_AA)
    out = cv2.addWeighted(overlay, 0.22, out, 0.78, 0.0)

    cv2.polylines(out, [gt_hull.astype(np.int32)], isClosed=True, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.polylines(out, [pred_hull.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    # === 改這裡：用真正 3D 幾何中心，而不是 hull 頂點平均 ===
    gt_center_world = np.mean(np.asarray(gt_world, dtype=np.float64), axis=0, keepdims=True)   # (1,3)
    pred_center_world = np.mean(np.asarray(pred_world, dtype=np.float64), axis=0, keepdims=True) # (1,3)

    gt_center_uv = _project_world_pts_to_carla_image(gt_center_world, cam_meta)
    pred_center_uv = _project_world_pts_to_carla_image(pred_center_world, cam_meta)

    if gt_center_uv is not None and np.all(np.isfinite(gt_center_uv)):
        gt_ctr = tuple(np.round(gt_center_uv[0]).astype(np.int32).tolist())
        cv2.drawMarker(out, gt_ctr, (0, 0, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)

    if pred_center_uv is not None and np.all(np.isfinite(pred_center_uv)):
        pd_ctr = tuple(np.round(pred_center_uv[0]).astype(np.int32).tolist())
        cv2.drawMarker(out, pd_ctr, (0, 255, 0),
                       markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)

    out = _crop_from_polys(out, [gt_hull, pred_hull], pad_px=44, min_size_px=280)
    draw_text_lines(out, 8, 58,
                    [f"{title}={iou:.4f}", f"Inter={inter_area:.1f} Union={union_area:.1f}", "GT=red  Pred=green"],
                    color=(255,255,255), bg=(24,24,24), font_scale=0.5, thickness=1)
    return out, iou


def draw_top_bev_from_carla_image(base_img: np.ndarray, top_cam_meta: dict, pred_world: np.ndarray, gt_world: np.ndarray,
                                  iou: float, inter_area: float, union_area: float) -> Optional[np.ndarray]:
    img = base_img.copy()
    pred_uv = _project_world_pts_to_top_image(pred_world[4:8], top_cam_meta)
    gt_uv = _project_world_pts_to_top_image(gt_world[4:8], top_cam_meta)
    if pred_uv is None or gt_uv is None:
        return None
    if (not np.all(np.isfinite(pred_uv))) or (not np.all(np.isfinite(gt_uv))):
        return None
    pred_pts = np.round(pred_uv).astype(np.int32).reshape(-1, 1, 2)
    gt_pts = np.round(gt_uv).astype(np.int32).reshape(-1, 1, 2)

    overlay = img.copy()
    cv2.fillPoly(overlay, [gt_pts], color=(0, 0, 255), lineType=cv2.LINE_AA)
    cv2.fillPoly(overlay, [pred_pts], color=(0, 255, 0), lineType=cv2.LINE_AA)
    img = cv2.addWeighted(overlay, 0.22, img, 0.78, 0.0)
    cv2.polylines(img, [gt_pts], isClosed=True, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.polylines(img, [pred_pts], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    gt_ctr = np.mean(gt_pts.reshape(-1, 2), axis=0).astype(np.int32)
    pd_ctr = np.mean(pred_pts.reshape(-1, 2), axis=0).astype(np.int32)
    cv2.drawMarker(img, tuple(gt_ctr), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
    cv2.drawMarker(img, tuple(pd_ctr), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)

    lines = [
        f"BEV IoU={iou:.4f}",
        f"Inter={inter_area:.3f}  Union={union_area:.3f}",
        "GT top face=red  Pred top face=green",
    ]
    draw_text_lines(img, 8, 58, lines, color=(255,255,255), bg=(24,24,24), font_scale=0.5, thickness=1)
    return img
