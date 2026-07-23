"""Ground-plane and projection-consistency translation solvers."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from scipy.special import logsumexp

from .geometry import project_points
from .visualization import (
    _bev_box_corners_xz, _ground_plane_sensor_from_w2s, _ground_y_from_kitti_xz,
)

def _project_pts(P2: np.ndarray, pts3: np.ndarray) -> np.ndarray:
    """
    pts3: (N,3) in camera coords
    return uv: (N,2)
    """
    N = pts3.shape[0]
    pts4 = np.concatenate([pts3, np.ones((N, 1), dtype=np.float64)], axis=1)  # (N,4)
    p = (P2 @ pts4.T).T  # (N,3)
    u = p[:, 0] / np.clip(p[:, 2], 1e-6, None)
    v = p[:, 1] / np.clip(p[:, 2], 1e-6, None)
    return np.stack([u, v], axis=1)


def _corners_3d_bottom_center(h: float, w: float, l: float) -> np.ndarray:
    """
    KITTI object coords (bottom center at origin):
      x: right, y: down, z: forward
      bottom face: y=0
      top face: y=-h
    Return corners (8,3).
    """
    x = w / 2.0
    z = l / 2.0
    # 4 bottom then 4 top
    corners = np.array([
        [ x, 0.0,  z],
        [ x, 0.0, -z],
        [-x, 0.0, -z],
        [-x, 0.0,  z],
        [ x, -h,  z],
        [ x, -h, -z],
        [-x, -h, -z],
        [-x, -h,  z],
    ], dtype=np.float64)
    return corners


def _Ry(ry: float) -> np.ndarray:
    c = math.cos(ry)
    s = math.sin(ry)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]], dtype=np.float64)


def _smooth_min(a: np.ndarray, k: float) -> float:
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(-logsumexp(-k * a) / k)


def _smooth_max(a: np.ndarray, k: float) -> float:
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(logsumexp(k * a) / k)


def _bbox_from_proj_uv_smooth(uv: np.ndarray, k: float = 50.0) -> Tuple[float, float, float, float]:
    u = uv[:, 0]
    v = uv[:, 1]
    xmin = _smooth_min(u, k)
    xmax = _smooth_max(u, k)
    ymin = _smooth_min(v, k)
    ymax = _smooth_max(v, k)
    return xmin, ymin, xmax, ymax


def solve_translation_tightfit(
    P2: np.ndarray,
    bbox_xyxy: Tuple[float, float, float, float],
    dims_hwl: Tuple[float, float, float],
    ry: float,
    t0_xyz: Tuple[float, float, float],
    k_smooth: float = 50.0,
    max_nfev: int = 15,
) -> Tuple[float, float, float]:
    """
    Deep3DBox-like: refine translation (x,y,z) so that projected 3D box bbox matches 2D bbox.
    Uses robust least squares. Falls back to t0 if solver fails.
    """
    try:
        from scipy.optimize import least_squares
    except Exception:
        return t0_xyz

    x1, y1, x2, y2 = bbox_xyxy
    h, w, l = dims_hwl

    corners = _corners_3d_bottom_center(h, w, l)  # (8,3)
    R = _Ry(float(ry))

    # normalize residual scale (avoid huge gradient when bbox large)
    bw = max(10.0, (x2 - x1))
    bh = max(10.0, (y2 - y1))
    inv_scale = np.array([1.0 / bw, 1.0 / bh, 1.0 / bw, 1.0 / bh], dtype=np.float64)

    def fun(t):
        tx, ty, tz = float(t[0]), float(t[1]), float(t[2])
        # enforce positive depth softly
        if tz <= 0.1:
            # large penalty
            return (np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float64))

        pts = (corners @ R.T) + np.array([tx, ty, tz], dtype=np.float64)  # (8,3)
        uv = _project_pts(P2, pts)
        px1, py1, px2, py2 = _bbox_from_proj_uv_smooth(uv, k=k_smooth)

        r = np.array([px1 - x1, py1 - y1, px2 - x2, py2 - y2], dtype=np.float64)
        return r * inv_scale

    t0 = np.array([t0_xyz[0], t0_xyz[1], t0_xyz[2]], dtype=np.float64)

    # bounds: keep it sane; 你可依場景調整
    lb = np.array([-80.0, -10.0,  0.1], dtype=np.float64)
    ub = np.array([ 80.0,  10.0, 200.0], dtype=np.float64)

    try:
        res = least_squares(
            fun,
            t0,
            bounds=(lb, ub),
            method="trf",
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=int(max_nfev),
        )
        if res.x is None:
            return t0_xyz
        return float(res.x[0]), float(res.x[1]), float(res.x[2])
    except Exception:
        return t0_xyz


def solve_translation_tightfit_x_ground_y_fixed_z(
    P2: np.ndarray,
    bbox_xyxy: Tuple[float, float, float, float],
    dims_hwl: Tuple[float, float, float],
    ry: float,
    x0: float,
    z_fixed: float,
    front_w2s: np.ndarray,
    k_smooth: float = 50.0,
    max_nfev: int = 15,
) -> Tuple[float, float, float]:
    """
    Tight-fit variant:
      - optimize only x
      - keep z fixed
      - recompute y from ground plane at every iteration

    Returns:
      (x, y, z_fixed)
    """
    try:
        from scipy.optimize import least_squares
    except Exception:
        y0 = _ground_y_from_kitti_xz(x0, z_fixed, front_w2s)
        if y0 is None:
            return float(x0), 0.0, float(z_fixed)
        return float(x0), float(y0), float(z_fixed)

    x1, y1, x2, y2 = bbox_xyxy
    h, w, l = dims_hwl

    corners = _corners_3d_bottom_center(h, w, l)  # (8,3)
    R = _Ry(float(ry))

    bw = max(10.0, (x2 - x1))
    bh = max(10.0, (y2 - y1))
    inv_scale = np.array([1.0 / bw, 1.0 / bh, 1.0 / bw, 1.0 / bh], dtype=np.float64)

    z_fixed = float(z_fixed)

    def fun(tx_arr):
        tx = float(tx_arr[0])

        if z_fixed <= 0.1:
            return np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float64)

        ty = _ground_y_from_kitti_xz(tx, z_fixed, front_w2s)
        if ty is None or (not math.isfinite(ty)):
            return np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float64)

        pts = (corners @ R.T) + np.array([tx, ty, z_fixed], dtype=np.float64)  # (8,3)
        uv = _project_pts(P2, pts)
        px1, py1, px2, py2 = _bbox_from_proj_uv_smooth(uv, k=k_smooth)

        r = np.array([px1 - x1, py1 - y1, px2 - x2, py2 - y2], dtype=np.float64)
        return r * inv_scale

    t0 = np.array([float(x0)], dtype=np.float64)

    lb = np.array([-80.0], dtype=np.float64)
    ub = np.array([+80.0], dtype=np.float64)

    try:
        res = least_squares(
            fun,
            t0,
            bounds=(lb, ub),
            method="trf",
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=int(max_nfev),
        )
        if res.x is None:
            tx = float(x0)
        else:
            tx = float(res.x[0])

        ty = _ground_y_from_kitti_xz(tx, z_fixed, front_w2s)
        if ty is None or (not math.isfinite(ty)):
            ty0 = _ground_y_from_kitti_xz(float(x0), z_fixed, front_w2s)
            if ty0 is None:
                return float(x0), 0.0, z_fixed
            return float(x0), float(ty0), z_fixed

        return float(tx), float(ty), z_fixed
    except Exception:
        ty0 = _ground_y_from_kitti_xz(float(x0), z_fixed, front_w2s)
        if ty0 is None:
            return float(x0), 0.0, z_fixed
        return float(x0), float(ty0), z_fixed


def solve_translation_geom_center_x_grounded(
    P2: np.ndarray,
    bbox_xyxy: Tuple[float, float, float, float],
    dims_hwl: Tuple[float, float, float],
    ry: float,
    x0: float,
    z_fixed: float,
    front_w2s: np.ndarray,
    u_geom_target: float,
    k_smooth: float = 50.0,
    max_nfev: int = 15,
    w_geom: float = 1.0,
    w_side: float = 0.30,
    w_bottom: float = 0.15,
    w_x_prior: float = 0.05,
) -> Tuple[float, float, float]:
    """
    Near-range x-only grounded refinement.

    Goal:
      - keep z fixed
      - keep y on ground plane
      - solve x so that the projected 3D geometric-center u approaches u_geom_target
      - still weakly respect bbox left/right/bottom supports

    This is intended for near/off-axis cases where:
      bbox-center x != 3D geometric-center projection x
    due to strong perspective asymmetry of the projected 3D box.
    """
    try:
        from scipy.optimize import least_squares
    except Exception:
        y0 = _ground_y_from_kitti_xz(x0, z_fixed, front_w2s)
        if y0 is None:
            return float(x0), 0.0, float(z_fixed)
        return float(x0), float(y0), float(z_fixed)

    x1, y1, x2, y2 = map(float, bbox_xyxy)
    h, w, l = map(float, dims_hwl)

    bw = max(10.0, x2 - x1)
    bh = max(10.0, y2 - y1)
    z_ref = max(abs(float(z_fixed)), 1.0)
    z_fixed = float(z_fixed)
    u_geom_target = float(u_geom_target)

    def fun(tx_arr: np.ndarray) -> np.ndarray:
        tx = float(tx_arr[0])

        if z_fixed <= 0.1:
            return np.array([10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float64)

        ty = _ground_y_from_kitti_xz(tx, z_fixed, front_w2s)
        if ty is None or (not math.isfinite(ty)):
            return np.array([10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float64)

        uv_geom = project_pred_geom_uv(P2, tx, ty, z_fixed, h)
        proj_bbox = project_pred_proj_bbox_info(
            P2=P2,
            x=tx, y=ty, z=z_fixed,
            h=h, w=w, l=l, ry=ry,
            k_smooth=float(k_smooth),
        )
        if (uv_geom is None) or (proj_bbox is None):
            return np.array([10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float64)

        return np.array([
            float(w_geom)    * (float(uv_geom[0]) - u_geom_target) / bw,
            float(w_side)    * (float(proj_bbox["x1"]) - x1) / bw,
            float(w_side)    * (float(proj_bbox["x2"]) - x2) / bw,
            float(w_bottom)  * (float(proj_bbox["y2"]) - y2) / bh,
            float(w_x_prior) * (tx - float(x0)) / z_ref,
        ], dtype=np.float64)

    t0 = np.array([float(x0)], dtype=np.float64)
    lb = np.array([-80.0], dtype=np.float64)
    ub = np.array([+80.0], dtype=np.float64)

    try:
        res = least_squares(
            fun,
            t0,
            bounds=(lb, ub),
            method="trf",
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=int(max_nfev),
        )
        tx = float(res.x[0]) if (res.x is not None) else float(x0)
    except Exception:
        tx = float(x0)

    ty = _ground_y_from_kitti_xz(tx, z_fixed, front_w2s)
    if ty is None or (not math.isfinite(ty)):
        ty0 = _ground_y_from_kitti_xz(float(x0), z_fixed, front_w2s)
        if ty0 is None:
            return float(x0), 0.0, float(z_fixed)
        return float(x0), float(ty0), float(z_fixed)

    return float(tx), float(ty), float(z_fixed)


def _ground_footprint_xyz_from_center(
    x: float,
    z: float,
    dims_hwl: Tuple[float, float, float],
    ry: float,
    front_w2s: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Build the 4 bottom footprint corners in KITTI camera coords, constrained to ground plane.
    Returns shape (4,3): [X, Y, Z] for each bottom corner.
    """
    h, w, l = dims_hwl
    xz_corners = _bev_box_corners_xz(float(x), float(z), float(ry), L=float(l), W=float(w))
    pts = []
    for X, Z in xz_corners:
        Y = _ground_y_from_kitti_xz(float(X), float(Z), front_w2s)
        if Y is None or (not math.isfinite(Y)):
            return None
        pts.append([float(X), float(Y), float(Z)])
    return np.asarray(pts, dtype=np.float64)


def _upright_box_xyz_from_ground_footprint(bottom_xyz: np.ndarray, h: float) -> np.ndarray:
    """
    Given 4 ground-constrained bottom corners (KITTI coords), make an upright 3D box:
      top corners = bottom corners shifted by -h along KITTI Y (down-positive).
    Output shape (8,3): bottom 4 then top 4.
    """
    bottom_xyz = np.asarray(bottom_xyz, dtype=np.float64).reshape(4, 3)
    top_xyz = bottom_xyz.copy()
    top_xyz[:, 1] -= float(h)
    return np.vstack([bottom_xyz, top_xyz])


def _ground_plane_kitti_from_w2s(front_w2s: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Convert CARLA sensor-frame ground plane to KITTI camera coords.

    Existing convention in this file:
      sensor point p_s = [z_kitti, x_kitti, -y_kitti]

    If plane in sensor frame is:
      n_s^T p_s + d = 0
    then plane in KITTI frame is:
      n_k^T p_k + d = 0
    with n_k = M^T n_s, where p_s = M p_k.
    """
    n_s, d_s = _ground_plane_sensor_from_w2s(front_w2s)
    n_s = np.asarray(n_s, dtype=np.float64).reshape(3)

    # p_s = M p_k,  p_k=[x,y,z], p_s=[z,x,-y]
    # => M = [[0,0,1],[1,0,0],[0,-1,0]]
    # => n_k = M^T n_s = [n_y, -n_z, n_x]
    n_k = np.array([float(n_s[1]), float(-n_s[2]), float(n_s[0])], dtype=np.float64)
    return n_k, float(d_s)


def _build_ground_metric_homography_from_w2s(
    P2: np.ndarray,
    front_w2s: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Build a metric 2D coordinate system on the ground plane, then derive the
    plane->image homography:

        x_img ~ H_pi2img [u, v, 1]^T

    where (u,v) are metric ground-plane coordinates in meters.

    Returns:
      H_pi2img : (3,3) homography from ground metric plane to image
      H_img2pi : inverse homography
      p0       : (3,) origin of the ground metric frame in KITTI camera coords
      e1,e2    : (3,) orthonormal basis vectors spanning the ground plane
      n_k,d_k  : KITTI-frame plane equation n_k^T X + d_k = 0
    """
    P2 = np.asarray(P2, dtype=np.float64).reshape(3, 4)
    n_k, d_k = _ground_plane_kitti_from_w2s(front_w2s)

    nn = float(np.linalg.norm(n_k))
    if nn < 1e-9:
        raise ValueError("ground plane normal norm too small")
    n_k = n_k / nn
    d_k = float(d_k) / nn

    # Closest point on plane to camera origin (metric plane origin)
    p0 = (-d_k) * n_k

    # Choose a stable in-plane forward-ish axis first, then fallback to x-axis.
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    e1 = z_axis - n_k * float(np.dot(n_k, z_axis))
    if float(np.linalg.norm(e1)) < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        e1 = x_axis - n_k * float(np.dot(n_k, x_axis))
    e1n = float(np.linalg.norm(e1))
    if e1n < 1e-9:
        raise ValueError("failed to build first in-plane axis")
    e1 = e1 / e1n

    e2 = np.cross(n_k, e1)
    e2n = float(np.linalg.norm(e2))
    if e2n < 1e-9:
        raise ValueError("failed to build second in-plane axis")
    e2 = e2 / e2n

    # X_cam_h = G [u,v,1]^T
    G = np.zeros((4, 3), dtype=np.float64)
    G[:3, 0] = e1
    G[:3, 1] = e2
    G[:3, 2] = p0
    G[3, 2] = 1.0

    H_pi2img = P2 @ G
    detH = float(np.linalg.det(H_pi2img))
    if abs(detH) < 1e-12:
        raise ValueError("ground homography is singular")
    H_img2pi = np.linalg.inv(H_pi2img)
    return H_pi2img, H_img2pi, p0, e1, e2, n_k, d_k


def _line_from_bbox_edge_x(x_edge: float) -> np.ndarray:
    return np.array([1.0, 0.0, -float(x_edge)], dtype=np.float64)


def _line_from_bbox_edge_y(y_edge: float) -> np.ndarray:
    return np.array([0.0, 1.0, -float(y_edge)], dtype=np.float64)


def _plane_line_from_image_line(H_pi2img: np.ndarray, l_img: np.ndarray) -> np.ndarray:
    """
    If x_img ~ H X_pi, then image line l_img maps to plane line:
      l_pi ~ H^T l_img
    """
    l_pi = np.asarray(H_pi2img, dtype=np.float64).T @ np.asarray(l_img, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(l_pi[:2]))
    if n > 1e-12:
        l_pi = l_pi / n
    return l_pi


def _heading_uv_on_ground_plane(ry: float, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    """
    Convert KITTI camera-frame heading to metric ground-plane coordinates.

    Existing BEV convention in this file:
      forward heading in XZ plane = [cos(ry), 0, -sin(ry)]
    """
    h_cam = np.array([math.cos(float(ry)), 0.0, -math.sin(float(ry))], dtype=np.float64)
    uv = np.array([float(np.dot(e1, h_cam)), float(np.dot(e2, h_cam))], dtype=np.float64)
    n = float(np.linalg.norm(uv))
    if n < 1e-9:
        return np.array([1.0, 0.0], dtype=np.float64)
    return uv / n


def _ground_plane_center_to_cam_xyz(c_uv: np.ndarray, p0: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    c_uv = np.asarray(c_uv, dtype=np.float64).reshape(2)
    return (
        np.asarray(p0, dtype=np.float64)
        + float(c_uv[0]) * np.asarray(e1, dtype=np.float64)
        + float(c_uv[1]) * np.asarray(e2, dtype=np.float64)
    )


def _ground_footprint_cam_xyz_from_plane_center(
    c_uv: np.ndarray,
    dims_hwl: Tuple[float, float, float],
    heading_uv: np.ndarray,
    p0: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
) -> np.ndarray:
    """
    Build the 4 ground-footprint corners from a metric-plane center (u,v).

    heading_uv : unit vector of vehicle longitudinal axis on metric plane.
    """
    h, w, l = map(float, dims_hwl)
    c_uv = np.asarray(c_uv, dtype=np.float64).reshape(2)
    d = np.asarray(heading_uv, dtype=np.float64).reshape(2)
    dn = float(np.linalg.norm(d))
    if dn < 1e-9:
        d = np.array([1.0, 0.0], dtype=np.float64)
    else:
        d = d / dn
    dp = np.array([-d[1], d[0]], dtype=np.float64)

    corners_uv = np.array([
        c_uv + 0.5 * l * d + 0.5 * w * dp,
        c_uv + 0.5 * l * d - 0.5 * w * dp,
        c_uv - 0.5 * l * d - 0.5 * w * dp,
        c_uv - 0.5 * l * d + 0.5 * w * dp,
    ], dtype=np.float64)

    pts = np.zeros((4, 3), dtype=np.float64)
    for i in range(4):
        pts[i] = _ground_plane_center_to_cam_xyz(corners_uv[i], p0, e1, e2)
    return pts


def solve_translation_metric_plane_support_xz(
    P2: np.ndarray,
    bbox_xyxy: Tuple[float, float, float, float],
    dims_hwl: Tuple[float, float, float],
    ry: float,
    x0: float,
    z0: float,
    front_w2s: np.ndarray,
    k_tangent: float = 60.0,
    k_top: float = 50.0,
    max_nfev: int = 25,
    w_bottom: float = 0.35,
    w_top: float = 0.20,
    w_z_prior: float = 0.15,
    w_center_x: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Metric-plane support-line solver.

    Idea:
      1) Build a metric 2D coordinate system on the ground plane.
      2) Map bbox support lines (left/right/bottom) back to that plane.
      3) Use vehicle heading + size prior to initialize the footprint center on the plane.
      4) Refine the center in plane coordinates by minimizing image support residuals.

    Returns refined KITTI bottom-center (x,y,z).
    """
    try:
        from scipy.optimize import least_squares
    except Exception:
        y0 = _ground_y_from_kitti_xz(float(x0), float(z0), front_w2s)
        if y0 is None:
            return float(x0), 0.0, float(z0)
        return float(x0), float(y0), float(z0)

    x1, y1, x2, y2 = map(float, bbox_xyxy)
    h, w, l = map(float, dims_hwl)

    try:
        H_pi2img, H_img2pi, p0, e1, e2, _n_k, _d_k = _build_ground_metric_homography_from_w2s(P2, front_w2s)
    except Exception:
        y0 = _ground_y_from_kitti_xz(float(x0), float(z0), front_w2s)
        if y0 is None:
            return float(x0), 0.0, float(z0)
        return float(x0), float(y0), float(z0)

    # metric-plane support lines
    lL_pi = _plane_line_from_image_line(H_pi2img, _line_from_bbox_edge_x(x1))
    lR_pi = _plane_line_from_image_line(H_pi2img, _line_from_bbox_edge_x(x2))
    lB_pi = _plane_line_from_image_line(H_pi2img, _line_from_bbox_edge_y(y2))

    d = _heading_uv_on_ground_plane(ry, e1, e2)
    dp = np.array([-d[1], d[0]], dtype=np.float64)

    # init center from current x0,z0 projected into metric plane
    y0 = _ground_y_from_kitti_xz(float(x0), float(z0), front_w2s)
    if y0 is None:
        y0 = 0.0
    c0_cam = np.array([float(x0), float(y0), float(z0)], dtype=np.float64)
    x0_img_h = P2 @ np.array([c0_cam[0], c0_cam[1], c0_cam[2], 1.0], dtype=np.float64)
    c0_uv_h = H_img2pi @ x0_img_h
    if abs(float(c0_uv_h[2])) > 1e-9:
        c0_uv = np.array([
            float(c0_uv_h[0] / c0_uv_h[2]),
            float(c0_uv_h[1] / c0_uv_h[2]),
        ], dtype=np.float64)
    else:
        c0_uv = np.zeros(2, dtype=np.float64)

    # closed-form candidate init from support equations
    # center lies midway between left/right supports along dp
    # and one half-length away from the bottom support along d (front/rear ambiguity).
    b_side = -0.5 * (float(lL_pi[2]) + float(lR_pi[2]))
    A = np.stack([dp, d], axis=0)

    cands = []
    for sgn in (+1.0, -1.0):
        b = np.array([
            b_side,
            -float(lB_pi[2]) + sgn * (0.5 * l),
        ], dtype=np.float64)
        try:
            c_init = np.linalg.solve(A, b)
        except Exception:
            c_init = c0_uv.copy()
        cands.append(c_init)

    bw = max(10.0, x2 - x1)
    bh = max(10.0, y2 - y1)
    z_ref = max(abs(float(z0)), 1.0)

    def residual_from_center(c_uv: np.ndarray) -> np.ndarray:
        bottom_xyz = _ground_footprint_cam_xyz_from_plane_center(
            c_uv=c_uv,
            dims_hwl=(h, w, l),
            heading_uv=d,
            p0=p0,
            e1=e1,
            e2=e2,
        )
        uv_bottom = project_points(P2, bottom_xyz)
        if (not np.all(np.isfinite(uv_bottom))) or np.any(bottom_xyz[:, 2] <= 0.05):
            return np.array([10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float64)

        u_left = _smooth_min(uv_bottom[:, 0], k_tangent)
        u_right = _smooth_max(uv_bottom[:, 0], k_tangent)
        v_bottom = _smooth_max(uv_bottom[:, 1], k_tangent)

        box_xyz = _upright_box_xyz_from_ground_footprint(bottom_xyz, h=h)
        uv_box = project_points(P2, box_xyz)
        if not np.all(np.isfinite(uv_box)):
            return np.array([10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float64)

        v_top = _smooth_min(uv_box[4:, 1], k_top)
        c_cam = _ground_plane_center_to_cam_xyz(c_uv, p0, e1, e2)

        center_xyz = np.array([[c_cam[0], c_cam[1] - 0.5 * h, c_cam[2]]], dtype=np.float64)
        uv_center = project_points(P2, center_xyz)
        u_center = float(uv_center[0, 0])
        u_bbox_center = 0.5 * (x1 + x2)

        return np.array([
            (u_left - x1) / bw,
            (u_right - x2) / bw,
            float(w_bottom) * (v_bottom - y2) / bh,
            float(w_top) * (v_top - y1) / bh,
            float(w_z_prior) * (float(c_cam[2]) - float(z0)) / z_ref,
            float(w_center_x) * (u_center - u_bbox_center) / bw,
        ], dtype=np.float64)

    best_cost = float("inf")
    best_xyz = None

    for c_init in cands:
        try:
            res = least_squares(
                lambda t: residual_from_center(np.asarray(t, dtype=np.float64)),
                x0=np.asarray(c_init, dtype=np.float64),
                method="trf",
                loss="soft_l1",
                f_scale=1.0,
                max_nfev=int(max_nfev),
            )
            c_opt = np.asarray(res.x if res.x is not None else c_init, dtype=np.float64)
        except Exception:
            c_opt = np.asarray(c_init, dtype=np.float64)

        r = residual_from_center(c_opt)
        cost = float(np.dot(r, r))
        c_cam = _ground_plane_center_to_cam_xyz(c_opt, p0, e1, e2)

        if np.all(np.isfinite(c_cam)) and float(c_cam[2]) > 0.05 and cost < best_cost:
            best_cost = cost
            best_xyz = c_cam

    if best_xyz is None:
        y_fb = _ground_y_from_kitti_xz(float(x0), float(z0), front_w2s)
        if y_fb is None:
            return float(x0), 0.0, float(z0)
        return float(x0), float(y_fb), float(z0)

    return float(best_xyz[0]), float(best_xyz[1]), float(best_xyz[2])


def solve_translation_tightfit_xz_ground_tangent(
    P2: np.ndarray,
    bbox_xyxy: Tuple[float, float, float, float],
    dims_hwl: Tuple[float, float, float],
    ry: float,
    x0: float,
    z0: float,
    front_w2s: np.ndarray,
    k_tangent: float = 60.0,
    k_top: float = 50.0,
    max_nfev: int = 25,
    w_bottom: float = 0.35,
    w_top: float = 0.20,
    w_z_prior: float = 0.15,
) -> Tuple[float, float, float]:
    """
    Ground-plane constrained footprint-tangent solver.

    Optimize:
      - x, z
    Recompute:
      - y from ground plane every iteration

    Main idea:
      - x1/x2 are matched by the projected left/right support of the GROUND FOOTPRINT
      - y2 is matched by footprint bottom support
      - y1 is weakly stabilized by the projected top support of the upright box
      - z0 is used only as a weak prior, not fixed
    """
    try:
        from scipy.optimize import least_squares
    except Exception:
        y0 = _ground_y_from_kitti_xz(float(x0), float(z0), front_w2s)
        if y0 is None:
            return float(x0), 0.0, float(z0)
        return float(x0), float(y0), float(z0)

    x1, y1, x2, y2 = map(float, bbox_xyxy)
    h, w, l = map(float, dims_hwl)

    bw = max(10.0, x2 - x1)
    bh = max(10.0, y2 - y1)
    z_ref = max(abs(float(z0)), 1.0)

    def fun(t: np.ndarray) -> np.ndarray:
        tx = float(t[0])
        tz = float(t[1])

        if tz <= 0.1:
            return np.array([10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float64)

        # 1) ground footprint
        bottom_xyz = _ground_footprint_xyz_from_center(
            x=tx, z=tz, dims_hwl=(h, w, l), ry=ry, front_w2s=front_w2s
        )
        if bottom_xyz is None:
            return np.array([10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float64)

        uv_bottom = project_points(P2, bottom_xyz)
        if not np.all(np.isfinite(uv_bottom)):
            return np.array([10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float64)

        # tangent / support from ground footprint
        u_left = _smooth_min(uv_bottom[:, 0], k_tangent)
        u_right = _smooth_max(uv_bottom[:, 0], k_tangent)
        v_bottom = _smooth_max(uv_bottom[:, 1], k_tangent)

        # 2) full upright box, only for weak top stabilization
        box_xyz = _upright_box_xyz_from_ground_footprint(bottom_xyz, h=h)
        uv_box = project_points(P2, box_xyz)
        if not np.all(np.isfinite(uv_box)):
            return np.array([10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float64)

        v_top = _smooth_min(uv_box[4:, 1], k_top)  # top 4 corners only

        r = np.array([
            (u_left - x1) / bw,
            (u_right - x2) / bw,
            float(w_bottom) * (v_bottom - y2) / bh,
            float(w_top) * (v_top - y1) / bh,
            float(w_z_prior) * (tz - float(z0)) / z_ref,
        ], dtype=np.float64)
        return r

    t0 = np.array([float(x0), float(z0)], dtype=np.float64)
    lb = np.array([-80.0, 0.1], dtype=np.float64)
    ub = np.array([+80.0, 200.0], dtype=np.float64)

    try:
        res = least_squares(
            fun,
            t0,
            bounds=(lb, ub),
            method="trf",
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=int(max_nfev),
        )
        if res.x is None:
            tx, tz = float(x0), float(z0)
        else:
            tx, tz = float(res.x[0]), float(res.x[1])
    except Exception:
        tx, tz = float(x0), float(z0)

    ty = _ground_y_from_kitti_xz(tx, tz, front_w2s)
    if ty is None or (not math.isfinite(ty)):
        ty0 = _ground_y_from_kitti_xz(float(x0), float(z0), front_w2s)
        if ty0 is None:
            return float(x0), 0.0, float(z0)
        return float(x0), float(ty0), float(z0)

    return float(tx), float(ty), float(tz)
