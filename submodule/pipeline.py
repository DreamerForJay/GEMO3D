"""Per-image GEMO3D inference pipeline.

This module retains the original numerical behavior while moving reusable helpers
out of the command-line entry point.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from egonet_ori import infer_batch as egonet_infer_batch
try:
    from deep3d_head import infer_batch as deep3d_infer_batch
except Exception:
    deep3d_infer_batch = None
from vehicletype_head import crop_bgr

from .config import (
    KittiObj, apply_ry_offset, backproject_uv_depth, iou_xyxy,
    read_kitti_label_2, resolve_comp_model_for_k, resolve_dims,
    resolve_fixed_dims_hwl_for_k, resolve_ry, theta_ray_from_u, wrap_deg, wrap_pi,
)
from .compensation import _predict_compensation
from .geometry import (
    choose_proj_uv, find_best_gt_by_iou, project_gt_center_uv, project_points,
    project_pred_bottom_uv, project_pred_feedback_uv, project_pred_geom_uv,
    project_pred_proj_bbox_info,
)
from .optimization import (
    solve_translation_geom_center_x_grounded,
    solve_translation_metric_plane_support_xz, solve_translation_tightfit,
)
from .visualization import (
    _ground_y_from_kitti_xz, bev_iou_world_corners, build_pair_overlay_lines,
    choose_pred_center_xyz, draw_bev_follow_pair, draw_bev_local, draw_box3d,
    draw_camera_iou_pair, draw_text_lines, draw_text_panel_top_right,
    draw_top_bev_from_carla_image, iou3d_kitti_boxes,
    world_corners_from_kitti_box,
)

def infer_one_image(
    img_path: Path,
    calib_path: Path,
    yolo_model,
    car_names: List[str],
    conf_thres: float,
    imgsz: int,
    yolo_device: str,
    use_half: bool,
    egonet_enable: bool,
    deep3d_enable: bool,
    P2: np.ndarray,
    K: Dict[str, float],
    height_m: float,
    dims_source: str,
    fixed_dims_hwl: Tuple[float, float, float],  # (h,w,l)
    comp_model: Optional[dict],
    out_pred_data_dir: Path,
    use_bottom_center: bool = True,
    min_z: float = 0.1,
    label_path: Optional[Path] = None,
    meta_path: Optional[Path] = None,
    top_bev_img_path: Optional[Path] = None,
    fr_iou_img_path: Optional[Path] = None,
    side_iou_img_path: Optional[Path] = None,
    ry_offset_deg: float = 0.0,
    vis_dir: Optional[Path] = None,
    vis_iou_thres: float = 0.5,
    vis_draw_gt: bool = True,
    depth_use_gt_height: bool = False,

    pred_use_gt_hwl: bool = False,
    pred_use_gt_hwl_iou: float = -1.0,
    pred_center_source: str = "bbox",
    pred_use_gt_center_iou: float = -1.0,
    tightfit_enable: bool = False,
    tightfit_max_nfev: int = 15,
    tightfit_k: float = 50.0,
    vehclf=None,
    comp_models_by_type = None,
    default_comp_model = None,
    vehclf_crop_pad: float = 0.15,
    vehclf_min_prob: float = 0.40,
    pred_meta_rows: Optional[List[Dict[str, object]]] = None,
    force_veh_type: str = "",
    proj_center_source: str = "bbox_bottom",
    proj_use_gt_center_iou: float = -1.0,
    proj_refine_iters: int = 3,
    fixed_ground_y: float = -1.0,
):
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {img_path}")

    carla_meta = None
    front_w2s_meta = None
    top_cam_meta = None
    fr_iou_cam_meta = None
    side_iou_cam_meta = None
    top_bev_img = None
    fr_iou_img = None
    side_iou_img = None
    if meta_path is not None and Path(meta_path).is_file():
        with open(meta_path, "r", encoding="utf-8") as f:
            carla_meta = json.load(f)
        try:
            front_w2s_meta = np.asarray(carla_meta.get("extrinsics", None), dtype=np.float64).reshape(4, 4)
        except Exception:
            front_w2s_meta = None
        if isinstance(carla_meta.get("top_camera"), dict):
            top_cam_meta = carla_meta["top_camera"]
        if isinstance(carla_meta.get("fr_iou_camera"), dict):
            fr_iou_cam_meta = carla_meta["fr_iou_camera"]
        if isinstance(carla_meta.get("side_iou_camera"), dict):
            side_iou_cam_meta = carla_meta["side_iou_camera"]
    if top_bev_img_path is not None and Path(top_bev_img_path).is_file():
        top_bev_img = cv2.imread(str(top_bev_img_path))
    if fr_iou_img_path is not None and Path(fr_iou_img_path).is_file():
        fr_iou_img = cv2.imread(str(fr_iou_img_path))
    if side_iou_img_path is not None and Path(side_iou_img_path).is_file():
        side_iou_img = cv2.imread(str(side_iou_img_path))

    # YOLO inference
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = yolo_model.predict(
        source=img_rgb,
        imgsz=imgsz,
        conf=conf_thres,
        verbose=False,
        device=yolo_device,
        half=bool(use_half),
    )[0]

    names_map = yolo_model.names
    dets = []  # for EgoNet batch: (x1,y1,x2,y2,cls,conf)
    raw = []   # keep (x1,y1,x2,y2,conf)
    for b in res.boxes:
        cls_id = int(b.cls.item())
        cls_name = str(names_map[cls_id]).lower()
        if cls_name not in car_names:
            continue
        conf = float(b.conf.item())
        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
        if (y2 - y1) < 5:
            continue
        raw.append((x1, y1, x2, y2, conf))
        dets.append((x1, y1, x2, y2, "Car", conf))

    # Heads predict ry / dims (optional)
    pred_by_k = {}
    if force_veh_type and raw:
        for k in range(len(raw)):
            pred_by_k.setdefault(k, {})
            pred_by_k[k]["veh_label"] = force_veh_type
            pred_by_k[k]["veh_prob"] = 1.0
            pred_by_k[k]["veh_raw"] = force_veh_type
            pred_by_k[k]["veh_raw_p"] = 1.0
    elif vehclf is not None and raw:
        crops = []
        for (x1, y1, x2, y2, _conf) in raw:
            crops.append(crop_bgr(img, x1, y1, x2, y2, pad_ratio=vehclf_crop_pad))
        vt_outs = vehclf.predict_crops(crops)
        for k, o in enumerate(vt_outs):
            pred_by_k.setdefault(k, {})
            prob = float(o.prob)
            lab  = str(o.label)
            pred_by_k[k]["veh_label"] = (lab if prob >= float(vehclf_min_prob) else "unknown")
            pred_by_k[k]["veh_prob"]  = prob
            pred_by_k[k]["veh_raw"]   = str(o.raw_label)
            pred_by_k[k]["veh_raw_p"] = float(o.raw_prob)

    # EgoNet (orientation head)
    if egonet_enable and dets:
        preds = egonet_infer_batch(img, dets, P2)  # returns list aligned to dets
        for k, p in enumerate(preds):
            if k in pred_by_k and isinstance(pred_by_k[k], dict) and isinstance(p, dict):
                merged = dict(pred_by_k[k])
                merged.update(p)
                pred_by_k[k] = merged
            else:
                pred_by_k[k] = p

    # Deep3DBox head (dims + yaw). We'll convert to KITTI-friendly keys.
    if deep3d_enable and dets:
        if deep3d_infer_batch is None:
            raise RuntimeError("deep3d_head not available (import failed).")
        preds_d3 = deep3d_infer_batch(img, dets, P2)  # aligned to dets
        for k, pd in enumerate(preds_d3):
            # deep3d_head returns dims as (l,w,h). Convert to (h,w,l) to match this pipeline's convention.
            if isinstance(pd, dict) and ("dims" in pd):
                l, w, h = pd["dims"]
                pd["dims"] = (float(h), float(w), float(l))
            # Deep3DBox: do NOT force yaw->ry here.
            # We will recompute KITTI-consistent ry later from (alpha + theta_ray_from_u).
            if isinstance(pd, dict):
                pd["_src"] = "deep3d"
                if "alpha" in pd:
                    pd["alpha"] = wrap_pi(float(pd["alpha"]))

            if k in pred_by_k and isinstance(pred_by_k[k], dict) and isinstance(pd, dict):
                # merge: when dims_source==deep3d, you typically want deep3d's dims/ry to override
                merged = dict(pred_by_k[k])
                merged.update(pd)
                pred_by_k[k] = merged
            else:
                pred_by_k[k] = pd

    
    gt_objs = read_kitti_label_2(label_path, keep_cls="Car") if label_path is not None else []

    # Write KITTI prediction file
    out_path = out_pred_data_dir / f"{img_path.stem}.txt"
    lines = []
    pred_vis_cache = {}
    for k, (x1, y1, x2, y2, score) in enumerate(raw):
        # per-detection vehicle type profile
        fixed_hwl_k = resolve_fixed_dims_hwl_for_k(pred_by_k, k, default_hwl=fixed_dims_hwl)
        comp_model_k = resolve_comp_model_for_k(pred_by_k, k, comp_models_by_type, default_comp_model)
        h_px = max(1.0, (y2 - y1))
        # ---- KITTI 2D bbox (clamp first; used by tight-fit & output) ----
        x1o = max(0.0, x1); y1o = max(0.0, y1)
        x2o = max(x1o + 1.0, x2); y2o = max(y1o + 1.0, y2)

        # ---- optional GT matching (MOVED UP; needed before choosing projection uv) ----
        best_iou = -1.0
        best_gt = None
        need_gt_match = (
            depth_use_gt_height
            or pred_use_gt_hwl
            or (str(pred_center_source).lower() == "gt")
            or str(proj_center_source).lower().startswith("gt_")
        )
        if (gt_objs is not None) and (len(gt_objs) > 0) and need_gt_match:
            pred_box = (float(x1o), float(y1o), float(x2o), float(y2o))
            for g in gt_objs:
                giou = iou_xyxy(pred_box, (g.x1, g.y1, g.x2, g.y2))
                if giou > best_iou:
                    best_iou = giou
                    best_gt = g

        proj_center_iou_thres = (
            float(proj_use_gt_center_iou)
            if float(proj_use_gt_center_iou) >= 0.0
            else float(vis_iou_thres)
        )

        # ---- choose (u,v) for theta_ray / backproject ----
        # ---- choose initial (u,v) for theta_ray / backproject ----
        proj_mode0 = str(proj_center_source).strip().lower()
        if proj_mode0 in ("pred_reproj_bottom", "pred_reproj_geom_x"):
            proj_mode0 = "bbox_bottom"

        u, v, proj_src = choose_proj_uv(
            mode=proj_mode0,
            x1=x1, y1=y1, x2=x2, y2=y2,
            P2=P2,
            best_gt=best_gt,
            best_iou=best_iou,
            iou_thres=proj_center_iou_thres,
        )
        theta_ray = theta_ray_from_u(u, K)

        if (k in pred_by_k):
            p = pred_by_k[k]
            ry = resolve_ry(p, theta_ray, allow_alpha_fallback=True)
            ry = apply_ry_offset(ry, ry_offset_deg)
            h3d, w3d, l3d = resolve_dims(p, dims_source, fixed_hwl_k)
        else:
            p = None
            ry = 0.0
            h3d, w3d, l3d = fixed_hwl_k

        # ---- (CHEAT) overwrite predicted h,w,l with matched GT h,w,l ----
        gt_hwl_iou_thres = float(pred_use_gt_hwl_iou) if float(pred_use_gt_hwl_iou) >= 0.0 else float(vis_iou_thres)
        if pred_use_gt_hwl and (best_gt is not None) and (best_iou >= gt_hwl_iou_thres):
            h3d, w3d, l3d = float(best_gt.h), float(best_gt.w), float(best_gt.l)

        # ---- depth (nearest_time) ----
        H_used = float(h3d) if (dims_source == "egonet") else float(height_m)

        # Optionally override H_used by matched GT height (label_2 h)
        if depth_use_gt_height and (best_gt is not None) and (best_iou >= float(vis_iou_thres)):
            H_used = float(best_gt.h)

        z_est = float(K["fy"] * H_used / h_px)

        # ---- compensation ----
        comp = 0.0
        yaw_comp_deg = None
        if comp_model_k is not None:
            # New convention:
            # 0 deg: car front points to camera (-Z)
            # +deg : CCW in BEV (x->right, z->up)
            #yaw_comp_deg = wrap_deg(-math.degrees(ry) + 90.0)
            theta_ray = theta_ray_from_u(u, K)
            yaw_comp_deg = wrap_deg(math.degrees(wrap_pi(ry - theta_ray - math.pi))+90)
            comp = float(_predict_compensation(yaw_comp_deg, comp_model_k, z_raw=z_est))

        z = max(min_z, z_est - comp)
        z = max(min_z, z_est - comp)

        # ---- init from old depth path ----
        x, y_bbox, z = backproject_uv_depth(u, v, z, P2)
        y = y_bbox

        used_ground_y = False

        if float(fixed_ground_y) > 0.0:
            y = float(fixed_ground_y)
            used_ground_y = True

        elif front_w2s_meta is not None:
            y_gp = _ground_y_from_kitti_xz(x, z, front_w2s_meta)
            if y_gp is not None:
                y = y_gp
                used_ground_y = True

        # ---- tight-fit refine translation ----
        if tightfit_enable:
            if used_ground_y and (front_w2s_meta is not None):
                # footprint tangent geometry:
                # optimize x,z jointly; y always from ground plane
                x, y, z = solve_translation_metric_plane_support_xz(
                    P2=P2,
                    bbox_xyxy=(x1o, y1o, x2o, y2o),
                    dims_hwl=(h3d, w3d, l3d),
                    ry=ry,
                    x0=x,
                    z0=z,
                    front_w2s=front_w2s_meta,
                    k_tangent=max(60.0, float(tightfit_k)),
                    k_top=float(tightfit_k),
                    max_nfev=max(int(tightfit_max_nfev), 25),
                    w_center_x=0.0,
                )
            else:
                # fallback: original joint xyz tight-fit
                x, y, z = solve_translation_tightfit(
                    P2=P2,
                    bbox_xyxy=(x1o, y1o, x2o, y2o),
                    dims_hwl=(h3d, w3d, l3d),
                    ry=ry,
                    t0_xyz=(x, y, z),
                    k_smooth=float(tightfit_k),
                    max_nfev=int(tightfit_max_nfev),
                )
        # ---- debug snapshot before feedback ----
        u_bbox_ctr = 0.5 * (float(x1o) + float(x2o))
        v_bbox_ctr = 0.5 * (float(y1o) + float(y2o))
        proj_src_init = str(proj_src)
        u_init = float(u)
        v_init = float(v)
        theta_ray_init = float(theta_ray)
        ry_init = float(ry)
        yaw_comp_deg_init = float(yaw_comp_deg) if yaw_comp_deg is not None else float("nan")
        comp_init = float(comp)
        x_pre_fb = float(x)
        y_pre_fb = float(y)
        z_pre_fb = float(z)
        feedback_used = False
        feedback_iters_used = 0

        # ---- optional feedback loop: refine x-ray by predicted reprojection ----
        proj_mode_l = str(proj_center_source).strip().lower()
        if proj_mode_l in ("pred_reproj_bottom", "pred_reproj_geom_x"):
            feedback_used = True
            for _fb_iter in range(max(1, int(proj_refine_iters))):
                feedback_iters_used = max(feedback_iters_used, int(_fb_iter) + 1)
                uv_ref = project_pred_feedback_uv(
                    mode=proj_mode_l,
                    P2=P2,
                    x=x,
                    y=y,
                    z=z,
                    h=h3d,
                    w=w3d,
                    l=l3d,
                    ry=ry,
                    k_smooth=float(tightfit_k),
                )
                if uv_ref is None:
                    break

                u2, v2, proj_src2 = uv_ref
                use_xonly_grounded_refine = False
                u_geom_target = None

                if proj_mode_l == "pred_reproj_geom_x":
                    # ---------------------------------------------------------
                    # Two-mode update:
                    #   (A) near/off-axis : use perspective-compensated target u
                    #       and solve x only on the ground plane
                    #   (B) otherwise     : keep the far-range heading-based step
                    #
                    # Key idea for near range:
                    #   u_geom_target = u_bbox_ctr - (u_proj_boxctr - u_proj_geom)
                    #
                    # i.e. subtract the CURRENT perspective offset between
                    # projected-box center and projected geometric-center, instead
                    # of assuming the residual is mainly along projected heading.
                    # ---------------------------------------------------------

                    # current projected 3D-box 2D center (from uv_ref)
                    u_proj_boxctr = float(u2)
                    v_proj_boxctr = 0.5 * (float(y1o) + float(y2o))

                    # current predicted 3D geometric center projection
                    uv_geom = project_pred_geom_uv(P2, x, y, z, h3d)
                    if uv_geom is None:
                        break
                    u_proj_geom = float(uv_geom[0])
                    v_proj_geom = float(uv_geom[1])

                    # observed YOLO bbox center
                    u_bbox_ctr = 0.5 * (float(x1o) + float(x2o))
                    v_bbox_ctr = 0.5 * (float(y1o) + float(y2o))

                    # current perspective offset between projected 3D-box center
                    # and projected 3D geometric center
                    du_geom = float(u_proj_boxctr - u_proj_geom)

                    # fallback target: bbox center corrected by current perspective offset
                    u_fallback_geom = float(u_bbox_ctr - du_geom)

                    bw_obs = max(10.0, float(x2o - x1o))
                    offaxis_px = abs(float(u_bbox_ctr - K["cx"]))
                    ray_bbox_deg = abs(math.degrees(theta_ray_from_u(float(u_bbox_ctr), K)))

                    near_offaxis = (
                        (float(z) < 30.0) and (
                            (offaxis_px > 0.12 * float(img.shape[1]))
                            or (ray_bbox_deg > 6.0)
                            or (bw_obs > 80.0)
                        )
                    )
                    #near_offaxis = (float(z) < 40.0)
                    if near_offaxis:
                        # near-range main target:
                        # bbox center minus current projected perspective shift
                        lambda_persp = 1
                        u_geom_target = float(u_bbox_ctr - lambda_persp * du_geom)

                        # keep target within a conservative search band
                        u_pad = max(14.0, 0.35 * bw_obs)
                        u_geom_target = max(
                            float(x1o) - u_pad,
                            min(float(x2o) + u_pad, float(u_geom_target))
                        )

                        u2 = float(u_geom_target)
                        use_xonly_grounded_refine = True

                    else:
                        # far / weak-perspective case:
                        # keep heading-based step from the previous patch
                        rho_head = max(1.0, 0.5 * float(l3d))
                        c_geom = np.array(
                            [[float(x), float(y) - 0.5 * float(h3d), float(z)]],
                            dtype=np.float64
                        )
                        c_head = c_geom + np.array(
                            [[rho_head * math.cos(float(ry)), 0.0, -rho_head * math.sin(float(ry))]],
                            dtype=np.float64
                        )

                        uv_head = project_points(P2, c_head)[0]
                        if not np.all(np.isfinite(uv_head)):
                            break

                        t = np.array(
                            [float(uv_head[0] - u_proj_geom), float(uv_head[1] - v_proj_geom)],
                            dtype=np.float64
                        )
                        t_norm = float(np.linalg.norm(t))

                        eps_dir = 1e-6
                        if t_norm <= eps_dir:
                            u2 = float(u_fallback_geom)
                        else:
                            t_hat = t / t_norm

                            heading_v_conf = min(
                                1.0,
                                abs(t_hat[1]) / max(abs(t_hat[0]) + abs(t_hat[1]), 1e-6)
                            )
                            w_geom = 0.15 + 0.55 * heading_v_conf

                            u_ref = float(w_geom * u_proj_geom + (1.0 - w_geom) * u_proj_boxctr)
                            v_ref = float(w_geom * v_proj_geom + (1.0 - w_geom) * v_proj_boxctr)

                            du_obs = float(u_bbox_ctr - u_ref)
                            dv_obs = float(v_bbox_ctr - v_ref)
                            d = np.array([du_obs, dv_obs], dtype=np.float64)

                            if float(np.dot(d, t_hat)) < 0.0:
                                t_hat = -t_hat

                            tu = float(t_hat[0])
                            tv = float(t_hat[1])

                            w_u = 1.0
                            w_v = 0.25 + 1.25 * heading_v_conf

                            denom = float(w_u * tu * tu + w_v * tv * tv)
                            if denom <= 1e-8:
                                if abs(tu) > 1e-6:
                                    s = float(du_obs / tu)
                                else:
                                    s = 0.0
                            else:
                                s = float((w_u * tu * du_obs + w_v * tv * dv_obs) / denom)

                            max_s_px = max(18.0, 0.45 * bw_obs)
                            s = max(-max_s_px, min(max_s_px, float(s)))

                            u2 = float(u_ref + s * t_hat[0])

                            if not math.isfinite(u2):
                                u2 = float(u_fallback_geom)

                    # keep bottom-consistent v for backprojection
                    v2 = float(y2o)

                    # clamp search range
                    u_lo = float(x1o) - 0.75 * bw_obs
                    u_hi = float(x2o) + 0.75 * bw_obs
                    u2 = max(u_lo, min(u_hi, float(u2)))

                if (_fb_iter > 0) and (abs(float(u2) - float(u)) < 0.25):
                    break

                theta_ray2 = theta_ray_from_u(u2, K)

                if p is not None:
                    ry2 = resolve_ry(p, theta_ray2, allow_alpha_fallback=True)
                    ry2 = apply_ry_offset(ry2, ry_offset_deg)
                else:
                    ry2 = ry

                # recompute compensation with refined ray-consistent yaw
                comp2 = 0.0
                yaw_comp_deg2 = None
                if comp_model_k is not None:
                    yaw_comp_deg2 = wrap_deg(
                        math.degrees(wrap_pi(ry2 - theta_ray2 - math.pi)) + 90.0
                    )
                    comp2 = float(_predict_compensation(yaw_comp_deg2, comp_model_k, z_raw=z_est))

                z2 = max(min_z, z_est - comp2)

                # backproject with corrected u (and bottom-consistent v)
                x2, y_bbox2, z2 = backproject_uv_depth(u2, v2, z2, P2)
                y2 = y_bbox2

                used_ground_y2 = False

                if float(fixed_ground_y) > 0.0:
                    y2 = float(fixed_ground_y)
                    used_ground_y2 = True

                elif front_w2s_meta is not None:
                    y_gp2 = _ground_y_from_kitti_xz(x2, z2, front_w2s_meta)
                    if y_gp2 is not None:
                        y2 = y_gp2
                        used_ground_y2 = True
                if tightfit_enable:
                    if (
                        proj_mode_l == "pred_reproj_geom_x"
                        and use_xonly_grounded_refine
                        and (u_geom_target is not None)
                        and used_ground_y2
                        and (front_w2s_meta is not None)
                    ):
                        # near/off-axis case:
                        # directly refine x so that projected 3D geometric-center u
                        # approaches the perspective-corrected target, while keeping
                        # z fixed and y on the ground plane.
                        x2, y2, z2 = solve_translation_geom_center_x_grounded(
                            P2=P2,
                            bbox_xyxy=(x1o, y1o, x2o, y2o),
                            dims_hwl=(h3d, w3d, l3d),
                            ry=ry2,
                            x0=x2,
                            z_fixed=z2,
                            front_w2s=front_w2s_meta,
                            u_geom_target=float(u_geom_target),
                            k_smooth=float(tightfit_k),
                            max_nfev=max(8, min(int(tightfit_max_nfev), 15)),
                            w_geom=1.0,
                            w_side=0.30,
                            w_bottom=0.15,
                            w_x_prior=0.05,
                        )

                        uv_geom2 = project_pred_geom_uv(P2, x2, y2, z2, h3d)
                        if (uv_geom2 is not None) and np.all(np.isfinite(uv_geom2)):
                            u2 = float(uv_geom2[0])

                    elif used_ground_y2 and (front_w2s_meta is not None):
                        # keep the existing light support-line stabilization for
                        # non-near or non-pred_reproj_geom_x cases
                        x2, y2, z2 = solve_translation_metric_plane_support_xz(
                            P2=P2,
                            bbox_xyxy=(x1o, y1o, x2o, y2o),
                            dims_hwl=(h3d, w3d, l3d),
                            ry=ry2,
                            x0=x2,
                            z0=z2,
                            front_w2s=front_w2s_meta,
                            k_tangent=max(60.0, float(tightfit_k)),
                            k_top=float(tightfit_k),
                            max_nfev=max(int(tightfit_max_nfev), 25),
                            w_center_x=(0.10 if proj_mode_l == "pred_reproj_geom_x" else 0.0),
                        )
                    else:
                        x2, y2, z2 = solve_translation_tightfit(
                            P2=P2,
                            bbox_xyxy=(x1o, y1o, x2o, y2o),
                            dims_hwl=(h3d, w3d, l3d),
                            ry=ry2,
                            t0_xyz=(x2, y2, z2),
                            k_smooth=float(tightfit_k),
                            max_nfev=int(tightfit_max_nfev),
                        )

                # accept this iteration
                u, v = u2, v2
                theta_ray = theta_ray2
                ry = ry2
                comp = float(comp2)
                yaw_comp_deg = yaw_comp_deg2
                x, y, z = x2, y2, z2
                proj_src = proj_src2
        # ---- alpha (use refined ray) ----
        theta_ray_ref = math.atan2(x, z)
        alpha = wrap_pi(ry - theta_ray_ref)

        # cache final corrected 3D center projection / projected 2D extent for visualization + CSV export
        pred_geom_uv = project_pred_geom_uv(P2, x, y, z, h3d)
        pred_bottom_uv = project_pred_bottom_uv(P2, x, y, z)
        pred_proj_bbox = project_pred_proj_bbox_info(
            P2=P2,
            x=x, y=y, z=z,
            h=h3d, w=w3d, l=l3d, ry=ry,
            k_smooth=float(tightfit_k),
        )
        pred_vis_cache[k] = {
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "ry": float(ry),
            "h": float(h3d),
            "w": float(w3d),
            "l": float(l3d),
            "geom_uv": pred_geom_uv,
            "bottom_uv": pred_bottom_uv,
            "proj_bbox": pred_proj_bbox,
            "proj_src": str(proj_src),
            "debug": {
                "proj_mode_req": str(proj_center_source),
                "proj_src_init": str(proj_src_init),
                "proj_src_final": str(proj_src),
                "entered_second_pass": int(bool(feedback_used)),
                "feedback_iters_used": int(feedback_iters_used),
                "u_init": float(u_init),
                "v_init": float(v_init),
                "u_final": float(u),
                "v_final": float(v),
                "u_bbox_ctr": float(u_bbox_ctr),
                "v_bbox_ctr": float(v_bbox_ctr),
                "theta_ray_init": float(theta_ray_init),
                "theta_ray_final_used": float(theta_ray),
                "theta_ray_final_from_xz": float(theta_ray_ref),
                "ry_init": float(ry_init),
                "ry_final": float(ry),
                "yaw_comp_deg_init": float(yaw_comp_deg_init),
                "yaw_comp_deg_final": float(yaw_comp_deg) if yaw_comp_deg is not None else float("nan"),
                "comp_init": float(comp_init),
                "comp_final": float(comp),
                "z_est": float(z_est),
                "x_pre_fb": float(x_pre_fb),
                "y_pre_fb": float(y_pre_fb),
                "z_pre_fb": float(z_pre_fb),
                "x_final": float(x),
                "y_final": float(y),
                "z_final": float(z),
                "used_ground_y": int(bool(used_ground_y)),
                "pred_geom_u": float(pred_geom_uv[0]) if pred_geom_uv is not None else float("nan"),
                "pred_geom_v": float(pred_geom_uv[1]) if pred_geom_uv is not None else float("nan"),
                "pred_bottom_u": float(pred_bottom_uv[0]) if pred_bottom_uv is not None else float("nan"),
                "pred_bottom_v": float(pred_bottom_uv[1]) if pred_bottom_uv is not None else float("nan"),
                "pred_boxctr_u": float(pred_proj_bbox.get("cx", float("nan"))) if isinstance(pred_proj_bbox, dict) else float("nan"),
                "pred_boxctr_v": float(pred_proj_bbox.get("cy", float("nan"))) if isinstance(pred_proj_bbox, dict) else float("nan"),
                "bbox_ctr_dx_from_cx_px": float(u_bbox_ctr - float(K["cx"])),
                "bbox_ctr_dy_from_cy_px": float(v_bbox_ctr - float(K["cy"])),
                "pred_geom_dx_from_cx_px": (float(pred_geom_uv[0]) - float(K["cx"])) if pred_geom_uv is not None else float("nan"),
                "pred_geom_dy_from_cy_px": (float(pred_geom_uv[1]) - float(K["cy"])) if pred_geom_uv is not None else float("nan"),
                "pred_geom_center_dist_px": (
                    float(np.hypot(float(pred_geom_uv[0]) - float(K["cx"]), float(pred_geom_uv[1]) - float(K["cy"])))
                    if pred_geom_uv is not None else float("nan")
                ),
            },
        }

        # truncation/occlusion: unknown -> set 0
        trunc = 0.0
        occ = 0

        line = (
            f"Car {trunc:.2f} {occ:d} {alpha:.6f} "
            f"{x1o:.2f} {y1o:.2f} {x2o:.2f} {y2o:.2f} "
            f"{h3d:.2f} {w3d:.2f} {l3d:.2f} "
            f"{x:.2f} {y:.2f} {z:.2f} "
            f"{ry:.6f} {score:.6f}"
        )
        lines.append(line)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines) + "\n")
        else:
            # important: create empty file if no detections
            f.write("")
    
    # ----------------------------
    # Pred meta CSV rows (optional)
    # ----------------------------
    if pred_meta_rows is not None:
        for k, (x1, y1, x2, y2, yolo_conf) in enumerate(raw):
            v = pred_by_k.get(k, {}) if isinstance(pred_by_k, dict) else {}
            pred_cache = pred_vis_cache.get(k, {}) if isinstance(pred_vis_cache, dict) else {}
            proj_bbox = pred_cache.get("proj_bbox", None) if isinstance(pred_cache, dict) else None
            dbg = pred_cache.get("debug", {}) if isinstance(pred_cache, dict) else {}

            bbox_h_px = float(max(0.0, float(y2) - float(y1)))
            proj_h_px = float(proj_bbox.get("h_px", float("nan"))) if isinstance(proj_bbox, dict) else float("nan")
            proj_h_diff_px = float(proj_h_px - bbox_h_px) if math.isfinite(proj_h_px) else float("nan")

            pred_bbox = (float(x1), float(y1), float(x2), float(y2))
            best_gt_2d, best_iou_2d = find_best_gt_by_iou(pred_bbox, gt_objs)
            gt_hline = None
            gt_proj_h_px = float("nan")
            gt_proj_h_diff_px = float("nan")
            if (best_gt_2d is not None) and (best_iou_2d >= float(vis_iou_thres)):
                #gt_hline = project_gt_heightline_info(best_gt_2d, P2)
                if isinstance(gt_hline, dict):
                    gt_proj_h_px = float(gt_hline.get("h_px", float("nan")))
                    if math.isfinite(gt_proj_h_px):
                        gt_proj_h_diff_px = float(gt_proj_h_px - bbox_h_px)

            pred_meta_rows.append({
                "image_id": img_path.stem,
                "pred_id": int(k),

                # vehicle type (post-thresholded)
                "veh_type": str(v.get("veh_label", "unknown")),
                "veh_conf": float(v.get("veh_prob", 0.0)),

                # raw backend output (debug)
                "veh_raw": str(v.get("veh_raw", "")),
                "veh_raw_conf": float(v.get("veh_raw_p", 0.0)),
                "pred_center_source": str(pred_center_source),

                # YOLO det info
                "yolo_conf": float(yolo_conf),
                "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),

                # predicted 3D pose / dims
                "pred_x": float(pred_cache.get("x", float("nan"))),
                "pred_y": float(pred_cache.get("y", float("nan"))),
                "pred_z": float(pred_cache.get("z", float("nan"))),
                "pred_ry": float(pred_cache.get("ry", float("nan"))),
                "pred_h": float(pred_cache.get("h", float("nan"))),
                "pred_w": float(pred_cache.get("w", float("nan"))),
                "pred_l": float(pred_cache.get("l", float("nan"))),

                # projected 3D-box extent in image
                "pred_proj_bbox_x1": float(proj_bbox.get("x1", float("nan"))) if isinstance(proj_bbox, dict) else float("nan"),
                "pred_proj_bbox_y1": float(proj_bbox.get("y1", float("nan"))) if isinstance(proj_bbox, dict) else float("nan"),
                "pred_proj_bbox_x2": float(proj_bbox.get("x2", float("nan"))) if isinstance(proj_bbox, dict) else float("nan"),
                "pred_proj_bbox_y2": float(proj_bbox.get("y2", float("nan"))) if isinstance(proj_bbox, dict) else float("nan"),
                "pred_proj_ctr_u": float(proj_bbox.get("cx", float("nan"))) if isinstance(proj_bbox, dict) else float("nan"),
                "pred_proj_ctr_v": float(proj_bbox.get("cy", float("nan"))) if isinstance(proj_bbox, dict) else float("nan"),
                "pred_proj_w_px": float(proj_bbox.get("w_px", float("nan"))) if isinstance(proj_bbox, dict) else float("nan"),
                "pred_proj_h_px": proj_h_px,
                "pred_bbox_h_px": bbox_h_px,
                "pred_proj_h_diff_px": proj_h_diff_px,
                "pred_proj_h_abs_diff_px": abs(proj_h_diff_px) if math.isfinite(proj_h_diff_px) else float("nan"),
                "proj_src": str(pred_cache.get("proj_src", "")),
                "proj_mode_req": str(dbg.get("proj_mode_req", "")),
                "proj_src_init": str(dbg.get("proj_src_init", "")),
                "proj_src_final": str(dbg.get("proj_src_final", "")),
                "entered_second_pass": int(dbg.get("entered_second_pass", 0) or 0),
                "feedback_iters_used": int(dbg.get("feedback_iters_used", 0) or 0),
                "u_init": float(dbg.get("u_init", float("nan"))),
                "v_init": float(dbg.get("v_init", float("nan"))),
                "u_final": float(dbg.get("u_final", float("nan"))),
                "v_final": float(dbg.get("v_final", float("nan"))),
                "u_bbox_ctr": float(dbg.get("u_bbox_ctr", float("nan"))),
                "v_bbox_ctr": float(dbg.get("v_bbox_ctr", float("nan"))),
                "theta_ray_init": float(dbg.get("theta_ray_init", float("nan"))),
                "theta_ray_final_used": float(dbg.get("theta_ray_final_used", float("nan"))),
                "theta_ray_final_from_xz": float(dbg.get("theta_ray_final_from_xz", float("nan"))),
                "ry_init": float(dbg.get("ry_init", float("nan"))),
                "ry_final": float(dbg.get("ry_final", float("nan"))),
                "yaw_comp_deg_init": float(dbg.get("yaw_comp_deg_init", float("nan"))),
                "yaw_comp_deg_final": float(dbg.get("yaw_comp_deg_final", float("nan"))),
                "comp_init": float(dbg.get("comp_init", float("nan"))),
                "comp_final": float(dbg.get("comp_final", float("nan"))),
                "z_est": float(dbg.get("z_est", float("nan"))),
                "x_pre_fb": float(dbg.get("x_pre_fb", float("nan"))),
                "y_pre_fb": float(dbg.get("y_pre_fb", float("nan"))),
                "z_pre_fb": float(dbg.get("z_pre_fb", float("nan"))),
                "x_final": float(dbg.get("x_final", float("nan"))),
                "y_final": float(dbg.get("y_final", float("nan"))),
                "z_final": float(dbg.get("z_final", float("nan"))),
                "used_ground_y": int(dbg.get("used_ground_y", 0) or 0),
                "pred_geom_u": float(dbg.get("pred_geom_u", float("nan"))),
                "pred_geom_v": float(dbg.get("pred_geom_v", float("nan"))),
                "pred_bottom_u": float(dbg.get("pred_bottom_u", float("nan"))),
                "pred_bottom_v": float(dbg.get("pred_bottom_v", float("nan"))),
                "pred_boxctr_u": float(dbg.get("pred_boxctr_u", float("nan"))),
                "pred_boxctr_v": float(dbg.get("pred_boxctr_v", float("nan"))),
                "bbox_ctr_dx_from_cx_px": float(dbg.get("bbox_ctr_dx_from_cx_px", float("nan"))),
                "bbox_ctr_dy_from_cy_px": float(dbg.get("bbox_ctr_dy_from_cy_px", float("nan"))),
                "pred_geom_dx_from_cx_px": float(dbg.get("pred_geom_dx_from_cx_px", float("nan"))),
                "pred_geom_dy_from_cy_px": float(dbg.get("pred_geom_dy_from_cy_px", float("nan"))),
                "pred_geom_center_dist_px": float(dbg.get("pred_geom_center_dist_px", float("nan"))),

                # GT-centered vertical height line (through GT 3D geometric-center projection)
                "gt_match_iou2d": float(best_iou_2d),
                "gt_proj_ctr_u": float(gt_hline.get("u_gt", float("nan"))) if isinstance(gt_hline, dict) else float("nan"),
                "gt_proj_ctr_v": float(gt_hline.get("v_gt", float("nan"))) if isinstance(gt_hline, dict) else float("nan"),
                "gt_proj_top_u": float(gt_hline.get("u_top", float("nan"))) if isinstance(gt_hline, dict) else float("nan"),
                "gt_proj_top_v": float(gt_hline.get("v_top", float("nan"))) if isinstance(gt_hline, dict) else float("nan"),
                "gt_proj_bottom_u": float(gt_hline.get("u_bottom", float("nan"))) if isinstance(gt_hline, dict) else float("nan"),
                "gt_proj_bottom_v": float(gt_hline.get("v_bottom", float("nan"))) if isinstance(gt_hline, dict) else float("nan"),
                "gt_proj_h_px": gt_proj_h_px,
                "gt_proj_h_diff_px": gt_proj_h_diff_px,
                "gt_proj_h_abs_diff_px": abs(gt_proj_h_diff_px) if math.isfinite(gt_proj_h_diff_px) else float("nan"),
            })

    # ----------------------------
    # Visualization (optional)
    # ----------------------------
    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------
        # (A) 2D-only visualization (vis/bbox2d/*.png)
        # -------------------------------------------------
        vis2d_dir = vis_dir / "bbox2d"
        vis2d_dir.mkdir(parents=True, exist_ok=True)
        vis2d = img.copy()

        gt_depth_lines = ["GT depth (m)"]
        if gt_objs:
            gt_sorted = sorted(gt_objs, key=lambda o: float(o.z))
            max_show = 8
            for i, g in enumerate(gt_sorted[:max_show]):
                gt_depth_lines.append(f"GT{i:02d}: {float(g.z):.2f}")
            extra = len(gt_sorted) - max_show
            if extra > 0:
                gt_depth_lines.append(f"... +{extra} more")
        else:
            gt_depth_lines.append("none")

        center_diff_lines = ["2D ctr / Pred3D ctr / GT3D ctr"]
        max_center_lines = 10

        # draw ALL predicted 2D boxes in green
        # yellow      = raw 2D bbox center
        # magenta     = predicted corrected 3D geometric center projection
        # light blue  = GT 3D geometric center projection
        for k, (x1, y1, x2, y2, _score) in enumerate(raw):
            x1i = int(round(max(0.0, x1)))
            y1i = int(round(max(0.0, y1)))
            x2i = int(round(max(x1i + 1, x2)))
            y2i = int(round(max(y1i + 1, y2)))

            cv2.rectangle(vis2d, (x1i, y1i), (x2i, y2i), (0, 255, 0), 1)

            pred_u = 0.5 * (float(x1i) + float(x2i))
            pred_v = 0.5 * (float(y1i) + float(y2i))

            # raw 2D bbox center = yellow
            cv2.circle(
                vis2d,
                (int(round(pred_u)), int(round(pred_v))),
                2,
                (0, 255, 255),
                -1,
                lineType=cv2.LINE_AA,
            )
            cv2.circle(
                vis2d,
                (int(round(pred_u)), int(round(pred_v))),
                4,
                (0, 0, 0),
                1,
                lineType=cv2.LINE_AA,
            )

            pred_bbox = (float(x1i), float(y1i), float(x2i), float(y2i))
            best_gt_2d, best_iou_2d = find_best_gt_by_iou(pred_bbox, gt_objs)

            pred3d_line = "pred3dc=n/a"
            gt3d_line = "gt3dc=n/a"

            # predicted corrected 3D geometric center projection = magenta
            pred_cache = pred_vis_cache.get(k, None)
            if isinstance(pred_cache, dict):
                pred_geom_uv = pred_cache.get("geom_uv", None)
                if pred_geom_uv is not None:
                    pd3_u, pd3_v = float(pred_geom_uv[0]), float(pred_geom_uv[1])
                    if math.isfinite(pd3_u) and math.isfinite(pd3_v):
                        cv2.circle(
                            vis2d,
                            (int(round(pd3_u)), int(round(pd3_v))),
                            2,
                            (255, 0, 255),
                            -1,
                            lineType=cv2.LINE_AA,
                        )
                        cv2.circle(
                            vis2d,
                            (int(round(pd3_u)), int(round(pd3_v))),
                            4,
                            (0, 0, 0),
                            1,
                            lineType=cv2.LINE_AA,
                        )
                        pred3d_line = f"pred3dc=({pd3_u:.1f},{pd3_v:.1f})"

                        # optional link from raw 2D center to corrected predicted 3D center
                        cv2.line(
                            vis2d,
                            (int(round(pred_u)), int(round(pred_v))),
                            (int(round(pd3_u)), int(round(pd3_v))),
                            (255, 0, 255),
                            1,
                            lineType=cv2.LINE_AA,
                        )

            line = f"det{k:02d}: pd2d=({pred_u:.1f},{pred_v:.1f})  {pred3d_line}  {gt3d_line}"

            if (best_gt_2d is not None) and (best_iou_2d >= float(vis_iou_thres)):
                gt_ctr_uv = project_gt_center_uv(best_gt_2d, P2, mode="gt_geom")
                if gt_ctr_uv is not None:
                    gt_u, gt_v = gt_ctr_uv

                    # GT 3D geometric center projection = light blue
                    cv2.circle(
                        vis2d,
                        (int(round(gt_u)), int(round(gt_v))),
                        2,
                        (255, 255, 128),
                        -1,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.circle(
                        vis2d,
                        (int(round(gt_u)), int(round(gt_v))),
                        4,
                        (0, 0, 0),
                        1,
                        lineType=cv2.LINE_AA,
                    )

                    gt3d_line = f"gt3dc=({gt_u:.1f},{gt_v:.1f})"

                    if isinstance(pred_cache, dict):
                        pred_geom_uv = pred_cache.get("geom_uv", None)
                        if pred_geom_uv is not None:
                            pd3_u, pd3_v = float(pred_geom_uv[0]), float(pred_geom_uv[1])
                            if math.isfinite(pd3_u) and math.isfinite(pd3_v):
                                du_pd = float(pd3_u - pred_u)
                                dv_pd = float(pd3_v - pred_v)
                                du_gt = float(pd3_u - gt_u)
                                dv_gt = float(pd3_v - gt_v)
                                line = (
                                    f"det{k:02d}: "
                                    f"pd2d=({pred_u:.1f},{pred_v:.1f})  "
                                    f"pred3d=({pd3_u:.1f},{pd3_v:.1f})  "
                                    f"gt3d=({gt_u:.1f},{gt_v:.1f})  "
                                    f"d23=({du_pd:+.1f},{dv_pd:+.1f})  "
                                    f"d3g=({du_gt:+.1f},{dv_gt:+.1f})"
                                )
                            else:
                                line = (
                                    f"det{k:02d}: "
                                    f"pd2d=({pred_u:.1f},{pred_v:.1f})  "
                                    f"{pred3d_line}  {gt3d_line}"
                                )
                    else:
                        line = (
                            f"det{k:02d}: "
                            f"pd2d=({pred_u:.1f},{pred_v:.1f})  "
                            f"{pred3d_line}  {gt3d_line}"
                        )

            if len(center_diff_lines) < (max_center_lines + 1):
                center_diff_lines.append(line)

        if len(raw) > max_center_lines:
            center_diff_lines.append(f"... +{len(raw) - max_center_lines} more")

        draw_text_lines(
            vis2d,
            8,
            98,
            center_diff_lines,
            color=(255, 255, 255),
            bg=(24, 24, 24),
            font_scale=0.48,
            thickness=1,
        )
        draw_text_panel_top_right(vis2d, gt_depth_lines)
        out_2d = vis2d_dir / f"{img_path.stem}.png"
        cv2.imwrite(str(out_2d), vis2d)

        # -------------------------------------------------
        # (A-2) 2D bbox + projected 3D height (vis/2dbbox_h/*.png)
        # -------------------------------------------------
        vis2dh_dir = vis_dir / "2dbbox_h"
        vis2dh_dir.mkdir(parents=True, exist_ok=True)
        vis2dh = img.copy()

        height_lines = ["GT-center vertical projected height vs 2D bbox"]
        max_height_lines = 10

        for k, (x1, y1, x2, y2, _score) in enumerate(raw):
            x1i = int(round(max(0.0, x1)))
            y1i = int(round(max(0.0, y1)))
            x2i = int(round(max(x1i + 1, x2)))
            y2i = int(round(max(y1i + 1, y2)))
            bbox_h_px = float(max(0, y2i - y1i))

            cv2.rectangle(vis2dh, (x1i, y1i), (x2i, y2i), (0, 255, 0), 1)

            pred_bbox = (float(x1i), float(y1i), float(x2i), float(y2i))
            best_gt_2d, best_iou_2d = find_best_gt_by_iou(pred_bbox, gt_objs)
            line = f"det{k:02d}: gt_proj_h=n/a  box_h={bbox_h_px:.1f}px  diff=n/a"

            if (best_gt_2d is not None) and (best_iou_2d >= float(vis_iou_thres)):
                #gt_hline = project_gt_heightline_info(best_gt_2d, P2)
                if isinstance(gt_hline, dict):
                    u_gt = float(gt_hline.get("u_gt", float("nan")))
                    v_gt = float(gt_hline.get("v_gt", float("nan")))
                    u_line = float(gt_hline.get("u_line", float("nan")))
                    v_top = float(gt_hline.get("v_top", float("nan")))
                    v_bot = float(gt_hline.get("v_bottom", float("nan")))
                    gh = float(gt_hline.get("h_px", float("nan")))
                    if all(math.isfinite(v) for v in [u_gt, v_gt, u_line, v_top, v_bot, gh]):
                        diff = float(gh - bbox_h_px)
                        cx_line = int(round(u_line))
                        y_top = int(round(v_top))
                        y_bot = int(round(v_bot))
                        tick = 8

                        # true GT-centered vertical height line
                        cv2.line(vis2dh, (cx_line, y_top), (cx_line, y_bot), (255, 255, 128), 2, lineType=cv2.LINE_AA)
                        cv2.line(vis2dh, (cx_line - tick, y_top), (cx_line + tick, y_top), (255, 255, 128), 2, lineType=cv2.LINE_AA)
                        cv2.line(vis2dh, (cx_line - tick, y_bot), (cx_line + tick, y_bot), (255, 255, 128), 2, lineType=cv2.LINE_AA)
                        cv2.circle(vis2dh, (cx_line, y_top), 3, (255, 255, 128), -1, lineType=cv2.LINE_AA)
                        cv2.circle(vis2dh, (cx_line, y_bot), 3, (255, 255, 128), -1, lineType=cv2.LINE_AA)

                        # GT 3D geometric-center projection point
                        cv2.circle(vis2dh, (int(round(u_gt)), int(round(v_gt))), 3, (0, 255, 255), -1, lineType=cv2.LINE_AA)
                        cv2.circle(vis2dh, (int(round(u_gt)), int(round(v_gt))), 5, (0, 0, 0), 1, lineType=cv2.LINE_AA)

                        line = f"det{k:02d}: gt_proj_h={gh:.1f}px  box_h={bbox_h_px:.1f}px  diff={diff:+.1f}px"

            if len(height_lines) < (max_height_lines + 1):
                height_lines.append(line)

        if len(raw) > max_height_lines:
            height_lines.append(f"... +{len(raw) - max_height_lines} more")

        draw_text_lines(
            vis2dh,
            8,
            58,
            height_lines,
            color=(255, 255, 255),
            bg=(24, 24, 24),
            font_scale=0.52,
            thickness=1,
        )
        out_2dh = vis2dh_dir / f"{img_path.stem}.png"
        cv2.imwrite(str(out_2dh), vis2dh)

        # -------------------------------------------------
        # (B) 3D-only visualization (vis/*.png) + BEV
        # -------------------------------------------------
        vis3d = img.copy()

        bev_dir = vis_dir / "bev"
        bev_dir.mkdir(parents=True, exist_ok=True)
        bev_iou_dir = vis_dir / "bev_iou"
        bev_iou_dir.mkdir(parents=True, exist_ok=True)
        fr_iou_out_dir = vis_dir / "fr_iou"
        fr_iou_out_dir.mkdir(parents=True, exist_ok=True)
        side_iou_out_dir = vis_dir / "side_iou"
        side_iou_out_dir.mkdir(parents=True, exist_ok=True)

        bev_objs = []
        vis3d_top_right_lines = ["Matched true depth (m)"]

        # draw 3D boxes (pred=green, matched GT=red)
        for k, (x1, y1, x2, y2, score) in enumerate(raw):
            x1o = max(0.0, x1)
            y1o = max(0.0, y1)
            x2o = max(x1o + 1.0, x2)
            y2o = max(y1o + 1.0, y2)
            pred_bbox = (x1o, y1o, x2o, y2o)

            # find best GT match by 2D IoU
            best_iou = 0.0
            best_gt: Optional[KittiObj] = None
            for g in gt_objs:
                iou = iou_xyxy(pred_bbox, (g.x1, g.y1, g.x2, g.y2))
                if iou > best_iou:
                    best_iou = iou
                    best_gt = g

            fixed_hwl_k = resolve_fixed_dims_hwl_for_k(pred_by_k, k, default_hwl=fixed_dims_hwl)
            comp_model_k = resolve_comp_model_for_k(pred_by_k, k, comp_models_by_type or {}, default_comp_model)

            proj_center_iou_thres = (
                float(proj_use_gt_center_iou)
                if float(proj_use_gt_center_iou) >= 0.0
                else float(vis_iou_thres)
            )

            # ---- choose initial (u,v) for theta_ray / backproject ----
            proj_mode0 = str(proj_center_source).strip().lower()
            if proj_mode0 in ("pred_reproj_bottom", "pred_reproj_geom_x"):
                proj_mode0 = "bbox_bottom"

            u, v, proj_src = choose_proj_uv(
                mode=proj_mode0,
                x1=x1, y1=y1, x2=x2, y2=y2,
                P2=P2,
                best_gt=best_gt,
                best_iou=best_iou,
                iou_thres=proj_center_iou_thres,
            )

            theta_ray = theta_ray_from_u(u, K)

            if (k in pred_by_k):
                p = pred_by_k[k]
                ry = resolve_ry(p, theta_ray, allow_alpha_fallback=True)
                ry = apply_ry_offset(ry, ry_offset_deg)
                h3d, w3d, l3d = resolve_dims(p, dims_source, fixed_hwl_k)
            else:
                p = None
                ry = 0.0
                h3d, w3d, l3d = fixed_hwl_k

            gt_hwl_iou_thres = float(pred_use_gt_hwl_iou) if float(pred_use_gt_hwl_iou) >= 0.0 else float(vis_iou_thres)
            if pred_use_gt_hwl and (best_gt is not None) and (best_iou >= gt_hwl_iou_thres):
                h3d, w3d, l3d = float(best_gt.h), float(best_gt.w), float(best_gt.l)

            h_px = max(1.0, (y2 - y1))
            H_used = float(h3d) if (dims_source == "egonet") else float(height_m)

            if depth_use_gt_height and (best_gt is not None) and (best_iou >= float(vis_iou_thres)):
                H_used = float(best_gt.h)

            z_est = float(K["fy"] * H_used / h_px)

            comp = 0.0
            yaw_comp_deg = None
            if comp_model_k is not None:
                theta_ray = theta_ray_from_u(u, K)
                yaw_comp_deg = wrap_deg(math.degrees(wrap_pi(ry - theta_ray - math.pi)) + 90.0)
                comp = float(_predict_compensation(yaw_comp_deg, comp_model_k, z_raw=z_est))

            z = max(min_z, z_est - comp)

            # init from old depth path
            x, y_bbox, z = backproject_uv_depth(u, v, z, P2)
            y = y_bbox

            used_ground_y = False

            if float(fixed_ground_y) > 0.0:
                y = float(fixed_ground_y)
                used_ground_y = True

            elif front_w2s_meta is not None:
                y_gp = _ground_y_from_kitti_xz(x, z, front_w2s_meta)
                if y_gp is not None:
                    y = y_gp
                    used_ground_y = True

            # tight-fit refine translation
            if tightfit_enable:
                if used_ground_y and (front_w2s_meta is not None):
                    x, y, z = solve_translation_metric_plane_support_xz(
                        P2=P2,
                        bbox_xyxy=(x1o, y1o, x2o, y2o),
                        dims_hwl=(h3d, w3d, l3d),
                        ry=ry,
                        x0=x,
                        z0=z,
                        front_w2s=front_w2s_meta,
                        k_tangent=max(60.0, float(tightfit_k)),
                        k_top=float(tightfit_k),
                        max_nfev=max(int(tightfit_max_nfev), 25),
                        w_center_x=0.0,
                    )
                    used_ground_y = True
                else:
                    x, y, z = solve_translation_tightfit(
                        P2=P2,
                        bbox_xyxy=(x1o, y1o, x2o, y2o),
                        dims_hwl=(h3d, w3d, l3d),
                        ry=ry,
                        t0_xyz=(x, y, z),
                        k_smooth=float(tightfit_k),
                        max_nfev=int(tightfit_max_nfev),
                    )
            # ---- optional feedback loop: refine x-ray by predicted reprojection ----
            proj_mode_l = str(proj_center_source).strip().lower()
            if proj_mode_l in ("pred_reproj_bottom", "pred_reproj_geom_x"):
                for _fb_iter in range(max(1, int(proj_refine_iters))):
                    uv_ref = project_pred_feedback_uv(
                        mode=proj_mode_l,
                        P2=P2,
                        x=x,
                        y=y,
                        z=z,
                        h=h3d,
                        w=w3d,
                        l=l3d,
                        ry=ry,
                        k_smooth=float(tightfit_k),
                    )
                    if uv_ref is None:
                        break

                    u2, v2, proj_src2 = uv_ref
                    use_xonly_grounded_refine = False
                    u_geom_target = None

                    if proj_mode_l == "pred_reproj_geom_x":
                        # ---------------------------------------------------------
                        # Two-mode update:
                        #   (A) near/off-axis : use perspective-compensated target u
                        #       and solve x only on the ground plane
                        #   (B) otherwise     : keep the far-range heading-based step
                        #
                        # Key idea for near range:
                        #   u_geom_target = u_bbox_ctr - (u_proj_boxctr - u_proj_geom)
                        #
                        # i.e. subtract the CURRENT perspective offset between
                        # projected-box center and projected geometric-center, instead
                        # of assuming the residual is mainly along projected heading.
                        # ---------------------------------------------------------

                        # current projected 3D-box 2D center (from uv_ref)
                        u_proj_boxctr = float(u2)
                        v_proj_boxctr = 0.5 * (float(y1o) + float(y2o))

                        # current predicted 3D geometric center projection
                        uv_geom = project_pred_geom_uv(P2, x, y, z, h3d)
                        if uv_geom is None:
                            break
                        u_proj_geom = float(uv_geom[0])
                        v_proj_geom = float(uv_geom[1])

                        # observed YOLO bbox center
                        u_bbox_ctr = 0.5 * (float(x1o) + float(x2o))
                        v_bbox_ctr = 0.5 * (float(y1o) + float(y2o))

                        # current perspective offset between projected 3D-box center
                        # and projected 3D geometric center
                        du_geom = float(u_proj_boxctr - u_proj_geom)

                        # fallback target: bbox center corrected by current perspective offset
                        u_fallback_geom = float(u_bbox_ctr - du_geom)

                        bw_obs = max(10.0, float(x2o - x1o))
                        offaxis_px = abs(float(u_bbox_ctr - K["cx"]))
                        ray_bbox_deg = abs(math.degrees(theta_ray_from_u(float(u_bbox_ctr), K)))

                        near_offaxis = (
                            (float(z) < 30.0) and (
                                (offaxis_px > 0.12 * float(img.shape[1]))
                                or (ray_bbox_deg > 6.0)
                                or (bw_obs > 80.0)
                            )
                        )
                        #near_offaxis = (float(z) < 40.0)
                        if near_offaxis:
                            # near-range main target:
                            # bbox center minus current projected perspective shift
                            lambda_persp = 0.9
                            u_geom_target = float(u_bbox_ctr - lambda_persp * du_geom)

                            # keep target within a conservative search band
                            u_pad = max(14.0, 0.35 * bw_obs)
                            u_geom_target = max(
                                float(x1o) - u_pad,
                                min(float(x2o) + u_pad, float(u_geom_target))
                            )

                            u2 = float(u_geom_target)
                            use_xonly_grounded_refine = True

                        else:
                            # far / weak-perspective case:
                            # keep heading-based step from the previous patch
                            rho_head = max(1.0, 0.5 * float(l3d))
                            c_geom = np.array(
                                [[float(x), float(y) - 0.5 * float(h3d), float(z)]],
                                dtype=np.float64
                            )
                            c_head = c_geom + np.array(
                                [[rho_head * math.cos(float(ry)), 0.0, -rho_head * math.sin(float(ry))]],
                                dtype=np.float64
                            )

                            uv_head = project_points(P2, c_head)[0]
                            if not np.all(np.isfinite(uv_head)):
                                break

                            t = np.array(
                                [float(uv_head[0] - u_proj_geom), float(uv_head[1] - v_proj_geom)],
                                dtype=np.float64
                            )
                            t_norm = float(np.linalg.norm(t))

                            eps_dir = 1e-6
                            if t_norm <= eps_dir:
                                u2 = float(u_fallback_geom)
                            else:
                                t_hat = t / t_norm

                                heading_v_conf = min(
                                    1.0,
                                    abs(t_hat[1]) / max(abs(t_hat[0]) + abs(t_hat[1]), 1e-6)
                                )
                                w_geom = 0.15 + 0.55 * heading_v_conf

                                u_ref = float(w_geom * u_proj_geom + (1.0 - w_geom) * u_proj_boxctr)
                                v_ref = float(w_geom * v_proj_geom + (1.0 - w_geom) * v_proj_boxctr)

                                du_obs = float(u_bbox_ctr - u_ref)
                                dv_obs = float(v_bbox_ctr - v_ref)
                                d = np.array([du_obs, dv_obs], dtype=np.float64)

                                if float(np.dot(d, t_hat)) < 0.0:
                                    t_hat = -t_hat

                                tu = float(t_hat[0])
                                tv = float(t_hat[1])

                                w_u = 1.0
                                w_v = 0.25 + 1.25 * heading_v_conf

                                denom = float(w_u * tu * tu + w_v * tv * tv)
                                if denom <= 1e-8:
                                    if abs(tu) > 1e-6:
                                        s = float(du_obs / tu)
                                    else:
                                        s = 0.0
                                else:
                                    s = float((w_u * tu * du_obs + w_v * tv * dv_obs) / denom)

                                max_s_px = max(18.0, 0.45 * bw_obs)
                                s = max(-max_s_px, min(max_s_px, float(s)))

                                u2 = float(u_ref + s * t_hat[0])

                                if not math.isfinite(u2):
                                    u2 = float(u_fallback_geom)

                        # keep bottom-consistent v for backprojection
                        v2 = float(y2o)

                        # clamp search range
                        u_lo = float(x1o) - 0.75 * bw_obs
                        u_hi = float(x2o) + 0.75 * bw_obs
                        u2 = max(u_lo, min(u_hi, float(u2)))

                    # pred_reproj_geom_x 改用幾何步長收斂；其他模式維持原本邏輯
                    if proj_mode_l == "pred_reproj_geom_x":
                        if (_fb_iter > 0) and (abs(float(u2) - float(u)) < 0.25):
                            break
                    else:
                        if (_fb_iter > 0) and (abs(float(u2) - float(u)) < 0.25):
                            break

                    theta_ray2 = theta_ray_from_u(u2, K)

                    if p is not None:
                        ry2 = resolve_ry(p, theta_ray2, allow_alpha_fallback=True)
                        ry2 = apply_ry_offset(ry2, ry_offset_deg)
                    else:
                        ry2 = ry

                    # recompute compensation with refined ray-consistent yaw
                    comp2 = 0.0
                    yaw_comp_deg2 = None
                    if comp_model_k is not None:
                        yaw_comp_deg2 = wrap_deg(
                            math.degrees(wrap_pi(ry2 - theta_ray2 - math.pi)) + 90.0
                        )
                        comp2 = float(_predict_compensation(yaw_comp_deg2, comp_model_k, z_raw=z_est))

                    z2 = max(min_z, z_est - comp2)

                    # backproject with corrected u (and bottom-consistent v)
                    x2, y_bbox2, z2 = backproject_uv_depth(u2, v2, z2, P2)
                    y2 = y_bbox2

                    used_ground_y2 = False

                    if float(fixed_ground_y) > 0.0:
                        y2 = float(fixed_ground_y)
                        used_ground_y2 = True

                    elif front_w2s_meta is not None:
                        y_gp2 = _ground_y_from_kitti_xz(x2, z2, front_w2s_meta)
                        if y_gp2 is not None:
                            y2 = y_gp2
                            used_ground_y2 = True

                    
                    if tightfit_enable:
                        if (
                            proj_mode_l == "pred_reproj_geom_x"
                            and use_xonly_grounded_refine
                            and (u_geom_target is not None)
                            and used_ground_y2
                            and (front_w2s_meta is not None)
                        ):
                            # near/off-axis case:
                            # directly refine x so that projected 3D geometric-center u
                            # approaches the perspective-corrected target, while keeping
                            # z fixed and y on the ground plane.
                            x2, y2, z2 = solve_translation_geom_center_x_grounded(
                                P2=P2,
                                bbox_xyxy=(x1o, y1o, x2o, y2o),
                                dims_hwl=(h3d, w3d, l3d),
                                ry=ry2,
                                x0=x2,
                                z_fixed=z2,
                                front_w2s=front_w2s_meta,
                                u_geom_target=float(u_geom_target),
                                k_smooth=float(tightfit_k),
                                max_nfev=max(8, min(int(tightfit_max_nfev), 15)),
                                w_geom=1.0,
                                w_side=0.30,
                                w_bottom=0.15,
                                w_x_prior=0.05,
                            )

                            uv_geom2 = project_pred_geom_uv(P2, x2, y2, z2, h3d)
                            if (uv_geom2 is not None) and np.all(np.isfinite(uv_geom2)):
                                u2 = float(uv_geom2[0])

                        elif used_ground_y2 and (front_w2s_meta is not None):
                            # keep the existing light support-line stabilization for
                            # non-near or non-pred_reproj_geom_x cases
                            x2, y2, z2 = solve_translation_metric_plane_support_xz(
                                P2=P2,
                                bbox_xyxy=(x1o, y1o, x2o, y2o),
                                dims_hwl=(h3d, w3d, l3d),
                                ry=ry2,
                                x0=x2,
                                z0=z2,
                                front_w2s=front_w2s_meta,
                                k_tangent=max(60.0, float(tightfit_k)),
                                k_top=float(tightfit_k),
                                max_nfev=max(int(tightfit_max_nfev), 25),
                                w_center_x=(0.10 if proj_mode_l == "pred_reproj_geom_x" else 0.0),
                            )
                        else:
                            x2, y2, z2 = solve_translation_tightfit(
                                P2=P2,
                                bbox_xyxy=(x1o, y1o, x2o, y2o),
                                dims_hwl=(h3d, w3d, l3d),
                                ry=ry2,
                                t0_xyz=(x2, y2, z2),
                                k_smooth=float(tightfit_k),
                                max_nfev=int(tightfit_max_nfev),
                            )

                    # accept this iteration
                    u, v = u2, v2
                    theta_ray = theta_ray2
                    ry = ry2
                    comp = float(comp2)
                    yaw_comp_deg = yaw_comp_deg2
                    x, y, z = x2, y2, z2
                    proj_src = proj_src2
            pred_center_iou_thres = float(pred_use_gt_center_iou) if float(pred_use_gt_center_iou) >= 0.0 else float(vis_iou_thres)
            x, y, z, center_src = choose_pred_center_xyz(
                center_mode=pred_center_source,
                x=x, y=y, z=z,
                best_gt=best_gt,
                best_iou=best_iou,
                iou_thres=pred_center_iou_thres,
            )

            if center_src == "bbox" and used_ground_y and tightfit_enable:
                center_src = "bbox_xztan_yground"
            elif center_src == "bbox" and used_ground_y:
                center_src = "bbox_yground"

            if not (
                math.isfinite(x) and math.isfinite(y) and math.isfinite(z) and
                math.isfinite(ry) and math.isfinite(h3d) and math.isfinite(w3d) and math.isfinite(l3d)
            ):
                continue
            if (h3d <= 0.0) or (w3d <= 0.0) or (l3d <= 0.0) or (z <= 0.0):
                continue

            pred_bev_box = {"cls": "pred", "x": x, "z": z, "ry": ry, "L": l3d, "W": w3d}
            bev_objs.append(pred_bev_box)

            draw_box3d(vis3d, P2, h3d, w3d, l3d, x, y, z, ry, color=(0, 255, 0), thickness=2)

            pair_bev_iou = None
            pair_3d_iou = None

            if (best_gt is not None) and (best_iou >= vis_iou_thres) and vis_draw_gt:
                gt_bev_box = {
                    "cls": "gt",
                    "x": best_gt.x,
                    "z": best_gt.z,
                    "ry": best_gt.ry,
                    "L": best_gt.l,
                    "W": best_gt.w,
                }
                gt_box_3d = {
                    "h": best_gt.h,
                    "w": best_gt.w,
                    "l": best_gt.l,
                    "x": best_gt.x,
                    "y": best_gt.y,
                    "z": best_gt.z,
                    "ry": best_gt.ry,
                }
                pred_box_3d = {
                    "h": h3d,
                    "w": w3d,
                    "l": l3d,
                    "x": x,
                    "y": y,
                    "z": z,
                    "ry": ry,
                }

                pair_3d_iou = iou3d_kitti_boxes(pred_box_3d, gt_box_3d)
                bev_objs.append(gt_bev_box)

                draw_box3d(
                    vis3d,
                    P2,
                    best_gt.h, best_gt.w, best_gt.l,
                    best_gt.x, best_gt.y, best_gt.z, best_gt.ry,
                    color=(0, 0, 255),
                    thickness=2,
                )

                out_bev_iou = bev_iou_dir / f"{img_path.stem}_det{k:02d}.png"

                pred_world = None
                gt_world = None
                if front_w2s_meta is not None:
                    pred_world = world_corners_from_kitti_box(h3d, w3d, l3d, x, y, z, ry, front_w2s_meta)
                    gt_world = world_corners_from_kitti_box(
                        best_gt.h, best_gt.w, best_gt.l,
                        best_gt.x, best_gt.y, best_gt.z, best_gt.ry,
                        front_w2s_meta,
                    )
                    pair_bev_iou, inter_area, union_area, _ = bev_iou_world_corners(pred_world, gt_world)

                    bev_iou_img = None
                    if (top_bev_img is not None) and isinstance(top_cam_meta, dict):
                        bev_iou_img = draw_top_bev_from_carla_image(
                            top_bev_img, top_cam_meta,
                            pred_world, gt_world,
                            pair_bev_iou, inter_area, union_area,
                        )
                    if bev_iou_img is None:
                        bev_iou_img, pair_bev_iou = draw_bev_follow_pair(pred_bev_box, gt_bev_box)

                    if (fr_iou_img is not None) and isinstance(fr_iou_cam_meta, dict):
                        fr_pair = draw_camera_iou_pair(
                            fr_iou_img, fr_iou_cam_meta, pred_world, gt_world, title="FR IoU"
                        )
                        if fr_pair is not None:
                            fr_img_out, _fr_iou = fr_pair
                            cv2.imwrite(str(fr_iou_out_dir / f"{img_path.stem}_det{k:02d}.png"), fr_img_out)

                    if (side_iou_img is not None) and isinstance(side_iou_cam_meta, dict):
                        side_pair = draw_camera_iou_pair(
                            side_iou_img, side_iou_cam_meta, pred_world, gt_world, title="Side IoU"
                        )
                        if side_pair is not None:
                            side_img_out, _side_iou = side_pair
                            cv2.imwrite(str(side_iou_out_dir / f"{img_path.stem}_det{k:02d}.png"), side_img_out)
                else:
                    bev_iou_img, pair_bev_iou = draw_bev_follow_pair(pred_bev_box, gt_bev_box)

                bev_iou_right_lines = build_pair_overlay_lines(
                    pred_id=k,
                    pred_z=z,
                    comp=comp,
                    ry=ry,
                    best_gt=best_gt,
                    best_iou=best_iou,
                    vis_iou_thres=vis_iou_thres,
                    pair_bev_iou=pair_bev_iou,
                    pair_3d_iou=pair_3d_iou,
                    vmeta=pred_by_k.get(k, {}),
                    center_src=center_src,
                    comp_angle_deg=yaw_comp_deg,
                )
                draw_text_panel_top_right(bev_iou_img, bev_iou_right_lines)
                cv2.imwrite(str(out_bev_iou), bev_iou_img)

            # text overlay on vis3d
            vmeta = pred_by_k.get(k, {})
            lines = build_pair_overlay_lines(
                pred_id=k,
                pred_z=z,
                comp=comp,
                ry=ry,
                best_gt=best_gt,
                best_iou=best_iou,
                vis_iou_thres=vis_iou_thres,
                pair_bev_iou=pair_bev_iou,
                pair_3d_iou=pair_3d_iou,
                vmeta=vmeta,
                center_src=center_src,
                comp_angle_deg=yaw_comp_deg,
            )
            draw_text_lines(vis3d, int(max(0, x1)), int(max(0, y1)) - 5, lines)

            matched = (best_gt is not None) and (best_iou >= vis_iou_thres)
            if matched:
                vis3d_top_right_lines.append(
                    f"det{k:02d}: GT={float(best_gt.z):.2f}  PD={float(z):.2f}  "
                    f"dZ={float(z - best_gt.z):+.2f}  3DIoU={float(pair_3d_iou or 0.0):.3f}  "
                    f"ctr={center_src.upper()}"
                )
            else:
                vis3d_top_right_lines.append(
                    f"det{k:02d}: GT=n/a  PD={float(z):.2f}  ctr={center_src.upper()}"
                )

        vis3d_right_panel = gt_depth_lines + ([""] + vis3d_top_right_lines[1:] if len(vis3d_top_right_lines) > 1 else [])
        draw_text_panel_top_right(vis3d, vis3d_right_panel)
        out_img = vis_dir / f"{img_path.stem}.png"
        cv2.imwrite(str(out_img), vis3d)

        bev = draw_bev_local(
            bev_objs,
            xlim=(-30.0, 30.0),
            zlim=(0.0, 80.0),
            ppm=10.0,
            img_w=int(img.shape[1]),
            fx=float(K["fx"]),
            draw_fov=True,
        )
        draw_text_panel_top_right(bev, gt_depth_lines)
        out_bev = bev_dir / f"{img_path.stem}.png"
        cv2.imwrite(str(out_bev), bev)
