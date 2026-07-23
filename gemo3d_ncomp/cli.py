"""Command-line interface for the refactored nearest-time NComp pipeline."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

from egonet_ori import load_head as egonet_load_head
try:
    from deep3d_head import load_head as deep3d_load_head
except Exception:
    deep3d_load_head = None
from vehicletype_head import build_vehicle_classifier

from .compensation import load_compensation_model
from .config import (
    K_from_P2,
    _parse_vt_comp_pkl_map,
    build_comp_models_by_vt5,
    read_kitti_calib_P2,
    read_split_ids,
)
from .pipeline import infer_one_image


PRED_META_FIELDS = [
    "image_id", "pred_id", "veh_type", "veh_conf", "veh_raw", "veh_raw_conf",
    "pred_center_source", "yolo_conf", "x1", "y1", "x2", "y2",
    "pred_x", "pred_y", "pred_z", "pred_ry", "pred_h", "pred_w", "pred_l",
    "pred_proj_bbox_x1", "pred_proj_bbox_y1", "pred_proj_bbox_x2", "pred_proj_bbox_y2",
    "pred_proj_ctr_u", "pred_proj_ctr_v", "pred_proj_w_px", "pred_proj_h_px",
    "pred_bbox_h_px", "pred_proj_h_diff_px", "pred_proj_h_abs_diff_px", "proj_src",
    "proj_mode_req", "proj_src_init", "proj_src_final", "entered_second_pass",
    "feedback_iters_used", "u_init", "v_init", "u_final", "v_final",
    "u_bbox_ctr", "v_bbox_ctr", "theta_ray_init", "theta_ray_final_used",
    "theta_ray_final_from_xz", "ry_init", "ry_final", "yaw_comp_deg_init",
    "yaw_comp_deg_final", "comp_init", "comp_final", "z_est", "x_pre_fb",
    "y_pre_fb", "z_pre_fb", "x_final", "y_final", "z_final", "used_ground_y",
    "pred_geom_u", "pred_geom_v", "pred_bottom_u", "pred_bottom_v",
    "pred_boxctr_u", "pred_boxctr_v", "bbox_ctr_dx_from_cx_px",
    "bbox_ctr_dy_from_cy_px", "pred_geom_dx_from_cx_px", "pred_geom_dy_from_cy_px",
    "pred_geom_center_dist_px", "gt_match_iou2d", "gt_proj_ctr_u", "gt_proj_ctr_v",
    "gt_proj_top_u", "gt_proj_top_v", "gt_proj_bottom_u", "gt_proj_bottom_v",
    "gt_proj_h_px", "gt_proj_h_diff_px", "gt_proj_h_abs_diff_px",
]


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    image_dir: Path
    calib_dir: Path
    label_dir: Path
    meta_dir: Path
    top_bev_dir: Path
    front_iou_dir: Path
    side_iou_dir: Path
    split_file: Path

    @classmethod
    def from_args(cls, root: Path, split: str) -> "DatasetPaths":
        training = root / "training"
        return cls(
            root=root,
            image_dir=training / "image_2",
            calib_dir=training / "calib",
            label_dir=training / "label_2",
            meta_dir=training / "meta",
            top_bev_dir=training / "bev_iou",
            front_iou_dir=training / "fr_iou",
            side_iou_dir=training / "side_iou",
            split_file=root / "ImageSets" / f"{split}.txt",
        )


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="GEMO3D monocular 3D vehicle inference with Rank-1 NComp.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--kitti_root", default="/home/e114/KITTI_ROOT", type=str, help="KITTI dataset root")
    ap.add_argument("--split", default="val", help="ImageSets/<split>.txt to run")
    ap.add_argument("--out_dir", default="/home/e114/KITTI_ROOT/out", type=str, help="output root")
    ap.add_argument("--yolo_weights", default="yolo11l.pt", type=str)
    ap.add_argument("--egonet_weights", default="weights/egonet", type=str)
    ap.add_argument("--no_egonet", action="store_true")
    ap.add_argument("--conf", default=0.65, type=float)
    ap.add_argument("--imgsz", default=1280, type=int)
    ap.add_argument("--device", default="cuda", type=str)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--car_names", default="car", type=str)
    ap.add_argument("--dims_source", default="fixed", choices=["fixed", "egonet", "deep3d"])
    ap.add_argument("--deep3d_weights", default="weights/epoch_50_converted.pth", type=str)
    ap.add_argument("--fixed_dims_hwl", default="1.55,1.74,3.86", type=str)
    ap.add_argument("--comp_pkl", default=None, type=str)
    ap.add_argument("--vt_comp_pkl_map", default="", type=str)
    ap.add_argument("--compact_comp_pkl", default=None, type=str)
    ap.add_argument("--sedan_comp_pkl", default=None, type=str)
    ap.add_argument("--suv_comp_pkl", default=None, type=str)
    ap.add_argument("--van_comp_pkl", default=None, type=str)
    ap.add_argument("--use_bbox_center", action="store_true")
    ap.add_argument("--vis_dir", default="", type=str)
    ap.add_argument("--vis_iou", default=0.5, type=float)
    ap.add_argument("--depth_use_gt_height", action="store_true")
    ap.add_argument("--height_m", type=float, default=1.55)
    ap.add_argument("--vis_draw_gt", action="store_true")
    ap.add_argument("--ry_offset_deg", default=0.0, type=float)
    ap.add_argument("--tightfit", action="store_true")
    ap.add_argument("--tightfit_max_nfev", default=15, type=int)
    ap.add_argument("--tightfit_k", default=50.0, type=float)
    ap.add_argument("--vehclf_backend", default="none", type=str, choices=["none", "ort", "paddleclas", "yolo_cls"])
    ap.add_argument("--vehclf_ort_onnx", default="/home/e114/Downloads/vehicletypenet_pruned_onnx_v1.1.0/resnet18_pruned.onnx", type=str)
    ap.add_argument("--vehclf_ort_labels6", default="coupe,sedan,suv,van,large_vehicle,truck", type=str)
    ap.add_argument("--vehclf_ort_bgr", action="store_true")
    ap.add_argument("--vehclf_ort_no_norm", action="store_true")
    ap.add_argument("--vehclf_ort_no_letterbox", action="store_true")
    ap.add_argument("--vehclf_paddle_use_gpu", action="store_true")
    ap.add_argument("--vehclf_paddle_batch", type=int, default=8)
    ap.add_argument("--vehclf_paddle_min_prob", type=float, default=0.3)
    ap.add_argument("--vehclf_paddle_compact_alias", type=str, default="hatchback,estate")
    ap.add_argument("--vehclf_yolo_weights", type=str, default="")
    ap.add_argument("--vehclf_yolo_imgsz", type=int, default=224)
    ap.add_argument("--vehclf_yolo_device", type=str, default="")
    ap.add_argument("--vehclf_yolo_half", action="store_true")
    ap.add_argument("--vehclf_yolo_label_map", type=str, default="hatchback=compact,pickup=compact,sedan=sedan,suv=suv,van=suv")
    ap.add_argument("--vehclf_crop_pad", type=float, default=0.15)
    ap.add_argument("--vehclf_min_prob", type=float, default=0.20)
    ap.add_argument("--pred_meta_csv", default="pred_meta.csv", type=str)
    ap.add_argument("--pred_use_gt_hwl", action="store_true")
    ap.add_argument("--pred_use_gt_hwl_iou", type=float, default=-1.0)
    ap.add_argument("--pred_center_source", default="bbox", choices=["bbox", "gt"])
    ap.add_argument("--pred_use_gt_center_iou", type=float, default=-1.0)
    ap.add_argument("--force_veh_type", default="", type=str)
    ap.add_argument(
        "--proj_center_source", default="bbox_bottom",
        choices=["bbox_bottom", "bbox_center", "gt_bottom", "gt_geom", "pred_reproj_bottom", "pred_reproj_geom_x"],
    )
    ap.add_argument("--proj_refine_iters", type=int, default=3)
    ap.add_argument("--proj_use_gt_center_iou", type=float, default=-1.0)
    ap.add_argument("--fixed_ground_y", type=float, default=-1.0)
    return ap


def parse_hwl(text: str) -> tuple[float, float, float]:
    values = tuple(float(x.strip()) for x in text.split(","))
    if len(values) != 3:
        raise ValueError("--fixed_dims_hwl must be 'h,w,l'")
    return values


def build_vehicle_compensation_models(args) -> tuple[Optional[dict], dict]:
    default_model = load_compensation_model(args.comp_pkl) if args.comp_pkl else None
    overrides = _parse_vt_comp_pkl_map(args.vt_comp_pkl_map)
    for label, arg_name in (
        ("compact", "compact_comp_pkl"), ("sedan", "sedan_comp_pkl"),
        ("suv", "suv_comp_pkl"), ("van", "van_comp_pkl"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            overrides[label] = str(value)
    return default_model, build_comp_models_by_vt5(overrides)


def build_runtime_models(args):
    if YOLO is None:
        raise RuntimeError("ultralytics is unavailable. Install it with: pip install ultralytics")

    detector = YOLO(args.yolo_weights)
    egonet_enabled = not args.no_egonet
    if egonet_enabled:
        if not args.egonet_weights:
            raise ValueError("--egonet_weights is required unless --no_egonet is used")
        egonet_load_head(args.egonet_weights)

    deep3d_enabled = args.dims_source.lower() == "deep3d"
    if deep3d_enabled:
        if deep3d_load_head is None:
            raise RuntimeError("deep3d_head import failed; check PYTHONPATH and dependencies")
        deep3d_load_head(args.deep3d_weights)

    classifier = build_vehicle_classifier(
        args.vehclf_backend,
        ort_onnx=args.vehclf_ort_onnx,
        ort_labels6=args.vehclf_ort_labels6,
        ort_bgr=bool(args.vehclf_ort_bgr),
        ort_no_norm=bool(args.vehclf_ort_no_norm),
        paddle_use_gpu=bool(args.vehclf_paddle_use_gpu),
        paddle_batch=int(args.vehclf_paddle_batch),
        paddle_min_prob=float(args.vehclf_paddle_min_prob),
        paddle_compact_alias=str(args.vehclf_paddle_compact_alias),
        ort_letterbox=not bool(args.vehclf_ort_no_letterbox),
        yolo_weights=str(args.vehclf_yolo_weights),
        yolo_imgsz=int(args.vehclf_yolo_imgsz),
        yolo_device=str(args.vehclf_yolo_device),
        yolo_half=bool(args.vehclf_yolo_half),
        yolo_label_map=str(args.vehclf_yolo_label_map),
    )
    return detector, classifier, egonet_enabled, deep3d_enabled


def write_prediction_metadata(rows: list[dict], destination: str) -> Optional[Path]:
    if not destination.strip():
        return None
    path = Path(destination.strip())
    if path.exists() and path.is_dir():
        path = path / "pred_meta.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=PRED_META_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def run(args) -> None:
    paths = DatasetPaths.from_args(Path(args.kitti_root), args.split)
    if not paths.split_file.is_file():
        raise FileNotFoundError(f"Split file not found: {paths.split_file}")

    image_ids = read_split_ids(paths.split_file)
    output_data_dir = Path(args.out_dir) / "pred" / "data"
    output_data_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = Path(args.vis_dir) if args.vis_dir.strip() else None

    detector, classifier, egonet_enabled, deep3d_enabled = build_runtime_models(args)
    fixed_hwl = parse_hwl(args.fixed_dims_hwl)
    default_comp, comp_by_type = build_vehicle_compensation_models(args)
    car_names = [name.strip().lower() for name in args.car_names.split(",") if name.strip()]

    force_type = args.force_veh_type.strip().lower()
    if force_type and force_type not in {"compact", "sedan", "suv", "van"}:
        raise ValueError("--force_veh_type must be compact, sedan, suv, or van")

    meta_rows = [] if args.pred_meta_csv.strip() else None
    for index, image_id in enumerate(image_ids, start=1):
        image_path = paths.image_dir / f"{image_id}.png"
        calib_path = paths.calib_dir / f"{image_id}.txt"
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image: {image_path}")
        if not calib_path.is_file():
            raise FileNotFoundError(f"Missing calibration: {calib_path}")

        p2 = read_kitti_calib_P2(calib_path)
        intrinsics = K_from_P2(p2)
        infer_one_image(
            img_path=image_path,
            calib_path=calib_path,
            yolo_model=detector,
            car_names=car_names,
            conf_thres=float(args.conf), imgsz=int(args.imgsz),
            yolo_device=str(args.device), use_half=bool(args.half),
            egonet_enable=egonet_enabled, deep3d_enable=deep3d_enabled,
            P2=p2, K=intrinsics, height_m=float(args.height_m),
            dims_source=str(args.dims_source), fixed_dims_hwl=fixed_hwl,
            comp_model=default_comp, out_pred_data_dir=output_data_dir,
            use_bottom_center=not args.use_bbox_center,
            label_path=paths.label_dir / f"{image_id}.txt",
            meta_path=paths.meta_dir / f"{image_id}.json",
            top_bev_img_path=paths.top_bev_dir / f"{image_id}.png",
            fr_iou_img_path=paths.front_iou_dir / f"{image_id}.png",
            side_iou_img_path=paths.side_iou_dir / f"{image_id}.png",
            ry_offset_deg=float(args.ry_offset_deg), vis_dir=vis_dir,
            vis_iou_thres=float(args.vis_iou), vis_draw_gt=bool(args.vis_draw_gt),
            depth_use_gt_height=bool(args.depth_use_gt_height),
            pred_use_gt_hwl=bool(args.pred_use_gt_hwl),
            pred_use_gt_hwl_iou=float(args.pred_use_gt_hwl_iou),
            pred_center_source=str(args.pred_center_source),
            pred_use_gt_center_iou=float(args.pred_use_gt_center_iou),
            tightfit_enable=bool(args.tightfit),
            tightfit_max_nfev=int(args.tightfit_max_nfev),
            tightfit_k=float(args.tightfit_k), vehclf=classifier,
            comp_models_by_type=comp_by_type, default_comp_model=default_comp,
            vehclf_crop_pad=float(args.vehclf_crop_pad),
            vehclf_min_prob=float(args.vehclf_min_prob), pred_meta_rows=meta_rows,
            force_veh_type=force_type, proj_center_source=str(args.proj_center_source),
            proj_use_gt_center_iou=float(args.proj_use_gt_center_iou),
            proj_refine_iters=int(args.proj_refine_iters),
            fixed_ground_y=float(args.fixed_ground_y),
        )
        if index % 50 == 0 or index == len(image_ids):
            print(f"[OK] {index}/{len(image_ids)} done")

    if meta_rows is not None:
        meta_path = write_prediction_metadata(meta_rows, args.pred_meta_csv)
        if meta_path is not None:
            print(f"[DONE] pred meta written to: {meta_path}")
    print(f"[DONE] predictions written to: {output_data_dir}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run(args)
