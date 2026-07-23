#!/usr/bin/env python3
"""Dependency-light smoke tests for the refactored pipeline."""
from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from gemo3d_ncomp.compensation import predict_compensation
from gemo3d_ncomp.config import K_from_P2, backproject_uv_depth
from gemo3d_ncomp.geometry import box3d_corners_cam, project_points
from gemo3d_ncomp.pipeline import infer_one_image
from gemo3d_ncomp.visualization import bev_iou_rotated_rects, iou3d_kitti_boxes


P2 = np.array(
    [[721.5, 0.0, 609.5, 0.0], [0.0, 721.5, 172.8, 0.0], [0.0, 0.0, 1.0, 0.0]],
    dtype=np.float64,
)
K = K_from_P2(P2)


class _Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _XYXY:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, _index):
        return self

    def tolist(self):
        return self.values


class _FakeBox:
    cls = _Scalar(0)
    conf = _Scalar(0.9)
    xyxy = _XYXY([500.0, 120.0, 700.0, 300.0])


class _FakeYOLO:
    names = {0: "car"}

    def __init__(self, with_detection: bool):
        self.with_detection = with_detection

    def predict(self, **_kwargs):
        boxes = [_FakeBox()] if self.with_detection else []
        return [SimpleNamespace(boxes=boxes)]


def test_geometry() -> None:
    x, _, _ = backproject_uv_depth(K["cx"], 300.0, 20.0, P2)
    assert abs(x) < 1e-9

    corners = box3d_corners_cam(1.5, 1.7, 4.0, 0.0, 1.65, 20.0, 0.0)
    uv = project_points(P2, corners)
    assert uv.shape == (8, 2)
    assert np.isfinite(uv).all()

    bev = {"x": 0.0, "z": 20.0, "ry": 0.0, "L": 4.0, "W": 1.7}
    assert abs(bev_iou_rotated_rects(bev, bev)[0] - 1.0) < 1e-6

    box = {"x": 0.0, "y": 1.65, "z": 20.0, "ry": 0.0, "h": 1.5, "w": 1.7, "l": 4.0}
    assert abs(iou3d_kitti_boxes(box, box) - 1.0) < 1e-6


def test_rank_ratio_compensation() -> None:
    model = {
        "model_type": "rank_ratio_linear_depth_svd",
        "angle_grid": np.array([0.0, 180.0]),
        "depth_grid": np.array([5.0, 40.0]),
        "R_reconstructed_grid": np.full((2, 2), 1.1),
    }
    comp = predict_compensation(90.0, model, z_raw=20.0)
    assert abs(comp + 2.0) < 1e-6


def run_fake_inference(with_detection: bool) -> tuple[str, list[dict]]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        image_path = root / "000001.png"
        cv2.imwrite(str(image_path), np.zeros((375, 1242, 3), dtype=np.uint8))
        output_dir = root / "pred" / "data"
        rows: list[dict] = []
        infer_one_image(
            img_path=image_path,
            calib_path=root / "000001.txt",
            yolo_model=_FakeYOLO(with_detection),
            car_names=["car"],
            conf_thres=0.5,
            imgsz=640,
            yolo_device="cpu",
            use_half=False,
            egonet_enable=False,
            deep3d_enable=False,
            P2=P2,
            K=K,
            height_m=1.55,
            dims_source="fixed",
            fixed_dims_hwl=(1.55, 1.74, 3.86),
            comp_model=None,
            out_pred_data_dir=output_dir,
            comp_models_by_type={},
            default_comp_model=None,
            pred_meta_rows=rows,
        )
        return (output_dir / "000001.txt").read_text(encoding="utf-8"), rows


def main() -> None:
    test_geometry()
    test_rank_ratio_compensation()

    empty_text, empty_rows = run_fake_inference(False)
    assert empty_text == ""
    assert empty_rows == []

    prediction, rows = run_fake_inference(True)
    fields = prediction.strip().split()
    assert fields[0] == "Car"
    assert len(fields) == 16
    assert len(rows) == 1
    print("SMOKE_TEST_OK")


if __name__ == "__main__":
    main()
