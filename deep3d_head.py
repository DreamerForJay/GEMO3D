# deep3d_head.py
import numpy as np
import torch
from torch import amp
from torchvision.models import vgg
import math
from torch_lib import Model, ClassAverages
from torch_lib.Dataset import generate_bins, DetectedObject  # 提供 theta_ray 與 224x224 crop

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_USE_AMP = (_DEVICE == "cuda")
_BINS = 2

_model = None
_avg = None
_bins_edges = None

_VALID = {"car":"Car","van":"Car","truck":"Car","bus":"Car",
          "person":"Pedestrian","pedestrian":"Pedestrian",
          "bicycle":"Cyclist","cyclist":"Cyclist","motorcycle":"Cyclist"}

def _norm_cls(c: str) -> str:
    if not isinstance(c, str): return "Car"
    return _VALID.get(c.lower(), "Car")

def _wrap_rad(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _to_tensor_224(crop) -> torch.Tensor:
    """
    將 crop 轉成 3x224x224 浮點張量，值域 0~1。
    支援 torch.Tensor、np.ndarray、PIL.Image。
    """
    if isinstance(crop, torch.Tensor):
        t = crop
    elif isinstance(crop, np.ndarray):
        t = torch.from_numpy(crop)
    else:
        # 嘗試將 PIL 或其他影像類型轉成 ndarray
        try:
            import numpy as _np
            t = torch.from_numpy(_np.array(crop))
        except Exception as e:
            raise TypeError(f"unsupported crop type: {type(crop)}") from e

    # HWC -> CHW
    if t.ndim == 3 and t.shape[-1] == 3:
        t = t.permute(2, 0, 1)
    if t.ndim != 3 or t.shape[0] != 3:
        raise ValueError(f"unexpected crop shape: {tuple(t.shape)} (expect 3xHxW)")

    # dtype 與值域
    if t.dtype in (torch.uint8, torch.int16, torch.int32, torch.int64):
        t = t.float().div_(255.0)
    elif t.dtype not in (torch.float16, torch.float32, torch.float64):
        t = t.float()

    # resize 至 224x224
    if t.shape[-2:] != (224, 224):
        t = torch.nn.functional.interpolate(
            t.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
        ).squeeze(0)

    return t


def load_head(weights_path: str = "weights/epoch_50_converted.pth") -> None:
    """
    載入 Deep3DBox 輕量頭。需先安裝 torch/torchvision，並確保 torch_lib 可被匯入。
    """
    global _model, _avg, _bins_edges
    if _model is not None:
        return

    torch.backends.cudnn.benchmark = True

    # Backbone
    back = vgg.vgg19_bn(weights=vgg.VGG19_BN_Weights.DEFAULT)
    _model = Model.Model(features=back.features, bins=_BINS).to(_DEVICE).eval()
    if _USE_AMP:
        _model.half()

    # 權重
    ckpt = torch.load(weights_path, map_location=_DEVICE)
    _model.load_state_dict(ckpt["model_state_dict"], strict=True)

    # 統計與角度分桶
    _avg = ClassAverages.ClassAverages()
    _bins_edges = generate_bins(_BINS)

    # 暖機
    dummy = torch.zeros(1, 3, 224, 224, device=_DEVICE)
    dummy = dummy.half() if _USE_AMP else dummy.float()
    with torch.no_grad():
        if _USE_AMP:
            with amp.autocast("cuda"):
                _model(dummy)
        else:
            _model(dummy)


def infer_batch(image_bgr, dets_xyxy_cls, P_or_calib):
    """
    以批次方式進行 Deep3DBox 推論（尺寸與朝向）。
    dets_xyxy_cls: [(x1,y1,x2,y2,cls,conf), ...]
    P_or_calib: 3x4 相機投影矩陣，或 KITTI calib 檔路徑
    return: [{'alpha':..., 'theta_ray':..., 'yaw':..., 'dims':(L,W,H), 'box_2d':..., 'cls':...}, ...]
    """
    assert _model is not None, "call load_head() before infer_batch()"

    crops, rays, boxes, clss = [], [], [], []

    for (x1, y1, x2, y2, cls, _conf) in dets_xyxy_cls:
        cls_kitti = _norm_cls(cls)
        box_2d = ((int(x1), int(y1)), (int(x2), int(y2)))
        # DetectedObject 內部會計算 theta_ray 與裁切影像
        dobj = DetectedObject(image_bgr, cls_kitti, box_2d, P_or_calib)
        crops.append(dobj.img)
        rays.append(dobj.theta_ray)
        boxes.append(box_2d)
        clss.append(cls)

    if not crops:
        return []

    # 準備批次輸入
    ts = [_to_tensor_224(c) for c in crops]
    inp = torch.stack(ts, dim=0).to(_DEVICE)
    inp = inp.half() if _USE_AMP else inp.float()

    # 前傳
    with torch.no_grad():
        if _USE_AMP:
            with amp.autocast("cuda"):
                orient, conf, dim = _model(inp)
        else:
            orient, conf, dim = _model(inp)

    orient = orient.float().cpu().numpy()
    conf = conf.float().cpu().numpy()
    dim = dim.float().cpu().numpy()

    # 還原實際尺寸 + 計算 global yaw
    out = []
    for i in range(len(crops)):
        d = dim[i] + _avg.get_item(clss[i])  # 復原至實際 (L,W,H)
        k = int(np.argmax(conf[i]))
        cos, sin = orient[i, k, 0], orient[i, k, 1]
        alpha = np.arctan2(sin, cos) + _bins_edges[k]  # local orientation (bin-based)
        alpha = _wrap_rad(float(alpha))

        # === FIX: global 180° systematic flip ===
        # If all samples are off by ±180° vs KITTI, alpha is globally shifted by π.
        # Apply a deterministic π offset so that alpha aligns with KITTI's observation angle.
        alpha = _wrap_rad(alpha + math.pi)

        yaw = _wrap_rad(alpha + float(rays[i]))        # global yaw (KITTI-style: ry = alpha + theta_ray)

        l,w,h = float(d[2]), float(d[1]), float(d[0])
        out.append(
            {
                "alpha": float(alpha),
                "theta_ray": float(rays[i]),
                "yaw": float(yaw),
                "dims": (l,w,h),
                "box_2d": boxes[i],
                "cls": clss[i],
            }
        )

    return out
