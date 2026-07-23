# vehicletype_head.py
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------- common utils ----------------

def _safe_uint8_bgr(img: np.ndarray) -> np.ndarray:
    if img is None or img.size == 0:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def crop_bgr(image_bgr: np.ndarray, x1: float, y1: float, x2: float, y2: float, pad_ratio: float = 0.0) -> Optional[np.ndarray]:
    """
    Crop bbox from image (BGR).

    pad_ratio: expand bbox by this fraction on each side (e.g. 0.15 adds 15% of box w/h).
    """
    H, W = image_bgr.shape[:2]
    x1f, y1f, x2f, y2f = float(x1), float(y1), float(x2), float(y2)

    pr = float(pad_ratio)
    if pr > 0.0:
        bw = max(1.0, x2f - x1f)
        bh = max(1.0, y2f - y1f)
        pw = bw * pr
        ph = bh * pr
        x1f -= pw; x2f += pw
        y1f -= ph; y2f += ph

    ix1 = int(max(0, min(W - 1, math.floor(x1f))))
    iy1 = int(max(0, min(H - 1, math.floor(y1f))))
    ix2 = int(max(0, min(W,     math.ceil (x2f))))
    iy2 = int(max(0, min(H,     math.ceil (y2f))))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return image_bgr[iy1:iy2, ix1:ix2].copy()


def _letterbox_resize_bgr(im: np.ndarray, out_hw: Tuple[int, int] = (224, 224), pad_val: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    Resize with unchanged aspect ratio using padding (letterbox).
    out_hw: (H,W)
    """
    import cv2
    oh, ow = int(out_hw[0]), int(out_hw[1])
    h, w = im.shape[:2]
    if h <= 0 or w <= 0:
        return np.full((oh, ow, 3), pad_val, dtype=np.uint8)

    r = min(ow / w, oh / h)
    nw = max(1, int(round(w * r)))
    nh = max(1, int(round(h * r)))

    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((oh, ow, 3), pad_val, dtype=resized.dtype)

    dw = (ow - nw) // 2
    dh = (oh - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    return canvas

# ---------------- unified output schema ----------------

@dataclass
class VehTypeOut:
    # "label" 是你最後想用的 coarse label（可為 6cls 或 3cls），由 backend 決定
    label: str
    prob: float
    # raw backend label（如果有）
    raw_label: str = ""
    raw_prob: float = 0.0
    # 可附帶完整機率向量/原始 dict
    extra: Optional[Dict[str, Any]] = None

class VehicleTypeClassifierBase:
    def predict_crops(self, crops_bgr: List[Optional[np.ndarray]]) -> List[VehTypeOut]:
        raise NotImplementedError

# =========================================================
# 1) VehicleTypeNet via ONNXRuntime (6-way)
# =========================================================

class VehicleTypeNetORT(VehicleTypeClassifierBase):
    """
    For resnet18_pruned.onnx (N,3,224,224) -> (N,6) softmax/logits.
    Default labels follow TAO VehicleTypeNet-like: coupe/sedan/suv/van/large_vehicle/truck
    """
    def __init__(
        self,
        onnx_path: str,
        labels_6: Tuple[str, ...] = ("coupe", "sedan", "suv", "van", "large_vehicle", "truck"),
        rgb: bool = True,
        do_norm: bool = True,
        providers: Tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider"),
        letterbox: bool = True,
        letterbox_pad: Tuple[int,int,int] = (0,0,0),
    ):
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as e:
            raise RuntimeError("onnxruntime not available. Install onnxruntime-gpu.") from e

        p = Path(onnx_path)
        if not p.is_file():
            raise FileNotFoundError(f"VehicleTypeNet ORT ONNX not found: {onnx_path}")

        self.labels_6 = list(labels_6)
        self.rgb = bool(rgb)
        self.do_norm = bool(do_norm)
        self.ort = ort
        self.sess = ort.InferenceSession(str(p), providers=list(providers))
        self.inp_name = self.sess.get_inputs()[0].name
        self.letterbox = bool(letterbox)
        self.letterbox_pad = tuple(int(x) for x in letterbox_pad)

    def _preprocess_one(self, crop_bgr: np.ndarray) -> np.ndarray:
        import cv2
        if self.letterbox:
            im = _letterbox_resize_bgr(crop_bgr, out_hw=(224,224), pad_val=self.letterbox_pad)
        else:
            im = cv2.resize(crop_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
        if self.rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(np.float32) / 255.0
        if self.do_norm:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            im = (im - mean) / std
        im = np.transpose(im, (2, 0, 1))  # CHW
        return im[None, :, :, :]  # NCHW

    def predict_crops(self, crops_bgr: List[Optional[np.ndarray]]) -> List[VehTypeOut]:
        if not crops_bgr:
            return []
        xs = []
        keep = []
        out = [VehTypeOut(label="unknown", prob=0.0) for _ in crops_bgr]
        for i, c in enumerate(crops_bgr):
            if c is None or c.size == 0:
                continue
            try:
                xs.append(self._preprocess_one(_safe_uint8_bgr(c)))
                keep.append(i)
            except Exception:
                continue
        if not xs:
            return out

        x = np.concatenate(xs, axis=0).astype(np.float32)  # (N,3,224,224)
        y = self.sess.run(None, {self.inp_name: x})[0]     # (N,6)
        for bi, idx in enumerate(keep):
            probs = y[bi].astype(np.float32)
            k = int(np.argmax(probs))
            lab = self.labels_6[k] if 0 <= k < len(self.labels_6) else str(k)
            out[idx] = VehTypeOut(
                label=lab,
                prob=float(probs[k]),
                raw_label=lab,
                raw_prob=float(probs[k]),
                extra={"probs6": probs},
            )
        return out

# =========================================================
# 2) PaddleClas vehicle_attribute (map to 3cls)
# =========================================================

_TYPE_RE = re.compile(r"Type:\s*\((\w+),\s*prob:\s*([0-9.]+)\)", re.IGNORECASE)
KNOWN_TYPES = {"hatchback", "sedan", "suv", "van", "mpv", "pickup", "bus", "truck", "estate"}

class PaddleClasVehicleAttr(VehicleTypeClassifierBase):
    """
    PaddleClas PULC model 'vehicle_attribute' wrapper.
    Output label is 5cls: {'compact','sedan','suv','van','unknown'}.
    """
    def __init__(
        self,
        use_gpu: bool = True,
        batch_size: int = 8,
        compact_alias=("hatchback",),
        van_alias=("van", "mpv"),
        min_prob: float = 0.0,
    ):
        try:
            import paddleclas  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "paddleclas not available. Install paddlepaddle(+gpu) + paddleclas in this env."
            ) from e

        # ---- critical: create the PaddleClas model ----
        self.model = paddleclas.PaddleClas(
            model_name="vehicle_attribute",
            use_gpu=bool(use_gpu),
            batch_size=int(batch_size),
        )

        # allow alias args to be either tuple/list or comma-separated string
        def _norm_alias(a):
            if a is None:
                return tuple()
            if isinstance(a, str):
                return tuple([s.strip().lower() for s in a.split(",") if s.strip()])
            # sequence
            return tuple([str(s).strip().lower() for s in a if str(s).strip()])

        self.compact_alias = set(_norm_alias(compact_alias))
        self.van_alias = set(_norm_alias(van_alias))
        self.min_prob = float(min_prob)

    def _parse_type(self, pred_item: Dict[str, Any]) -> Tuple[str, float]:
        attr = pred_item.get("attributes") or pred_item.get("attr")
        if isinstance(attr, str):
            m = _TYPE_RE.search(attr)
            if m:
                return m.group(1).lower(), float(m.group(2))

        names = pred_item.get("label_names")
        if isinstance(names, list):
            joined = " ".join([str(x) for x in names])
            m = _TYPE_RE.search(joined)
            if m:
                return m.group(1).lower(), float(m.group(2))
            for t in KNOWN_TYPES:
                if t in joined.lower():
                    scores = pred_item.get("scores")
                    p = float(scores[0]) if isinstance(scores, list) and scores else 0.0
                    return t, p

        return "unknown", 0.0

    def _map_5cls(self, t: str, p: float) -> str:
        if p < self.min_prob:
            return "unknown"
        t = (t or "unknown").lower()

        if t == "hatchback":
            return "compact"
        # FIX: this must be a tuple membership check; the old `or "estate"` is always-true.
        if t in ("sedan", "estate"):
            return "sedan"
        if t == "suv":
            return "suv"
        if (t in ("van", "mpv")) or (t in self.van_alias):
            return "van"
        return "unknown"

    def predict_crops(self, crops_bgr: List[Optional[np.ndarray]]) -> List[VehTypeOut]:
        if not crops_bgr:
            return []

        out: List[VehTypeOut] = []
        for c in crops_bgr:
            if c is None or not isinstance(c, np.ndarray) or c.size == 0:
                out.append(VehTypeOut(label="unknown", prob=0.0, raw_label="unknown", raw_prob=0.0))
                continue

            img = _safe_uint8_bgr(c)

            # 保底：確保 3ch
            if img.ndim == 2:
                img = np.repeat(img[:, :, None], 3, axis=2)
            elif img.shape[2] == 4:
                img = img[:, :, :3]

            # PaddleClas wheel: ndarray 需要 RGB (不是 BGR)
            img_rgb = img[:, :, ::-1].copy()

            try:
                gen = self.model.predict(input_data=img_rgb)   # 單張 ndarray
                batch_out = next(gen)                          # list[dict], 通常長度 1
                item = batch_out[0] if batch_out else {}
                t, p = self._parse_type(item)
                t5 = self._map_5cls(t, p)
                out.append(VehTypeOut(label=t5, prob=float(p), raw_label=t, raw_prob=float(p), extra={"raw": item}))
            except Exception as e:
                out.append(VehTypeOut(label="unknown", prob=0.0, raw_label="unknown", raw_prob=0.0, extra={"err": str(e)}))

        return out

# =========================================================
# 3) Ultralytics YOLO classification backend (crop -> type)
# =========================================================

class YOLOClsVehicleTypeClassifier(VehicleTypeClassifierBase):
    """
    Ultralytics YOLO classification wrapper for vehicle-type crops.

    This backend is intentionally independent from the main YOLO detector used by
    nearest_time_topbev_xp.py. The main detector still provides 2D boxes; this
    classifier only receives already-cropped vehicle images and returns the
    unified 5-class schema used downstream: compact/sedan/suv/van/unknown.
    """

    DEFAULT_LABEL_MAP = {
        "hatchback": "compact",
        "pickup": "compact",
        "sedan": "sedan",
        "suv": "suv",
        "van": "van",
        "compact": "compact",
    }
    VALID_OUT = {"compact", "sedan", "suv", "van"}

    def __init__(
        self,
        weights: str,
        imgsz: int = 224,
        device: str = "",
        half: bool = False,
        label_map: str = "hatchback=compact,pickup=compact,sedan=sedan,suv=suv,van=van",
        verbose: bool = False,
    ):
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            raise RuntimeError("ultralytics not available. Install with: pip install ultralytics") from e

        p = Path(str(weights)).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"YOLO classification weights not found: {weights}")

        self.model = YOLO(str(p))
        self.imgsz = int(imgsz)
        self.device = str(device or "").strip()
        self.half = bool(half)
        self.verbose = bool(verbose)
        self.label_map = self._parse_label_map(label_map)

    @classmethod
    def _parse_label_map(cls, text: str) -> Dict[str, str]:
        m = dict(cls.DEFAULT_LABEL_MAP)
        text = str(text or "").strip()
        if not text:
            return m
        for item in text.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" in item:
                k, v = item.split("=", 1)
            elif ":" in item:
                k, v = item.split(":", 1)
            else:
                continue
            k = k.strip().lower()
            v = v.strip().lower()
            if k and v:
                m[k] = v
        return m

    def _coarse_label(self, raw_label: str) -> str:
        raw = str(raw_label or "unknown").strip().lower()
        mapped = str(self.label_map.get(raw, raw)).strip().lower()
        return mapped if mapped in self.VALID_OUT else "unknown"

    def predict_crops(self, crops_bgr: List[Optional[np.ndarray]]) -> List[VehTypeOut]:
        if not crops_bgr:
            return []

        out: List[VehTypeOut] = [
            VehTypeOut(label="unknown", prob=0.0, raw_label="unknown", raw_prob=0.0)
            for _ in crops_bgr
        ]
        valid_imgs: List[np.ndarray] = []
        valid_idx: List[int] = []

        for i, c in enumerate(crops_bgr):
            if c is None or not isinstance(c, np.ndarray) or c.size == 0:
                continue
            img = _safe_uint8_bgr(c)
            if img.ndim == 2:
                img = np.repeat(img[:, :, None], 3, axis=2)
            elif img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            if img.ndim != 3 or img.shape[2] != 3:
                continue
            valid_imgs.append(img)
            valid_idx.append(i)

        if not valid_imgs:
            return out

        try:
            kwargs: Dict[str, Any] = {
                "source": valid_imgs,
                "imgsz": self.imgsz,
                "verbose": self.verbose,
            }
            if self.device:
                kwargs["device"] = self.device
            if self.half:
                kwargs["half"] = True
            results = self.model.predict(**kwargs)
            names = getattr(self.model, "names", {}) or {}

            for local_i, r in enumerate(results):
                idx = valid_idx[local_i]
                probs = getattr(r, "probs", None)
                if probs is None:
                    continue
                top1 = int(getattr(probs, "top1", -1))
                conf_obj = getattr(probs, "top1conf", 0.0)
                conf = float(conf_obj.item()) if hasattr(conf_obj, "item") else float(conf_obj)
                if isinstance(names, dict):
                    raw = str(names.get(top1, str(top1)))
                elif isinstance(names, (list, tuple)) and 0 <= top1 < len(names):
                    raw = str(names[top1])
                else:
                    raw = str(top1)
                lab = self._coarse_label(raw)
                out[idx] = VehTypeOut(
                    label=lab,
                    prob=conf,
                    raw_label=raw.strip().lower(),
                    raw_prob=conf,
                    extra={"top1": top1},
                )
        except Exception as e:
            for idx in valid_idx:
                out[idx] = VehTypeOut(
                    label="unknown",
                    prob=0.0,
                    raw_label="unknown",
                    raw_prob=0.0,
                    extra={"err": str(e)},
                )

        return out

# =========================================================
# factory
# =========================================================

def build_vehicle_classifier(
    backend: str,
    *,
    ort_onnx: str = "",
    ort_labels6: str = "coupe,sedan,suv,van,large_vehicle,truck",
    ort_bgr: bool = False,         # default: RGB (so bgr=False)
    ort_no_norm: bool = False,
    paddle_use_gpu: bool = True,
    paddle_batch: int = 8,
    paddle_min_prob: float = 0.3,
    paddle_compact_alias: str = "hatchback",
    paddle_van_alias: str = "van,mpv",   # NEW
    ort_letterbox: bool = True,
    yolo_weights: str = "",
    yolo_imgsz: int = 224,
    yolo_device: str = "",
    yolo_half: bool = False,
    yolo_label_map: str = "hatchback=compact,pickup=compact,sedan=sedan,suv=suv,van=van",
    
) -> Optional[VehicleTypeClassifierBase]:
    b = (backend or "none").lower()
    if b in ("none", "off", "disable", "disabled"):
        return None

    if b in ("ort", "vehicletypenet", "vtnet"):
        labels6 = tuple([s.strip() for s in ort_labels6.split(",") if s.strip()])
        return VehicleTypeNetORT(
            onnx_path=ort_onnx,
            labels_6=labels6 if labels6 else ("coupe","sedan","suv","van","large_vehicle","truck"),
            rgb=(not ort_bgr),
            do_norm=(not ort_no_norm),
            letterbox=bool(ort_letterbox),
        )

    if b in ("paddle", "paddleclas", "vehicle_attribute", "vehattr"):
        compact_alias = tuple([s.strip().lower() for s in paddle_compact_alias.split(",") if s.strip()])
        van_alias = tuple([s.strip().lower() for s in paddle_van_alias.split(",") if s.strip()])
        return PaddleClasVehicleAttr(
            use_gpu=bool(paddle_use_gpu),
            batch_size=int(paddle_batch),
            compact_alias=compact_alias if compact_alias else ("hatchback",),
            van_alias=van_alias if van_alias else ("van","mpv"),
            min_prob=float(paddle_min_prob),
        )

    if b in ("yolo", "yolo_cls", "ultralytics", "ultralytics_cls"):
        return YOLOClsVehicleTypeClassifier(
            weights=yolo_weights,
            imgsz=int(yolo_imgsz),
            device=str(yolo_device),
            half=bool(yolo_half),
            label_map=str(yolo_label_map),
        )

    raise ValueError(f"unknown vehclf backend: {backend}")
