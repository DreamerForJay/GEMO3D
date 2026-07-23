# GEMO3D 完整安裝與使用教學

---

## 目錄

1. [GEMO3D 是什麼](#1-gemo3d-是什麼)
2. [整體推論流程](#2-整體推論流程)
3. [硬體與軟體需求](#3-硬體與軟體需求)
4. [下載專案](#4-下載專案)
5. [建立 Python 環境](#5-建立-python-環境)
6. [確認 PyTorch 與 CUDA](#6-確認-pytorch-與-cuda)
7. [準備 YOLO 權重](#7-準備-yolo-權重)
8. [準備 EgoNet](#8-準備-egonet)
9. [準備 Deep3DBox（可選）](#9-準備-deep3dbox可選)
10. [準備車型分類器（可選）](#10-準備車型分類器可選)
11. [準備深度補償模型](#11-準備深度補償模型)
12. [準備 KITTI 格式資料集](#12-準備-kitti-格式資料集)
13. [第一次執行前的檢查](#13-第一次執行前的檢查)
14. [最小流程測試](#14-最小流程測試)
15. [完整 GEMO3D 推論](#15-完整-gemo3d-推論)
16. [多車型推論](#16-多車型推論)
17. [主要參數說明](#17-主要參數說明)
18. [輸出檔案說明](#18-輸出檔案說明)
19. [如何檢查推論是否成功](#19-如何檢查推論是否成功)
20. [常見錯誤與排除方式](#20-常見錯誤與排除方式)
21. [建立小型測試 split](#21-建立小型測試-split)
22. [資源不足時的安全設定](#22-資源不足時的安全設定)
23. [正式實驗的建議設定](#23-正式實驗的建議設定)
24. [環境備份與移植](#24-環境備份與移植)
25. [專案結構](#25-專案結構)

---

## 1. GEMO3D 是什麼

GEMO3D 是一套以單張 RGB 影像進行車輛 3D 定位的模組化推論流程。它不以單一端到端網路直接回歸完整 3D 邊界框，而是將問題拆成數個可解釋步驟：

1. 使用 YOLO 偵測車輛的 2D 邊界框。
2. 使用 EgoNet 或 Deep3DBox 估計車輛朝向。
3. 使用固定尺寸或車型尺寸先驗取得 `(h, w, l)`。
4. 根據針孔相機模型，以 2D 邊界框高度估計初始深度。
5. 使用投影高度比例模型修正深度。
6. 使用道路平面約束修正 3D bottom center。
7. 使用 3D 邊界框投影一致性進行 tight-fit 最佳化。
8. 輸出 KITTI label 格式的 3D 偵測結果。

GEMO3D 採用 KITTI camera coordinate system：

- `X`：影像右方。
- `Y`：影像下方。
- `Z`：相機前方。
- 3D 位置 `(x, y, z)`：3D 邊界框的底部中心。
- 尺寸順序：`(h, w, l)`。
- `ry`：繞相機 Y 軸旋轉的角度，單位為弧度。



## 2. 整體推論流程

```text
輸入 RGB 影像
    │
    ▼
YOLO 2D 車輛偵測
    │
    ├──────────────► 車型分類器（可選）
    │                     │
    │                     ▼
    │              車型尺寸與補償模型
    │
    ▼
EgoNet／Deep3DBox 朝向估計
    │
    ▼
針孔模型初始深度
z_est = fy × H / bbox_height
    │
    ▼
投影高度比例補償
    │
    ▼
像素反投影與道路平面約束
    │
    ▼
pred_reproj feedback（可選）
    │
    ▼
tight-fit 投影一致性最佳化（可選）
    │
    ▼
KITTI 3D prediction
```

---

## 3. 硬體與軟體需求

### 3.1 最低需求

- Linux、Windows 或其他可執行 Python 的系統。
- Python 3.10 或 3.11。
- 至少 8 GB RAM。
- 足夠的磁碟空間存放：
  - KITTI／CARLA 資料。
  - YOLO 權重。
  - EgoNet 權重。
  - 補償模型。
  - 推論輸出。

### 3.2 建議需求

- Ubuntu 20.04、22.04 或相近版本。
- NVIDIA GPU。
- 8 GB 以上 GPU VRAM。
- 16 GB 以上系統記憶體。
- Conda 或 Miniconda。


## 4. 下載專案

```bash
cd ~/Desktop
git clone https://github.com/twqs111/GEMO3D.git
cd GEMO3D
```

---

## 5. 建立 Python 環境

以下以 Conda 為例。

### 5.1 建立環境

```bash
conda create -n gemo3d python=3.10 -y
conda activate gemo3d
```

更新基本安裝工具：

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 5.2 先安裝 PyTorch

PyTorch 必須先依照主機的 NVIDIA driver 與 CUDA 相容情況安裝。請使用 PyTorch 官方安裝選擇器產生適合本機的指令：

- <https://pytorch.org/get-started/locally/>

例如，使用 CUDA 12.1 的安裝方式可能為：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 5.3 安裝 GEMO3D 基本套件

```bash
pip install -r requirements.txt
```

若需要讀寫或分析 CSV，可另外安裝：

```bash
pip install pandas openpyxl
```

### 5.4 無圖形介面的伺服器

若在沒有桌面環境的伺服器遇到：

```text
ImportError: libGL.so.1
```

可以改用 headless OpenCV：

```bash
pip uninstall -y opencv-python
pip install opencv-python-headless
```

---

## 6. 確認 PyTorch 與 CUDA

執行：

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

GPU 環境預期：

```text
CUDA available: True
GPU: NVIDIA ...
```

再確認 Ultralytics：

```bash
python - <<'PY'
from ultralytics import YOLO
print("Ultralytics import OK")
PY
```

注意：即使使用 `--no_egonet`，目前 `egonet_ori.py` 仍會在模組匯入階段載入 `torch`，因此 PyTorch 仍是必要套件。

---

## 7. 準備 YOLO 權重

### 7.1 快速測試


第一次執行時，Ultralytics 通常會直接嘗試下載公開權重。若主機無法連線，請先在可連網主機下載，再複製到專案，例如：

```text
GEMO3D/
└── weights/
    └── yolo11n.pt
```

執行時可以參數指定權重路徑：

```bash
--yolo_weights weights/yolo11n.pt
```

### 7.2 正式實驗

若有足夠 GPU 記憶體，可用較大的權重，例如：

```text
yolo11l.pt
```

目前程式預設值為 `yolo11l.pt`。

---

## 8. 準備 EgoNet

目前儲存庫已包含：

```text
EgoNet-master/
```

因此不需要再次 clone 官方 EgoNet 原始碼，但仍需要準備其預訓練權重。

### 8.1 下載預訓練權重

依 `EgoNet-master/docs/preparation.md` 的預訓練模型連結下載並解壓縮。建議放置為：

```text
GEMO3D/
└── weights/
    └── egonet/
        ├── HC.pth
        ├── L.pth
        ├── LS.npy
        └── 其他下載內容
```

實際檔案名稱應以官方壓縮檔內容為準。

### 8.2 設定環境變數

在 GEMO3D 根目錄執行：

```bash
export EGONET_ROOT="$PWD/EgoNet-master"
export EGONET_CFG="$PWD/EgoNet-master/configs/KITTI_inference:test_submission.yml"
```

確認：

```bash
echo "$EGONET_ROOT"
echo "$EGONET_CFG"
test -f "$EGONET_ROOT/libs/model/egonet.py" && echo "EgoNet source OK"
test -f "$EGONET_CFG" && echo "EgoNet config OK"
```



### 8.3 每次開啟終端機都自動設定

可建立：

```bash
mkdir -p scripts
cat > scripts/env.sh <<'EOF'
export EGONET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/EgoNet-master"
export EGONET_CFG="$EGONET_ROOT/configs/KITTI_inference:test_submission.yml"
EOF
```

使用：

```bash
source scripts/env.sh
```

### 8.4 使用 EgoNet 輸出尺寸

預設建議只使用 EgoNet 朝向，尺寸使用固定值或車型先驗。

若確實要使用 EgoNet 估計尺寸：

```bash
export EGONET_EXPORT_DIMS=1
export EGONET_DIMS_FALLBACK_HWL="1.52,1.64,3.86"
```

1.52,1.64,3.86為fallback值，可自行設定平均車輛尺寸。

並在命令中使用：

```bash
--dims_source egonet
```

關閉尺寸輸出：

```bash
unset EGONET_EXPORT_DIMS
unset EGONET_DIMS_FALLBACK_HWL
```

---

## 9. 準備 Deep3DBox（可選）

Deep3DBox 不是目前GEMO3D項目。只有在命令改用Deep3DBox預測：

```bash
--dims_source deep3d
```

時才需要完整設定。

所需內容：

1. `deep3d_head.py`。
2. 可匯入的 `torch_lib`。
3. Deep3DBox checkpoint，例如：

```text
weights/epoch_50_converted.pth
```

確認：

```bash
python - <<'PY'
import torch_lib
print("torch_lib import OK")
PY
```

執行參數：

```bash
--dims_source deep3d \
--deep3d_weights weights/epoch_50_converted.pth
```

注意事項：

- `deep3d_head.py` 使用 VGG19-BN backbone。
- 第一次載入可能需要下載 torchvision 的 VGG19-BN 預訓練權重。
- 離線主機應事先準備 torchvision cache。
- 目前 pipeline 會把 Deep3DBox 的尺寸轉換為 KITTI 的 `(h, w, l)`。
- 使用 `--dims_source deep3d` 時，初始深度的高度仍由 `--height_m` 提供；Deep3DBox 尺寸主要用於最終 3D box 與投影修正。

---

## 10. 準備車型分類器（可選）

GEMO3D 支援四種設定：

| `--vehclf_backend` | 說明 |
|---|---|
| `none` | 不使用車型分類器 |
| `yolo_cls` | 使用 Ultralytics classification 權重 |
| `ort` | 使用 ONNXRuntime VehicleTypeNet |
| `paddleclas` | 使用 PaddleClas vehicle attribute |

### 10.1 YOLO classification

模型結構範例：

```text
weights/
└── vehicle_type_cls.pt
```

執行參數：

```bash
--vehclf_backend yolo_cls \
--vehclf_yolo_weights weights/vehicle_type_cls.pt \
--vehclf_yolo_imgsz 224 \
--vehclf_yolo_device cuda:0 \
--vehclf_yolo_half \
--vehclf_min_prob 0.20
```

預設 label map：

```text
hatchback=compact
pickup=compact //不太正確，但尚未訓練PICKUP補償模型
sedan=sedan
suv=suv
van=suv
```

建議明確改為：

```bash
--vehclf_yolo_label_map "hatchback=compact,pickup=compact,sedan=sedan,suv=suv,van=van"
```

### 10.2 ONNXRuntime

GPU：

```bash
pip install onnxruntime-gpu
```

CPU：

```bash
pip install onnxruntime
```

執行參數：

```bash
--vehclf_backend ort \
--vehclf_ort_onnx weights/resnet18_pruned.onnx
```

### 10.3 PaddleClas

依主機 CUDA 版本安裝 PaddlePaddle，再安裝：

```bash
pip install paddleclas
```

執行參數：

```bash
--vehclf_backend paddleclas \
--vehclf_paddle_use_gpu
```

---

## 11. 準備深度補償模型

補償模型通常是 `.pkl`。

建議結構：

```text
GEMO3D/
└── models/
    ├── compact_rank_ratio_linear_model.pkl
    ├── sedan_rank_ratio_linear_model.pkl
    ├── suv_rank_ratio_linear_model.pkl
    └── van_rank_ratio_linear_model.pkl
```

### 11.1 全域單一模型

```bash
--comp_pkl models/sedan_rank_ratio_linear_model.pkl
```

### 11.2 各車型模型

```bash
--vt_comp_pkl_map \
"compact=models/compact_rank_ratio_linear_model.pkl,\
sedan=models/sedan_rank_ratio_linear_model.pkl,\
suv=models/suv_rank_ratio_linear_model.pkl,\
van=models/van_rank_ratio_linear_model.pkl"
```

實際 shell 建議寫成同一行：

```bash
--vt_comp_pkl_map "compact=models/compact_rank_ratio_linear_model.pkl,sedan=models/sedan_rank_ratio_linear_model.pkl,suv=models/suv_rank_ratio_linear_model.pkl,van=models/van_rank_ratio_linear_model.pkl"
```

### 11.3 停用所有車型專屬模型

目前 `submodule/config.py` 仍保留預設絕對路徑。其他主機執行時可能看到：

```text
[vehclf] WARN: comp pkl not found ...
```

這通常不會中止程式，但會造成不必要的警告。

若不使用車型補償模型，請明確停用：

```bash
--vt_comp_pkl_map "compact=-,sedan=-,suv=-,van=-"
```

### 11.4 已知整組資料為單一車型

例如全部都是 sedan，強制轉型為統一車型：

```bash
--force_veh_type sedan \
--vt_comp_pkl_map "compact=-,sedan=models/sedan_rank_ratio_linear_model.pkl,suv=-,van=-"
```

支援：

```text
compact
sedan
suv
van
```

---

## 12. 準備 KITTI 格式資料集

### 12.1 必要結構

```text
KITTI_ROOT/
├── ImageSets/
│   └── val.txt
└── training/
    ├── image_2/
    │   ├── 000000.png
    │   ├── 000001.png
    │   └── ...
    ├── calib/
    │   ├── 000000.txt
    │   ├── 000001.txt
    │   └── ...
    ├── label_2/          # 正式純推論可省略；評估、GT 視覺化時需要
    ├── meta/             # CARLA 地面平面與外參，可選
    ├── bev_iou/          # 可選
    ├── fr_iou/           # 可選
    └── side_iou/         # 可選
```

### 12.2 split 格式

`ImageSets/val.txt`：

```text
000000
000001
000002
```

以下格式也可以：

```text
000000.png
000001.png
```

程式會自動取檔名 stem。

### 12.3 影像限制

目前程式會固定尋找：

```text
training/image_2/<id>.png
```

因此 JPG 影像需先轉成 PNG，而不是只改副檔名。

例如：

```bash
python - <<'PY'
from pathlib import Path
import cv2

src = Path("input.jpg")
dst = Path("KITTI_ROOT/training/image_2/000000.png")

img = cv2.imread(str(src))
if img is None:
    raise RuntimeError(f"Cannot read {src}")

dst.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(dst), img)
print(dst)
PY
```

### 12.4 calibration 檔案

每張影像都必須有同名 calibration：

```text
training/calib/000000.txt
```

至少需包含 `P2`：

```text
P2: 721.5377 0.0 609.5593 0.0 0.0 721.5377 172.8540 0.0 0.0 0.0 1.0 0.0
```

上述數值只是常見 KITTI 範例。正式推論必須使用該影像實際相機的 calibration，否則深度與 3D 位置不具物理意義。

### 12.5 label_2

純推論不需要 `label_2`。以下功能才會使用真值：

```text
--depth_use_gt_height
--pred_use_gt_hwl
--pred_center_source gt
--proj_center_source gt_bottom
--proj_center_source gt_geom
--vis_draw_gt
```

正式 monocular 3D 推論與公平比較時，不應啟用 GT 輔助參數。

### 12.6 CARLA meta

若存在：

```text
training/meta/<id>.json
```

且其中 `extrinsics` 能轉成 `4×4` world-to-sensor matrix，程式可使用道路平面估計 bottom-center Y。

若沒有 meta：

- 程式仍可執行。
- `tight-fit` 會退回一般的 `(x, y, z)` 最佳化。
- 也可使用 `--fixed_ground_y` 指定固定 bottom-center Y。

---

## 13. 第一次執行前的檢查

### 13.1 語法檢查

```bash
python -m compileall -q \
  gemo3d.py \
  submodule \
  egonet_ori.py \
  deep3d_head.py \
  vehicletype_head.py
```

### 13.2 主程式 help

```bash
python gemo3d.py --help
```

若這一步失敗，先不要執行正式資料集。

### 13.3 修正目前 smoke test 的舊套件名稱

目前 `tests/smoke_test.py` 仍引用 `gemo3d_ncomp`。可先備份並替換：

```bash
cp tests/smoke_test.py /tmp/smoke_test.py.bak
sed -i 's/gemo3d_ncomp/submodule/g' tests/smoke_test.py
python tests/smoke_test.py
```

成功時應看到：

```text
SMOKE_TEST_OK
```

若不希望保留修改：

```bash
git restore tests/smoke_test.py
```

### 13.4 檢查 EgoNet import

```bash
source scripts/env.sh 2>/dev/null || true

python - <<'PY'
import os
import sys

root = os.environ.get("EGONET_ROOT")
if not root:
    raise RuntimeError("EGONET_ROOT is not set")

sys.path.insert(0, root)
from libs.model.egonet import EgoNet

print("EgoNet import OK")
PY
```

---

## 14. 最小流程測試

這個版本的目的是確認：

- 能讀取 KITTI split。
- 能讀取影像與 calibration。
- YOLO 可以推論。
- 可以輸出 KITTI prediction。

它不使用 EgoNet、不使用補償模型，所有車輛 `ry` 會退回 `0`，因此不能視為完整 GEMO3D 精度。

### 14.1 GPU

```bash
KITTI_ROOT=/path/to/KITTI_ROOT
OUT=/path/to/outputs/gemo3d_baseline

python gemo3d.py \
  --kitti_root "$KITTI_ROOT" \
  --split val \
  --out_dir "$OUT" \
  --yolo_weights weights/yolo11n.pt \
  --no_egonet \
  --dims_source fixed \
  --fixed_dims_hwl 1.55,1.74,3.86 \
  --height_m 1.55 \
  --comp_pkl "" \
  --vt_comp_pkl_map "compact=-,sedan=-,suv=-,van=-" \
  --conf 0.25 \
  --imgsz 640 \
  --device cuda:0 \
  --half \
  --pred_meta_csv "$OUT/pred_meta.csv"
```

### 14.2 CPU

CPU 不要加 `--half`：

```bash
KITTI_ROOT=/path/to/KITTI_ROOT
OUT=/path/to/outputs/gemo3d_cpu

python gemo3d.py \
  --kitti_root "$KITTI_ROOT" \
  --split val \
  --out_dir "$OUT" \
  --yolo_weights weights/yolo11n.pt \
  --no_egonet \
  --dims_source fixed \
  --fixed_dims_hwl 1.55,1.74,3.86 \
  --height_m 1.55 \
  --comp_pkl "" \
  --vt_comp_pkl_map "compact=-,sedan=-,suv=-,van=-" \
  --conf 0.25 \
  --imgsz 640 \
  --device cpu \
  --pred_meta_csv "$OUT/pred_meta.csv"
```

---

## 15. 完整 GEMO3D 推論

以下以整組 sedan CARLA 資料為例。

### 15.1 設定路徑

```bash
cd ~/Desktop/GEMO3D
conda activate gemo3d

export EGONET_ROOT="$PWD/EgoNet-master"
export EGONET_CFG="$PWD/EgoNet-master/configs/KITTI_inference:test_submission.yml"

KITTI_ROOT=/home/e114/KITTI_ROOT/sed_fr_T1R8/data/KITTIDataset_Carla
OUT=/home/e114/eval_kitti/build/results/sed_fr_T1R8/SVD
EGONET_WEIGHTS="$PWD/weights/egonet"
COMP_MODEL="$PWD/models/sed_rank_ratio_linear_model.pkl"
```

### 15.2 執行

```bash
python gemo3d.py \
  --kitti_root "$KITTI_ROOT" \
  --split val \
  --out_dir "$OUT" \
  --yolo_weights weights/yolo11n.pt \
  --egonet_weights "$EGONET_WEIGHTS" \
  --dims_source fixed \
  --fixed_dims_hwl 1.52,1.64,3.86 \
  --height_m 1.52 \
  --force_veh_type sedan \
  --vt_comp_pkl_map "compact=-,sedan=$COMP_MODEL,suv=-,van=-" \
  --proj_center_source pred_reproj_geom_x \
  --proj_refine_iters 3 \
  --tightfit \
  --tightfit_max_nfev 15 \
  --tightfit_k 50 \
  --conf 0.25 \
  --imgsz 1280 \
  --device cuda:0 \
  --half \
  --pred_meta_csv "$OUT/pred_meta.csv"
```

### 15.3 參數含義

- `--egonet_weights`：EgoNet checkpoint 目錄。
- `--dims_source fixed`：使用固定或車型尺寸先驗。
- `--force_veh_type sedan`：略過分類器，所有偵測視為 sedan。
- `--vt_comp_pkl_map`：為 sedan 指定補償模型，其他車型停用。
- `--proj_center_source pred_reproj_geom_x`：使用預測 3D 幾何中心的水平回饋修正射線。
- `--proj_refine_iters 3`：最多三次 feedback。
- `--tightfit`：啟用投影一致性最佳化。
- `--pred_meta_csv`：輸出中間幾何量，建議永遠明確指定到本次輸出資料夾。

---

## 16. 多車型推論

以 YOLO classification 為例：

```bash
KITTI_ROOT=/path/to/KITTI_ROOT
OUT=/path/to/outputs/gemo3d_multitype

python gemo3d.py \
  --kitti_root "$KITTI_ROOT" \
  --split val \
  --out_dir "$OUT" \
  --yolo_weights weights/yolo11n.pt \
  --egonet_weights weights/egonet \
  --dims_source fixed \
  --fixed_dims_hwl 1.55,1.74,3.86 \
  --height_m 1.55 \
  --vehclf_backend yolo_cls \
  --vehclf_yolo_weights weights/vehicle_type_cls.pt \
  --vehclf_yolo_imgsz 224 \
  --vehclf_yolo_device cuda:0 \
  --vehclf_yolo_half \
  --vehclf_yolo_label_map "hatchback=compact,pickup=compact,sedan=sedan,suv=suv,van=van" \
  --vehclf_min_prob 0.20 \
  --vt_comp_pkl_map "compact=models/compact_rank_ratio_linear_model.pkl,sedan=models/sedan_rank_ratio_linear_model.pkl,suv=models/suv_rank_ratio_linear_model.pkl,van=models/van_rank_ratio_linear_model.pkl" \
  --proj_center_source pred_reproj_geom_x \
  --proj_refine_iters 3 \
  --tightfit \
  --tightfit_max_nfev 15 \
  --conf 0.25 \
  --imgsz 1280 \
  --device cuda:0 \
  --half \
  --pred_meta_csv "$OUT/pred_meta.csv"
```

當分類信心低於 `--vehclf_min_prob` 時，車型會成為 `unknown`，尺寸與補償將退回全域設定。

若希望 `unknown` 使用全域補償：

```bash
--comp_pkl models/default_rank_ratio_linear_model.pkl
```

---

## 17. 主要參數說明

### 17.1 資料與輸出

| 參數 | 說明 | 預設 |
|---|---|---|
| `--kitti_root` | KITTI 格式資料集根目錄 | `/home/e114/KITTI_ROOT` |
| `--split` | 執行 `ImageSets/<split>.txt` | `val` |
| `--out_dir` | 推論輸出根目錄 | `/home/e114/KITTI_ROOT/out` |
| `--pred_meta_csv` | 中間量 CSV 路徑；空字串可關閉 | `pred_meta.csv` |
| `--vis_dir` | 視覺化輸出目錄；空字串關閉 | 空 |

### 17.2 YOLO

| 參數 | 說明 | 預設 |
|---|---|---|
| `--yolo_weights` | YOLO detector 權重 | `yolo11l.pt` |
| `--conf` | 偵測信心門檻 | `0.65` |
| `--imgsz` | YOLO 輸入尺寸 | `1280` |
| `--device` | `cuda`、`cuda:0` 或 `cpu` | `cuda` |
| `--half` | 使用 FP16 | 關閉 |
| `--car_names` | 視為車輛的 YOLO 類別名稱 | `car` |

實驗建議先從：

```text
conf=0.25
imgsz=640
yolo11n.pt
```

開始，再逐步增加模型與影像尺寸。

### 17.3 朝向與尺寸

| 參數 | 說明 |
|---|---|
| `--no_egonet` | 停用 EgoNet |
| `--egonet_weights` | EgoNet checkpoint 目錄 |
| `--dims_source fixed` | 使用固定／車型尺寸 |
| `--dims_source egonet` | 使用 EgoNet 尺寸 |
| `--dims_source deep3d` | 使用 Deep3DBox 尺寸 |
| `--fixed_dims_hwl` | 固定尺寸，順序必須是 `h,w,l` |
| `--height_m` | 針孔初始深度使用的車高 |
| `--ry_offset_deg` | 對 `ry` 加上全域角度偏移 |

內建車型尺寸先驗：

| 車型 | `(h, w, l)` |
|---|---|
| compact | `(1.47, 1.72, 3.95)` |
| sedan | `(1.52, 1.64, 3.86)` |
| suv | `(1.65, 1.78, 4.40)` |
| van | `(2.05, 1.92, 5.10)` |

### 17.4 補償

| 參數 | 說明 |
|---|---|
| `--comp_pkl` | 全域補償模型 |
| `--vt_comp_pkl_map` | 四車型補償模型 map |
| `--compact_comp_pkl` | compact 單獨覆寫 |
| `--sedan_comp_pkl` | sedan 單獨覆寫 |
| `--suv_comp_pkl` | suv 單獨覆寫 |
| `--van_comp_pkl` | van 單獨覆寫 |
| `--force_veh_type` | 強制所有偵測為同一車型 |

### 17.5 投影修正

`--proj_center_source` 可選：

| 模式 | 說明 |
|---|---|
| `bbox_bottom` | 使用 2D bbox 底邊中心 |
| `bbox_center` | 使用 2D bbox 幾何中心 |
| `pred_reproj_bottom` | 使用預測 bottom-center 投影回饋 |
| `pred_reproj_geom_x` | 使用預測幾何中心修正水平射線，垂直位置保持 bottom 約束 |
| `gt_bottom` | 使用 GT bottom，僅限診斷 |
| `gt_geom` | 使用 GT 幾何中心，僅限診斷 |

正式推論建議：

```bash
--proj_center_source pred_reproj_geom_x \
--proj_refine_iters 3
```

### 17.6 tight-fit

| 參數 | 說明 |
|---|---|
| `--tightfit` | 啟用最佳化 |
| `--tightfit_max_nfev` | 最大函式評估次數 |
| `--tightfit_k` | smooth min/max 銳利程度 |
| `--fixed_ground_y` | 固定 bottom-center Y；大於 0 才啟用 |

---

## 18. 輸出檔案說明

### 18.1 KITTI prediction

輸出位置：

```text
<out_dir>/pred/data/
```

例如：

```text
outputs/gemo3d/
└── pred/
    └── data/
        ├── 000000.txt
        ├── 000001.txt
        └── ...
```

每一行格式：

```text
type trunc occl alpha x1 y1 x2 y2 h w l x y z ry score
```

範例：

```text
Car 0.00 0 0.123456 500.00 120.00 700.00 300.00 1.52 1.64 3.86 0.10 1.65 20.00 0.100000 0.950000
```

某張影像沒有任何偵測時，程式仍會建立空白 `.txt`。這是正常行為，方便 KITTI evaluator 對齊所有影像 ID。

### 18.2 pred_meta.csv

建議設定：

```bash
--pred_meta_csv "$OUT/pred_meta.csv"
```

重要欄位：

| 欄位 | 說明 |
|---|---|
| `image_id` | 影像 ID |
| `veh_type` | 最終車型 |
| `yolo_conf` | YOLO 信心 |
| `pred_x/y/z` | 最終 3D bottom center |
| `pred_ry` | 最終旋轉角 |
| `pred_h/w/l` | 最終尺寸 |
| `z_est` | 補償前初始深度 |
| `comp_final` | 最終補償量 |
| `entered_second_pass` | 是否進入 feedback |
| `feedback_iters_used` | feedback 次數 |
| `u_init/u_final` | 射線像素水平位置變化 |
| `theta_ray_init/theta_ray_final_used` | 射線角變化 |
| `ry_init/ry_final` | 朝向角變化 |
| `used_ground_y` | 是否使用地面 Y |

### 18.3 視覺化

啟用：

```bash
--vis_dir "$OUT/vis"
```

顯示 GT：

```bash
--vis_dir "$OUT/vis" \
--vis_draw_gt
```

視覺化會增加：

- CPU 負擔。
- 記憶體用量。
- 磁碟寫入量。
- 整體執行時間。

大量推論時建議關閉。

---

## 19. 如何檢查推論是否成功

### 19.1 prediction 數量是否與 split 一致

```bash
wc -l "$KITTI_ROOT/ImageSets/val.txt"
find "$OUT/pred/data" -maxdepth 1 -name '*.txt' | wc -l
```

兩者應相同。

### 19.2 查看第一個非空結果

```bash
find "$OUT/pred/data" -type f -size +0c | sort | head
```

顯示內容：

```bash
FIRST=$(find "$OUT/pred/data" -type f -size +0c | sort | head -n 1)
echo "$FIRST"
cat "$FIRST"
```

### 19.3 檢查每行欄位數

KITTI prediction 含 score 時每行應有 16 欄：

```bash
python - "$OUT/pred/data" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
bad = []

for path in sorted(root.glob("*.txt")):
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        n = len(line.split())
        if n != 16:
            bad.append((path.name, line_no, n))

print("bad rows:", len(bad))
for row in bad[:20]:
    print(row)
PY
```

### 19.4 檢查 feedback 是否生效

```bash
python - "$OUT/pred_meta.csv" <<'PY'
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])

cols = [
    "entered_second_pass",
    "feedback_iters_used",
    "u_init",
    "u_final",
    "theta_ray_init",
    "theta_ray_final_used",
]

print(df[cols].head(20))
print()
print("second-pass ratio:", df["entered_second_pass"].mean())
print("mean |u_final-u_init|:", (df["u_final"] - df["u_init"]).abs().mean())
PY
```

若 `entered_second_pass=1`，但 `u_init` 與 `u_final` 幾乎相同，代表分支有進入，只是該樣本的幾何修正量很小。

---

## 20. 常見錯誤與排除方式

### 20.1 找不到 `nearest_time_topbev_ncomp.py`

錯誤原因：目前入口已更名。

正確：

```bash
python gemo3d.py --help
```

### 20.2 `No module named 'gemo3d_ncomp'`

目前套件已更名為 `submodule`。

主程式不需修改；若是 smoke test，執行：

```bash
sed -i 's/gemo3d_ncomp/submodule/g' tests/smoke_test.py
```

### 20.3 `No module named 'torch'`

先安裝 PyTorch。即使使用 `--no_egonet`，目前模組匯入仍需要 torch。

### 20.4 `Cannot import 'libs.model.egonet'`

確認：

```bash
export EGONET_ROOT="$PWD/EgoNet-master"
export EGONET_CFG="$PWD/EgoNet-master/configs/KITTI_inference:test_submission.yml"
```

檢查：

```bash
test -f "$EGONET_ROOT/libs/model/egonet.py"
```

### 20.5 找不到 EgoNet config

明確指定：

```bash
export EGONET_CFG="$PWD/EgoNet-master/configs/KITTI_inference:test_submission.yml"
```

注意檔名中包含冒號 `:`，shell 仍可正常處理，建議加雙引號。

### 20.6 EgoNet checkpoint 載入失敗

確認 `--egonet_weights` 指向的是目錄，而不是單一 `.pth`：

```bash
ls -lah weights/egonet
```

### 20.7 出現開發者主機的 `/home/e114/...pkl` 警告

明確覆寫或停用四車型模型：

```bash
--vt_comp_pkl_map "compact=-,sedan=-,suv=-,van=-"
```

### 20.8 所有 prediction 都是空白

檢查：

1. YOLO 權重是否正確。
2. `--conf` 是否過高。
3. `--car_names` 是否符合模型類別名稱。
4. 影像是否可讀。
5. 影像中是否真的有車輛。

嘗試：

```bash
--conf 0.25 \
--car_names car,vehicle,truck,bus
```

### 20.9 `Missing image`

程式固定尋找：

```text
training/image_2/<id>.png
```

檢查：

```bash
ID=$(head -n 1 "$KITTI_ROOT/ImageSets/val.txt")
ID="${ID%.*}"
ls "$KITTI_ROOT/training/image_2/$ID.png"
```

### 20.10 `Missing calibration`

每張影像都需同名：

```text
training/calib/<id>.txt
```

且必須含 `P2:`。

### 20.11 CUDA out of memory

依序降低：

```text
yolo11l.pt → yolo11n.pt
imgsz 1280 → 960 → 640
```

並使用：

```bash
--half
```

關閉：

```text
--vis_dir
vehclf backend
```

### 20.12 CPU 使用 `--half` 發生錯誤

CPU 執行時移除：

```text
--half
--vehclf_yolo_half
```

### 20.13 `libGL.so.1` 錯誤

改用：

```bash
pip uninstall -y opencv-python
pip install opencv-python-headless
```

### 20.14 NumPy pickle 相容問題

目前程式已有 `numpy._core` 相容 shim。若仍失敗：

- 在建立模型與推論時使用相同 NumPy 主版本。
- 重新輸出 `.pkl`。
- 不要任意對不可信 `.pkl` 執行反序列化。

### 20.15 `--max_images` 或 `--continue_on_error` 不支援

目前 CLI 沒有這兩個參數。不要加入：

```text
--max_images
--continue_on_error
```

需限制影像數量時，建立小型 split，方式見下一節。

### 20.16 `unrecognized arguments`

先確認目前版本支援的參數：

```bash
python gemo3d.py --help
```

不要直接複製舊版 `nearest_time_topbev_ncomp.py` 的所有參數。

---

## 21. 建立小型測試 split

先取 `val.txt` 的前 10 張：

```bash
head -n 10 \
  "$KITTI_ROOT/ImageSets/val.txt" \
  > "$KITTI_ROOT/ImageSets/debug10.txt"
```

執行：

```bash
python gemo3d.py \
  --kitti_root "$KITTI_ROOT" \
  --split debug10 \
  --out_dir "$OUT/debug10" \
  ...
```

隨機取 20 張：

```bash
shuf "$KITTI_ROOT/ImageSets/val.txt" \
  | head -n 20 \
  > "$KITTI_ROOT/ImageSets/debug20.txt"
```

建議先成功執行 5～20 張，再跑完整資料集。

---

## 22. 資源不足時的安全設定

推薦的低資源測試設定：

```bash
python gemo3d.py \
  --kitti_root "$KITTI_ROOT" \
  --split debug10 \
  --out_dir "$OUT/debug10" \
  --yolo_weights weights/yolo11n.pt \
  --no_egonet \
  --dims_source fixed \
  --fixed_dims_hwl 1.55,1.74,3.86 \
  --height_m 1.55 \
  --vt_comp_pkl_map "compact=-,sedan=-,suv=-,van=-" \
  --conf 0.25 \
  --imgsz 640 \
  --device cuda:0 \
  --half \
  --pred_meta_csv "$OUT/debug10/pred_meta.csv"
```

避免同時：

- 執行多個資料集。
- 開啟大量視覺化。
- 使用大型 YOLO。
- 使用多個車型分類器。
- 將 `tightfit_max_nfev` 設得非常大。

建議監控：

```bash
watch -n 1 nvidia-smi
```

系統記憶體：

```bash
watch -n 1 free -h
```

---

## 23. 正式實驗的建議設定

### 23.1 不使用 GT 輸入

正式結果不要啟用：

```text
--depth_use_gt_height
--pred_use_gt_hwl
--pred_center_source gt
--proj_center_source gt_bottom
--proj_center_source gt_geom
```

### 23.2 明確固定所有路徑

不要依賴程式內的 `/home/e114/...` 預設值。正式 shell script 應明確指定：

```text
KITTI_ROOT
OUT
YOLO weights
EgoNet weights
vehicle classifier weights
compensation models
pred_meta.csv
```

### 23.3 保留執行命令與版本

```bash
git rev-parse HEAD > "$OUT/git_commit.txt"
python --version > "$OUT/python_version.txt"
pip freeze > "$OUT/pip_freeze.txt"
nvidia-smi > "$OUT/nvidia_smi.txt"
```

將實際執行命令存成：

```text
run_gemo3d.sh
```

### 23.4 建議 shell script

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

conda activate gemo3d

export EGONET_ROOT="$PWD/EgoNet-master"
export EGONET_CFG="$PWD/EgoNet-master/configs/KITTI_inference:test_submission.yml"

KITTI_ROOT=/path/to/KITTIDataset_Carla
OUT=/path/to/results/experiment_name
COMP_MODEL="$PWD/models/sedan_rank_ratio_linear_model.pkl"

mkdir -p "$OUT"

python gemo3d.py \
  --kitti_root "$KITTI_ROOT" \
  --split val \
  --out_dir "$OUT" \
  --yolo_weights "$PWD/weights/yolo11n.pt" \
  --egonet_weights "$PWD/weights/egonet" \
  --dims_source fixed \
  --fixed_dims_hwl 1.52,1.64,3.86 \
  --height_m 1.52 \
  --force_veh_type sedan \
  --vt_comp_pkl_map "compact=-,sedan=$COMP_MODEL,suv=-,van=-" \
  --proj_center_source pred_reproj_geom_x \
  --proj_refine_iters 3 \
  --tightfit \
  --tightfit_max_nfev 15 \
  --tightfit_k 50 \
  --conf 0.25 \
  --imgsz 1280 \
  --device cuda:0 \
  --half \
  --pred_meta_csv "$OUT/pred_meta.csv"

git rev-parse HEAD > "$OUT/git_commit.txt"
pip freeze > "$OUT/pip_freeze.txt"
```

賦予執行權限：

```bash
chmod +x run_gemo3d.sh
./run_gemo3d.sh
```

---

## 24. 環境備份與移植

### 24.1 匯出 Conda 環境

完整匯出：

```bash
conda env export -n gemo3d > environment-gemo3d.yml
```

較精簡、只保留手動安裝項目：

```bash
conda env export -n gemo3d --from-history > environment-gemo3d-history.yml
```

### 24.2 在另一台主機重建

```bash
conda env create -f environment-gemo3d.yml
conda activate gemo3d
```

注意：

- 不同 GPU、driver 或 CUDA 環境可能需要重新安裝 PyTorch。
- 權重、資料集與 `.pkl` 不會包含在 Conda 環境中，需另外複製。
- `EGONET_ROOT` 與 `EGONET_CFG` 需依新路徑重新設定。

### 24.3 使用 conda-pack 原封不動搬移

來源主機：

```bash
conda install -n base -c conda-forge conda-pack -y
conda pack -n gemo3d -o gemo3d.tar.gz
```

目標主機：

```bash
mkdir -p ~/conda_envs/gemo3d
tar -xzf gemo3d.tar.gz -C ~/conda_envs/gemo3d
source ~/conda_envs/gemo3d/bin/activate
conda-unpack
```

跨不同 CUDA／driver 主機時仍需重新確認：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 25. 專案結構

目前主要結構：

```text
GEMO3D/
├── gemo3d.py                  # 目前 CLI 入口
├── submodule/
│   ├── __init__.py
│   ├── cli.py                 # CLI、資料集迴圈、模型初始化
│   ├── config.py              # KITTI I/O、車型尺寸、共用設定
│   ├── compensation.py        # 補償模型載入與推論
│   ├── geometry.py            # 投影、反投影、3D box 幾何
│   ├── optimization.py        # ground-plane 與 tight-fit
│   ├── pipeline.py            # 單張影像推論主流程
│   └── visualization.py       # 2D／3D／BEV 視覺化
├── egonet_ori.py              # EgoNet adapter
├── deep3d_head.py             # Deep3DBox adapter
├── vehicletype_head.py        # 車型分類器 adapter
├── EgoNet-master/             # EgoNet 原始碼
├── requirements.txt
├── README.md
└── 教學.md
```

建議自行新增：

```text
weights/
├── yolo11n.pt
├── vehicle_type_cls.pt
└── egonet/

models/
├── compact_rank_ratio_linear_model.pkl
├── sedan_rank_ratio_linear_model.pkl
├── suv_rank_ratio_linear_model.pkl
└── van_rank_ratio_linear_model.pkl
