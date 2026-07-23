# GEMO3D Nearest-Time NComp 推論工具

此專案將原本單一檔案、超過 4,000 行的 `nearest_time_topbev_ncomp.py` 重構為可維護的 Python 套件，同時保留原本的命令列入口與主要推論行為。

本工具以單張 RGB 影像為輸入，整合：

1. YOLO 2D 車輛偵測
2. EgoNet 或 Deep3DBox 朝向／尺寸估計
3. 車型分類與車型尺寸先驗
4. 針孔模型初始深度估計
5. Rank-1 投影高度比例補償或舊式加法補償
6. 道路平面約束
7. 3D 邊界框投影一致性修正
8. KITTI label 格式輸出與可選的除錯視覺化

---

## 1. 重構內容

原始程式的主要問題是所有功能都集中在同一個檔案，且 `infer_one_image()` 單一函式接近 1,700 行。重構後依功能拆分如下：

```text
nearest_time_topbev_ncomp_refactored/
├── nearest_time_topbev_ncomp.py   # 相容原指令的執行入口
├── README.md
├── requirements.txt
├── tests/
│   └── smoke_test.py
└── gemo3d_ncomp/
    ├── __init__.py
    ├── cli.py                     # CLI、路徑整理、模型初始化、資料集迴圈
    ├── config.py                  # 車型設定、KITTI I/O、共用工具
    ├── compensation.py            # 補償 PKL 載入與推論
    ├── geometry.py                # 相機投影、3D box 幾何
    ├── optimization.py            # 地面約束與 tight-fit 最佳化
    ├── pipeline.py                # 單張影像推論流程
    └── visualization.py           # 2D/3D/BEV 視覺化與 IoU
```

已移除的確定冗餘內容：

- 重複執行第二次的車型分類程式碼
- 已以三引號停用的舊版 3D 平移流程
- `pred_reproj_geom_x` 中重複的像素範圍限制
- 完全相同的回饋迴圈收斂判斷分支
- 未被正式流程使用的 VT6 固定設定與載入函式

原始 CLI 參數大致保留，因此既有 shell script 通常只需要調整 Python 檔案所在位置。

---

## 2. 座標與輸出定義

本程式遵循 KITTI camera coordinate system：

- `X`：影像右方
- `Y`：影像下方
- `Z`：相機前方
- `(x, y, z)`：3D 邊界框底部中心
- 尺寸順序：`(h, w, l)`
- `ry`：繞相機 Y 軸旋轉角，單位為弧度

輸出格式：

```text
type trunc occl alpha x1 y1 x2 y2 h w l x y z ry score
```

輸出位置：

```text
<out_dir>/pred/data/000000.txt
<out_dir>/pred/data/000001.txt
...
```

即使某張影像沒有偵測結果，也會建立空白 `.txt`，方便直接交給 KITTI 評估工具。

---

## 3. 資料集目錄

預期資料結構如下：

```text
KITTI_ROOT/
├── ImageSets/
│   └── val.txt
└── training/
    ├── image_2/
    │   ├── 000000.png
    │   └── ...
    ├── calib/
    │   ├── 000000.txt
    │   └── ...
    ├── label_2/          # GT／消融與視覺化時才需要
    ├── meta/             # CARLA 外參、地面平面與額外相機資訊
    ├── bev_iou/          # 可選鳥瞰底圖
    ├── fr_iou/           # 可選前側 IoU 相機影像
    └── side_iou/         # 可選側視 IoU 相機影像
```

`ImageSets/val.txt` 可使用下列任一格式：

```text
000000
000001.png
```

---

## 4. 安裝

### 4.1 基本套件

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch 與 CUDA 版本應依本機顯示卡與既有 EgoNet／Deep3DBox 環境安裝，不建議直接由 `requirements.txt` 強制指定版本。

### 4.2 外部模組

請將下列檔案或套件放在專案根目錄或 `PYTHONPATH`：

```text
egonet_ori.py
deep3d_head.py
vehicletype_head.py
```

使用 EgoNet 時，另需官方 EgoNet repository，並可設定：

```bash
export EGONET_ROOT=/path/to/EgoNet-master
export EGONET_CFG=/path/to/EgoNet-master/configs/KITTI_inference:test_submission.yml
```

不同車型分類後端的額外需求：

| 後端 | 額外套件／模型 |
|---|---|
| `none` | 無 |
| `ort` | `onnxruntime-gpu` 或 `onnxruntime`、ONNX 權重 |
| `paddleclas` | PaddlePaddle、PaddleClas |
| `yolo_cls` | Ultralytics classification `.pt` 權重 |

---

## 5. 快速開始

先確認命令列介面可正常載入：

```bash
python nearest_time_topbev_ncomp.py --help
```

### 5.1 最小基準版本

固定車輛尺寸、不使用 EgoNet、不使用補償：

```bash
python nearest_time_topbev_ncomp.py \
  --kitti_root /path/to/KITTI_ROOT \
  --split val \
  --out_dir outputs/baseline \
  --yolo_weights yolo11l.pt \
  --no_egonet \
  --dims_source fixed \
  --fixed_dims_hwl 1.55,1.74,3.86 \
  --comp_pkl "" \
  --conf 0.25 \
  --imgsz 1280 \
  --device cuda \
  --half
```

### 5.2 EgoNet + Rank-1 補償 + tight-fit

```bash
python nearest_time_topbev_ncomp.py \
  --kitti_root /path/to/KITTI_ROOT \
  --split val \
  --out_dir outputs/gemo3d \
  --yolo_weights yolo11l.pt \
  --egonet_weights weights/egonet \
  --dims_source fixed \
  --fixed_dims_hwl 1.52,1.64,3.86 \
  --comp_pkl models/sed_rank_ratio_linear_model.pkl \
  --proj_center_source pred_reproj_geom_x \
  --proj_refine_iters 3 \
  --tightfit \
  --tightfit_max_nfev 15 \
  --tightfit_k 50 \
  --conf 0.25 \
  --imgsz 1280 \
  --device cuda \
  --half
```

### 5.3 車型分類與各車型補償模型

以下範例使用 YOLO classification backend：

```bash
python nearest_time_topbev_ncomp.py \
  --kitti_root /path/to/KITTI_ROOT \
  --split val \
  --out_dir outputs/gemo3d_vehicle_type \
  --yolo_weights yolo11l.pt \
  --egonet_weights weights/egonet \
  --vehclf_backend yolo_cls \
  --vehclf_yolo_weights weights/vehicle_type_cls.pt \
  --vehclf_yolo_imgsz 224 \
  --vehclf_yolo_half \
  --vehclf_min_prob 0.20 \
  --vt_comp_pkl_map "compact=models/compact.pkl,sedan=models/sedan.pkl,suv=models/suv.pkl,van=models/van.pkl" \
  --proj_center_source pred_reproj_geom_x \
  --tightfit \
  --conf 0.25 \
  --device cuda \
  --half
```

也可分別覆寫：

```bash
--compact_comp_pkl models/compact.pkl \
--sedan_comp_pkl models/sedan.pkl \
--suv_comp_pkl models/suv.pkl \
--van_comp_pkl models/van.pkl
```

### 5.4 已知單一車型的 CARLA 實驗

若整組資料皆為同一車型，可跳過分類器：

```bash
python nearest_time_topbev_ncomp.py \
  --kitti_root /path/to/KITTI_ROOT/sedan_fr_T1R8/data/KITTIDataset_Carla \
  --split val \
  --out_dir outputs/sedan_fr_T1R8 \
  --force_veh_type sedan \
  --sedan_comp_pkl models/sed_rank_ratio_linear_model.pkl \
  --proj_center_source pred_reproj_geom_x \
  --proj_refine_iters 3 \
  --tightfit \
  --conf 0.25 \
  --device cuda \
  --half
```

---

## 6. 推論流程

每個 2D 偵測框依序進行：

### 6.1 2D 車輛偵測

YOLO 輸出：

```text
(x1, y1, x2, y2, confidence)
```

僅保留 `--car_names` 指定的類別，預設為 `car`。若資料集的類別名稱為 `vehicle`、`truck` 或 `bus`，可設定：

```bash
--car_names car,vehicle,truck,bus
```

### 6.2 尺寸與朝向

- `--dims_source fixed`：使用固定 `(h,w,l)` 或車型尺寸先驗
- `--dims_source egonet`：使用 EgoNet 輸出的尺寸
- `--dims_source deep3d`：使用 Deep3DBox 輸出的尺寸

當 head 提供 `alpha` 時，程式會以目前射線角重新計算：

```text
ry = alpha + theta_ray
```

如此可避免回饋更新像素射線後，`ry` 仍停留在舊射線的問題。

### 6.3 初始深度

以 2D 邊界框高度 `h_px` 與物體高度 `H` 計算：

```text
z_raw = fy * H / h_px
```

### 6.4 深度補償

支援兩種 PKL：

#### 舊式加法補償

```text
z_corrected = z_raw - compensation(angle)
```

PKL 可包含：

```text
spline
fourier
poly_coef
```

#### Rank-1 高度比例模型

```text
ratio = r(angle, z_raw)
z_corrected = z_raw * ratio
```

程式內部將其轉為相容既有流程的加法量：

```text
comp = z_raw * (1 - ratio)
z_corrected = z_raw - comp
```

### 6.5 地面平面約束

優先順序：

1. `--fixed_ground_y > 0`
2. `training/meta/<id>.json` 中的前視相機外參
3. 2D 像素反投影所得的 Y

當 CARLA meta 可用時，會由地面平面計算符合道路幾何的 bottom-center Y。

### 6.6 投影一致性修正

`--tightfit` 啟用後，程式會調整 3D 平移，使投影邊界更接近 2D bbox。

常用投影中心模式：

| 模式 | 說明 |
|---|---|
| `bbox_bottom` | 2D bbox 底邊中心 |
| `bbox_center` | 2D bbox 幾何中心 |
| `pred_reproj_bottom` | 以預測 3D bottom-center 投影進行回饋 |
| `pred_reproj_geom_x` | 以預測 3D 幾何中心修正水平射線，Y 維持底邊約束 |
| `gt_bottom` | 僅供診斷／消融 |
| `gt_geom` | 僅供診斷／消融 |

正式推論建議：

```bash
--proj_center_source pred_reproj_geom_x --proj_refine_iters 3
```

---

## 7. 重要參數

### 偵測

| 參數 | 說明 |
|---|---|
| `--yolo_weights` | YOLO detector 權重 |
| `--conf` | 偵測信心閾值 |
| `--imgsz` | YOLO 輸入尺寸 |
| `--car_names` | 視為 Car 的類別名稱 |
| `--device` | `cuda`、`cuda:0` 或 `cpu` |
| `--half` | YOLO FP16 |

### 3D 頭與尺寸

| 參數 | 說明 |
|---|---|
| `--no_egonet` | 停用 EgoNet |
| `--egonet_weights` | EgoNet 權重目錄 |
| `--dims_source` | `fixed`、`egonet`、`deep3d` |
| `--fixed_dims_hwl` | 固定 `(h,w,l)` |
| `--deep3d_weights` | Deep3DBox 權重 |
| `--ry_offset_deg` | 全域朝向角偏移 |

### 補償

| 參數 | 說明 |
|---|---|
| `--comp_pkl` | 全域補償模型 |
| `--vt_comp_pkl_map` | 車型對應補償模型 |
| `--force_veh_type` | 強制指定 `compact/sedan/suv/van` |

### 幾何修正

| 參數 | 說明 |
|---|---|
| `--proj_center_source` | 射線／反投影的像素來源 |
| `--proj_refine_iters` | 回饋迭代次數 |
| `--tightfit` | 啟用投影一致性最佳化 |
| `--tightfit_max_nfev` | 最佳化函數評估上限 |
| `--tightfit_k` | smooth min/max 銳利程度 |
| `--fixed_ground_y` | 固定 KITTI bottom-center Y |

### 輸出與除錯

| 參數 | 說明 |
|---|---|
| `--pred_meta_csv` | 每筆預測的中間量 CSV |
| `--vis_dir` | 視覺化輸出目錄 |
| `--vis_draw_gt` | 顯示 GT 3D box |
| `--vis_iou` | GT 配對的 2D IoU 閾值 |

---

## 8. GT 輔助參數警告

以下參數會使用 `label_2` 的真值，只應用於診斷、消融或上限分析，不應納入正式推論結果：

```text
--depth_use_gt_height
--pred_use_gt_hwl
--pred_center_source gt
--proj_center_source gt_bottom
--proj_center_source gt_geom
```

正式比較其他 monocular 3D 方法時，請確認上述選項皆未啟用。

---

## 9. 視覺化輸出

設定：

```bash
--vis_dir outputs/vis --vis_draw_gt
```

程式會依可用資料建立 2D bbox、3D box、局部 BEV、CARLA top-view 與 IoU 對照影像。視覺化會增加 CPU、記憶體與磁碟負擔；大量推論或資源有限時建議關閉。

---

## 10. 硬體資源控制

為避免 GPU 記憶體或主記憶體耗盡：

- 先使用 `--imgsz 640` 或 `960` 驗證流程
- GPU 支援時啟用 `--half`
- 不需要時將 `--vehclf_backend none`
- 不設定 `--vis_dir`
- 將 `--tightfit_max_nfev` 保持在 3～15
- 使用單一 GPU，例如 `--device cuda:0`
- 一次只處理一個資料集 split

除錯時可先使用只含 5～20 張影像的小型 split。

---

## 11. 常見問題

### 11.1 所有 prediction txt 都是空的

檢查：

```bash
--car_names
--conf
--yolo_weights
```

若 YOLO 模型輸出名稱不是 `car`，需加入正確名稱，例如：

```bash
--car_names car,vehicle
```

### 11.2 3D box 旋轉 90°

可先測試：

```bash
--ry_offset_deg 90
```

或：

```bash
--ry_offset_deg -90
```

但應優先確認 head 的 `alpha`、`ry` 定義是否與 KITTI 一致。

### 11.3 補償模型沒有生效

確認啟動訊息中有成功載入 PKL，並檢查：

- 檔案路徑是否存在
- 車型分類結果是否為 `unknown`
- PKL 是否包含支援的 key
- Rank-1 模型的角度與深度網格是否涵蓋推論範圍

### 11.4 NumPy 2.x 建立的 PKL 無法在 NumPy 1.x 載入

程式內已保留 `numpy._core` 相容 shim。若仍失敗，建議在建立 PKL 與推論環境使用相同 NumPy 主版本重新輸出模型。

### 11.5 有 meta 但地面約束未生效

確認 JSON 內的：

```text
extrinsics
```

可轉為 `4x4` world-to-sensor matrix。若格式錯誤，程式會退回 bbox 像素反投影。

### 11.6 `pred_reproj_geom_x` 看起來沒有改變結果

查看 `pred_meta.csv`：

```text
entered_second_pass
feedback_iters_used
u_init
u_final
theta_ray_init
theta_ray_final_used
```

若 `u_init` 與 `u_final` 幾乎相同，代表回饋已進入但幾何修正量很小，而非分支未執行。

---

## 12. 開發與測試

語法檢查：

```bash
python -m compileall -q .
```

執行不載入真實模型的 smoke test：

```bash
PYTHONPATH=/path/to/project/root:$PYTHONPATH \
python tests/smoke_test.py
```

測試涵蓋：

- KITTI 相機反投影
- 3D box 投影
- BEV IoU／3D IoU
- Rank-1 ratio 補償
- 無偵測影像建立空 prediction file
- 單筆假偵測建立合法 KITTI prediction line

真實模型、EgoNet、Deep3DBox、CARLA meta 與完整資料集仍需在原研究環境中進行端到端驗證。

---

## 13. 建議的 GitHub 使用方式

將此資料夾放入既有 repository，並使外部 head adapter 與入口檔同層：

```text
repo/
├── nearest_time_topbev_ncomp.py
├── egonet_ori.py
├── deep3d_head.py
├── vehicletype_head.py
├── gemo3d_ncomp/
├── tests/
├── README.md
└── requirements.txt
```

執行：

```bash
cd repo
python nearest_time_topbev_ncomp.py --help
```

不要將大型權重、資料集與推論輸出直接提交到 Git。建議在 repository 的 `.gitignore` 中加入：

```gitignore
*.pt
*.pth
*.onnx
*.pkl
outputs/
pred_meta.csv
__pycache__/
*.pyc
```
