# YOLO26

> 本文档用于深入学习 YOLO26：**NMS-free 原生端到端检测**、**DFL 移除**、**双头 one2one/one2many 训练**，以及与 YOLOv5 / YOLOv8 / YOLO11 的**全维度对比**。  
> 检测头 TAL、Loss 细节见 [YOLOv8 笔记](./YOLOv8.md)；C3k2 / C2PSA 见 [YOLO11 笔记](./YOLOv11.md)；Pose26 / RLE 见 [YOLOv8 §20](./YOLOv8.md#20-pose26-与-rle-lossyolo26-姿态分支)。  
> 代码：[ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) · 官方文档：[YOLO26](https://docs.ultralytics.com/models/yolo26/)

---

## 1. 基线：YOLO26 概览

### 1.1 基本信息

| 项目 | 内容 |
|------|------|
| 作者/机构 | Ultralytics |
| 发布时间 | **2025 年**（YV25 大会发布） |
| 定位 | **Edge-first**：面向 CPU / 嵌入式 / 无 GPU 边缘部署 |
| 任务 | detect / seg / **sem** / pose / obb / cls（比 v11 多 **语义分割**） |
| 代码 | `pip install ultralytics`，`YOLO("yolo26n.pt")` |
| 论文 | **无正式论文**，工程迭代；端到端思想源自 YOLOv10（清华） |
| 相对 v11 | **NMS-free + reg_max=1 + MuSGD + ProgLoss/STAL** |

### 1.2 五大核心卖点（重点）

```
┌─────────────────────────────────────────────────────────────────┐
│  ① NMS-free 原生端到端    推理直接出 (N,300,6)，无需后处理 NMS   │
│  ② 移除 DFL (reg_max=1)   框回归简化为 4 通道，导出/边缘兼容更好  │
│  ③ CPU 推理最高 +43%      同档 n 模型：11n 56ms → 26n 39ms ONNX  │
│  ④ 双头训练 one2many+one2one  训练多正样本，推理对齐无 NMS 分支   │
│  ⑤ 训练创新 MuSGD + ProgLoss + STAL  小目标 AP 提升、收敛更稳    │
└─────────────────────────────────────────────────────────────────┘
```

| 卖点 | 解决的问题 | 受益场景 |
|------|-----------|----------|
| **NMS-free** | NMS 在 CPU 上慢、各平台实现不一致、TopK 算子难导出 | 摄像头、机器人、工业 PC 无 GPU |
| **无 DFL** | reg_max=16 的 Softmax+积分解码增加算子与延迟 | TensorRT / ONNX / MCU 类边缘芯片 |
| **双头架构** | 纯 one2one 训练难收敛；纯 one2many 推理仍需 NMS | 默认 e2e 部署，需要时可 `end2end=False` 换精度 |
| **MuSGD** | 纯 SGD 在大 batch / 长 schedule 下震荡 | COCO 全量训练、迁移微调 |
| **ProgLoss + STAL** | 小目标在 TAL 中易被大目标压制 | 无人机、监控远景、IoT 视觉 |

### 1.3 整体架构（相对 v11 的变化标注）

```
Input (B, 3, 640, 640)
    │
    ▼
Backbone: C3k2 + SPPF★ + C2PSA
    │   SPPF: k=5,3 + concat 变体（非 v8/v11 标准 SPPF）
    ▼
Neck: PAN-FPN（C3k2，neck 层普遍 c3k=True）
    │
    ▼
Head: Detect（双头）
    │   one2many (o2m): 传统 (B, nc+4, 8400) → 训练 + 可选 NMS 推理
    │   one2one  (o2o):  (B, 300, 6)         → 默认 NMS-free 推理
    │   reg_max=1 → 回归仅 4 通道（无 64 通道 DFL）
    ▼
Output（默认 end2end）:
  Detection:  (N, 300, 6)     [x1,y1,x2,y2, conf, cls_id]
  Pose:       (N, 300, 57)    17 关键点 × 3 + box+cls
  Seg:        (N, 300, 6+nm) + proto
```

### 1.4 官方精度（COCO val2017，640）

| 模型 | mAP@0.5:0.95 | mAP e2e | 参数量(M) | FLOPs(B) | CPU ONNX(ms) | T4 TRT10(ms) |
|------|--------------|---------|-----------|----------|--------------|--------------|
| YOLO26n | **40.9** | 40.1 | **2.4** | **5.4** | **38.9** | 1.7 |
| YOLO26s | **48.6** | 47.8 | 9.5 | 20.7 | 87.2 | 2.5 |
| YOLO26m | **53.1** | 52.5 | 20.4 | 68.2 | 220.0 | 4.7 |
| YOLO26l | **55.0** | 54.4 | 24.8 | 86.4 | 286.2 | 6.2 |
| YOLO26x | **57.5** | 56.9 | 55.7 | 193.9 | 525.8 | 11.8 |

> **e2e 列**：one2one 头、无 NMS 的验证 mAP，通常比 o2m+NMS **低 ~0.5~0.8**；默认部署用 e2e。  
> **fuse 后参数量**：导出时融合 Conv+BN 并去掉 o2m 辅助头，会比训练 checkpoint 计数更低。

---

## 2. 核心创新详解

### 2.1 NMS-free 端到端推理

**传统 v5/v8/v11 流水线：**

```
模型 raw 输出 (8400 候选)
    → conf 过滤
    → NMS（IoU 抑制重复框）    ← CPU 瓶颈、各平台实现差异大
    → 最终 boxes
```

**YOLO26 默认流水线：**

```
模型 one2one 头
    → TopK 内置（最多 300 框）
    → conf 阈值过滤即可        ← 无 NMS
    → 最终 boxes
```

| 任务 | end2end 输出形状 | 后处理 |
|------|-----------------|--------|
| Detection | `(N, 300, 6)` | conf 阈值 |
| Segmentation | `(N, 300, 6+nm)` + proto | conf + mask 系数 × proto |
| Pose | `(N, 300, 57)` | conf + 关键点 |
| OBB | `(N, 300, 7)` | conf + 角度 |

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# 默认：one2one，NMS-free
results = model.predict("image.jpg")

# 回退：one2many + NMS（精度通常更高 ~0.5 mAP）
results = model.predict("image.jpg", end2end=False)
```

**不支持 end2end 的导出格式（自动回退 o2m）：** NCNN、RKNN、PaddlePaddle、ExecuTorch、IMX、Edge TPU 等。

### 2.2 双头架构：one2many + one2one

训练时**两个检测头同时存在**，推理/导出默认只用 **one2one**。

```
                    训练阶段
                    ────────
特征 P3/P4/P5 ──┬── cv2/cv3 (one2many) ──► (B, nc+4, 8400)  ──► E2ELoss × o2m 权重
                │                              TAL topk=10
                └── cv2/cv3 (one2one)  ──► (B, 300, 6)       ──► E2ELoss × o2o 权重
                                               TAL topk=1~7

                    推理阶段（默认 end2end=True）
                    ────────────────────────────
                仅 one2one 头 ──► (N, 300, 6)
```

**渐进权重调度（Progressive Loss Balancing / ProgLoss 的一部分）：**

```
epoch 初期:  o2m 权重高 (~0.8)  → 多正样本，收敛快
epoch 后期:  o2o 权重高 (~0.9)  → 对齐 NMS-free 推理分布
```

这与 YOLOv10 的「训练用 aux 头、推理用 main 头」一脉相承，YOLO26 将其与 Ultralytics 统一的 TAL / 导出链路深度整合。

### 2.3 DFL 移除：reg_max=1

| 维度 | v8 / v11 | YOLO26 |
|------|----------|--------|
| reg_max | **16** | **1** |
| 回归通道 | 4×16=**64** | **4** |
| 解码 | Softmax 分布 → 积分期望 | **直接回归 LTRB 距离** |
| Loss | CIoU + **DFLLoss** | CIoU（无 DFL 项） |
| 导出 | DFL 算子在部分 NPU 上难实现 | 纯 Conv，兼容性更好 |

```yaml
# yolo26.yaml 全局参数
end2end: True
reg_max: 1
```

**代价：** 极端高 IoU 框的亚像素精度可能略逊于 reg_max=16；官方用 ProgLoss + 更长 schedule 补偿，检测 mAP 仍高于 v11。

### 2.4 SPPF 变体

```yaml
# v8 / v11
- [-1, 1, SPPF, [1024, 5]]

# YOLO26
- [-1, 1, SPPF, [1024, 5, 3, True]]
#                        │  │   └── concat 多尺度 pool 输出
#                        │  └── 第二级 kernel=3
#                        └── 第一级 kernel=5
```

在 P5 特征上并行/串行多核 MaxPool 后 concat，增强多尺度上下文，配合 STAL 改善小目标。

### 2.5 训练创新：MuSGD / ProgLoss / STAL

| 组件 | 作用 | 相对 v8/v11 |
|------|------|-------------|
| **MuSGD** | SGD + Muon 混合优化器（借鉴 LLM Kimi K2 训练） | v8/v11 默认 SGD/AdamW |
| **ProgLoss** | 训练过程中动态平衡 o2m / o2o 及各 loss 项权重 | 固定 loss 权重 |
| **STAL** | Small-Target-Aware Label Assignment，小目标在 TAL 中获得更多正样本 | 标准 TAL |

### 2.6 yaml 骨架（相对 v11 的差异）

| 位置 | YOLO11 | YOLO26 |
|------|--------|--------|
| 全局 | — | `end2end: True`, `reg_max: 1` |
| SPPF | `[1024, 5]` | `[1024, 5, 3, True]` |
| C2PSA | `[-1, 2, C2PSA, [1024]]` | 同左 |
| Neck C3k2 | 多数 `c3k=False` | 多数 **`c3k=True`** |
| P5 neck | `C3k2 [1024, True]` | `C3k2 [1024, True, 0.5, True]`（e=0.5 + c3k） |
| Head | Detect | Detect（内置双头，由 end2end 控制） |

Backbone 层数与 v11 **高度同构**（C3k2 + C2PSA），YOLO26 的突破主要在 **Head + Loss + 训练策略**，而非再换一套 backbone。

---

## 3. 扩展任务亮点

| 任务 | YOLO26 相对 v11 的增量 |
|------|------------------------|
| **Detect** | NMS-free e2e（核心） |
| **Segment** | 语义分割 loss + 多尺度 proto |
| **Semantic Seg** | **新增** yolo26*-sem（Cityscapes） |
| **Pose** | **Pose26 + RLE**（见 [YOLOv8 §20](./YOLOv8.md#20-pose26-与-rle-lossyolo26-姿态分支)） |
| **OBB** | 专用 angle loss + 边界连续解码 |
| **Open-Vocab** | **YOLOE-26**（NMS-free + 文本/视觉 prompt） |

**Pose 官方精度（COCO，e2e）：**

| 模型 | mAP pose 50-95 | mAP pose 50 | CPU ONNX(ms) |
|------|----------------|-------------|--------------|
| YOLO26n-pose | **57.2** | 83.3 | 40.3 |
| YOLO26s-pose | **63.0** | 86.6 | 85.3 |
| YOLO26m-pose | **68.8** | 89.6 | 218.0 |

---

## 4. 训练与推理

### 4.1 训练命令

```bash
# 检测（默认 MuSGD + E2ELoss + end2end yaml）
yolo detect train model=yolo26n.pt data=coco.yaml epochs=300 imgsz=640

# 姿态（Pose26 + RLE + E2ELoss）
yolo pose train model=yolo26n-pose.pt data=coco-pose.yaml epochs=100 imgsz=640
```

### 4.2 推理与验证

```bash
# NMS-free（默认）
yolo predict model=yolo26n.pt source=image.jpg
yolo val model=yolo26n.pt data=coco.yaml

# 传统 NMS 路径（精度优先）
yolo predict model=yolo26n.pt source=image.jpg end2end=False
yolo val model=yolo26n.pt data=coco.yaml end2end=False
```

### 4.3 导出注意

```bash
yolo export model=yolo26n.pt format=onnx          # 默认 e2e
yolo export model=yolo26n.pt format=onnx end2end=False  # o2m + 需外部 NMS
```

| 格式 | end2end 支持 | 说明 |
|------|-------------|------|
| ONNX / TensorRT / CoreML / OpenVINO | ✅ | 推荐 e2e 部署 |
| NCNN / RKNN / ExecuTorch | ❌ 自动禁用 | 回退 o2m |

---

## 5. YOLOv5 / v8 / v11 / v26 详细对比

> 四份本地笔记：[YOLOv5](./YOLOv5.md) · [YOLOv8](./YOLOv8.md) · [YOLO11](./YOLOv11.md) · **本文**

### 5.1 演进时间线与定位

```
2020 YOLOv5 ──► 2023 YOLOv8 ──► 2024 YOLO11 ──► 2025 YOLO26
 Anchor-based    Anchor-free      轻量化+注意力     Edge NMS-free E2E
 工程成熟        统一 ultralytics  参数↓ mAP↑        CPU↑ 部署简化
```

| 版本 | 年份 | 一句话定位 |
|------|------|-----------|
| **YOLOv5** | 2020 | Anchor-based 工程标杆，独立 yolov5 仓库 |
| **YOLOv8** | 2023 | Anchor-free + TAL + DFL，Ultralytics 统一框架起点 |
| **YOLO11** | 2024 | C3k2 + C2PSA + DWConv cls，**参数更少、mAP 更高** |
| **YOLO26** | 2025 | **NMS-free 原生 e2e**，去 DFL，**边缘 CPU 优先** |

### 5.2 架构总表（检测任务）

| 维度 | YOLOv5 | YOLOv8 | YOLO11 | **YOLO26** |
|------|--------|--------|--------|------------|
| **Anchor** | 9 个 K-Means | 无 | 无 | 无 |
| **Backbone 块** | C3 | C2f | C3k2 | C3k2 |
| **P5 注意力** | 无 | 无 | **C2PSA** | **C2PSA** |
| **SPPF** | 标准 k=5 | 标准 k=5 | 标准 k=5 | **k=5,3 concat 变体** |
| **Neck 块** | C3 | C2f | C3k2 | C3k2（**neck 多 c3k=True**） |
| **Cls 头** | 耦合 1×1 | Conv×2 legacy | **DWConv×2** | DWConv×2（同 v11） |
| **Reg 头** | 直接 4 值 | DFL reg_max=16 | DFL reg_max=16 | **直接 4 值 reg_max=1** |
| **obj 分支** | **有** | 无 | 无 | 无 |
| **检测头数量** | 1 | 1 | 1 | **2（o2m + o2o）** |
| **标签分配** | build_targets | TAL | TAL | TAL + **STAL** |
| **Box Loss** | CIoU | CIoU + DFL | CIoU + DFL | CIoU（**无 DFL**） |
| **优化器** | SGD | SGD/AdamW | SGD/AdamW | **MuSGD** |
| **推理后处理** | **NMS 必须** | **NMS 必须** | **NMS 必须** | **默认无 NMS** |
| **默认输出** | 动态框数 | (84,8400) raw | (84,8400) raw | **(300,6) 最终框** |
| **end2end** | 否 | 否 | 否 | **是（yaml 默认）** |

### 5.3 精度 / 参数量 / 速度（n 档，COCO 640）

| 指标 | YOLOv5n | YOLOv8n | YOLO11n | **YOLO26n** | 26 vs 11 |
|------|---------|---------|---------|-------------|----------|
| mAP@0.5:0.95 | 28.0 | 37.3 | 39.5 | **40.9** (e2e 40.1) | **+1.4** |
| 参数量 | 1.9M | 3.2M | 2.6M | **2.4M** | **-8%** |
| FLOPs | 4.5G | 8.9G | 6.6G | **5.4G** | **-18%** |
| CPU ONNX | — | — | 56.1 ms | **38.9 ms** | **~31% 更快** |
| T4 TRT10 | — | ~1.5 ms | ~1.5 ms | **1.7 ms** | 基本持平 |

### 5.4 全档位 mAP 与参数量

| 模型 | v5 mAP | v8 mAP | v11 mAP | **v26 mAP** | v26 e2e | v11 参数 | **v26 参数** |
|------|--------|--------|---------|-------------|---------|----------|--------------|
| n | 28.0 | 37.3 | 39.5 | **40.9** | 40.1 | 2.6M | **2.4M** |
| s | 37.4 | 44.9 | 47.0 | **48.6** | 47.8 | 9.4M | **9.5M** |
| m | 45.4 | 50.2 | 51.5 | **53.1** | 52.5 | 20.1M | **20.4M** |
| l | 49.0 | 52.9 | 53.4 | **55.0** | 54.4 | 25.3M | **24.8M** |
| x | 50.7 | 53.9 | 54.7 | **57.5** | 56.9 | 56.9M | **55.7M** |

**规律：** v26 在**全档位** mAP 均为系列最高；n/s 档参数量与 v11 接近或略少；m/l/x 档参数量与 v11 同量级。

### 5.5 训练范式对比

| 维度 | v5 | v8 | v11 | **v26** |
|------|----|----|-----|---------|
| 正样本分配 | 多 anchor + 邻域扩展 | TAL topk=10 | TAL topk=10 | TAL + **STAL**（小目标加权） |
| 回归目标 | anchor 偏移 | DFL 分布 + dist2bbox | 同 v8 | **4 边距离直接回归** |
| 置信度 | obj × cls | max(cls) | max(cls) | max(cls)，o2o 内置 TopK |
| Loss 项 | box + obj + cls | box + cls + dfl | 同 v8 | box + cls + **o2m/o2o 双路** |
| 特殊 Loss | — | — | — | **ProgLoss 动态权重** |
| Pose Loss | — | v8PoseLoss + OKS | 同 v8 | **PoseLoss26 + RLE** |

### 5.6 推理与部署对比

| 维度 | v5 | v8 | v11 | **v26** |
|------|----|----|-----|---------|
| 预测点数 | 25200 (3×8400) | 8400 | 8400 | o2m: 8400 / **o2o: ≤300** |
| NMS | **必须** | **必须** | **必须** | **默认不需要** |
| ONNX detect 输出 | (1,N,6) 后处理 | (1,84,8400) | (1,84,8400) | **(1,300,6) e2e** |
| DFL 算子 | 无 | **有** | **有** | **无** |
| TopK 算子 | 无 | 无 | 无 | **有（e2e 内置）** |
| 边缘友好度 | 中 | 中 | 中高 | **高（官方主打）** |
| 权重互迁 | — | ↔ v11 部分不可直载 | ↔ v8 不可直载 | ↔ v11 **不可直载** |

### 5.7 yaml 结构对照（n 档 backbone 摘要）

| Stage | YOLOv8n | YOLO11n | **YOLO26n** |
|-------|---------|---------|-------------|
| P2→P3 | C2f×3, 128ch | C3k2×2, e=0.25 | C3k2×2, e=0.25 |
| P3→P4 | C2f×6, 256ch | C3k2×2, e=0.25 | C3k2×2, e=0.25 |
| P4→P5 | C2f×6, 512ch | C3k2×2, c3k | C3k2×2, c3k |
| P5 | C2f×3 + SPPF | C3k2×2 + SPPF + **C2PSA×2** | C3k2×2 + **SPPF变体** + C2PSA×2 |
| depth_mult | **0.33** | **0.50** | **0.50** |
| width_mult | 0.25 | 0.25 | 0.25 |

v26 与 v11 骨架 **90%+ 相同**；主要差在 **SPPF 参数、neck c3k 开关、全局 reg_max/end2end**。

### 5.8 任务支持矩阵

| 任务 | v5 | v8 | v11 | **v26** |
|------|:--:|:--:|:---:|:-------:|
| Detect | ✅ | ✅ | ✅ | ✅ **e2e** |
| Segment | ✅ | ✅ | ✅ | ✅ **e2e + sem loss** |
| Semantic Seg | ❌ | ❌ | ❌ | ✅ **新增** |
| Pose | ✅ | ✅ | ✅ | ✅ **Pose26+RLE e2e** |
| OBB | ❌ | ✅ | ✅ | ✅ **angle loss** |
| Cls | ✅ | ✅ | ✅ | ✅ |
| Open-Vocab | ❌ | World | YOLOE-11 | **YOLOE-26** |

### 5.9 何时选哪个版本（决策表）

| 场景 | 推荐 | 原因 |
|------|------|------|
| 遗留 yolov5 产线 / TensorRT7 | **YOLOv5** | 生态最老、资料最多、anchor 流程成熟 |
| 已有 v8 大量微调权重 | **YOLOv8** | 换 v11/v26 需重训 |
| GPU 服务器 + 要最高 mAP/参数量比 | **YOLO11** | mAP 已很高，TRT 延迟与 v26 接近 |
| **CPU / 嵌入式 / 无 NMS 运行时** | **YOLO26** | e2e 省 NMS，CPU +43%，导出更简单 |
| 工业相机固定 pipeline、怕 NMS 不一致 | **YOLO26** | 输出 (300,6) 确定性更强 |
| 遮挡场景姿态估计 | **YOLO26-pose** | RLE 不确定性建模 |
| 开放词汇 + 边缘 | **YOLOE-26** | NMS-free + prompt |
| 需要 o2m 略高 0.5 mAP | v8/v11 或 `end2end=False` | NMS 路径精度略优 |
| RKNN / NCNN 移动端 | v11 或 v26 `end2end=False` | e2e TopK 可能不支持 |

### 5.10 四版本「一张表记住差异」

```
          v5          v8          v11         v26
Anchor    有          无          无          无
Block     C3          C2f         C3k2        C3k2
Attention 无          无          C2PSA       C2PSA
Cls头     耦合        Conv        DWConv      DWConv
Reg       4值         DFL×16      DFL×16      4值
NMS       必须        必须        必须        默认免
输出      25k点       8400 raw    8400 raw    300框
CPU       慢          中          中          最快
mAP(n)    28.0        37.3        39.5        40.9
```

---

## 6. 与 YOLO11 的专项差异（精读）

YOLO26 可视为 **「v11 骨架 + v10 式 e2e 头 + 训练/损失升级」**，而非全新 backbone 革命。

| 升级项 | 保留自 v11 | YOLO26 新增/修改 |
|--------|-----------|-----------------|
| C3k2 | ✅ | neck 层 c3k 更激进 |
| C2PSA | ✅ | 同结构 |
| DWConv cv3 | ✅ | 同 legacy=False |
| DFL | ✅ reg_max=16 | ❌ **reg_max=1** |
| NMS | ✅ 推理必须 | ❌ **默认内置 TopK** |
| 优化器 | SGD | **MuSGD** |
| 小目标 | 标准 TAL | **+ STAL** |
| SPPF | 标准 | **多核变体** |

**CPU 延迟对比（官方 ONNX，n 档）：** 11n **56.1 ms** → 26n **38.9 ms**（约 **31%** 提升；相对 v11 官方口径最高 **43%** 来自更广 benchmark 集合）。

---

## 7. 精度优化与实验建议

- **数据**：小目标多 → 优先 v26 + STAL；可试 `yolo26-p2.yaml`（P2 小目标头，需自训）
- **训练**：COCO 预训练 `yolo26*.pt`；长 schedule（300 epoch）配合 MuSGD
- **验证**：报告 mAP 时注明 **e2e=True/False**（差 ~0.5–0.8）
- **部署**：CPU 量产后端用 `end2end=True` 导出 ONNX；RKNN 等用 `end2end=False`
- **Pose**：遮挡数据集优先 `yolo26*-pose` + RLE（见 [YOLOv8 §20](./YOLOv8.md#20-pose26-与-rle-lossyolo26-姿态分支)）

---

## 8. 相关论文与参考

| 资源 | 关联 |
|------|------|
| [Ultralytics YOLO26 文档](https://docs.ultralytics.com/models/yolo26/) | 官方 benchmark / API |
| [End-to-End Detection 指南](https://docs.ultralytics.com/guides/end2end-detection/) | 双头 / 导出 / 格式兼容 |
| [YOLOv10](https://arxiv.org/abs/2405.14458) | NMS-free 端到端思想先驱 |
| [RLE Pose (ICCV 2021)](https://arxiv.org/abs/2107.11291) | Pose26 损失 |
| [YOLOv5 本地笔记](./YOLOv5.md) | Anchor 时代基线 |
| [YOLOv8 本地笔记](./YOLOv8.md) | TAL / DFL / Loss / 部署 |
| [YOLO11 本地笔记](./YOLOv11.md) | C3k2 / C2PSA / 轻量化 |

### 源码索引（ultralytics）

| 路径 | 内容 |
|------|------|
| `cfg/models/26/yolo26.yaml` | 检测结构 + end2end + reg_max |
| `cfg/models/26/yolo26-pose.yaml` | Pose26 |
| `nn/modules/head.py` | Detect 双头 / Pose26 |
| `utils/loss.py` | E2ELoss / PoseLoss26 / RLELoss |
| `cfg/default.yaml` | MuSGD 等训练超参 |

---

## 附录 A. 文档覆盖清单

**已覆盖：**

- [x] YOLO26 概览与五大卖点
- [x] NMS-free / 双头 / DFL 移除 / SPPF 变体
- [x] MuSGD / ProgLoss / STAL 训练创新
- [x] **YOLOv5 / v8 / v11 / v26 全维度对比（§5）**
- [x] yaml 与 v11 对照
- [x] 扩展任务（sem / Pose26 / YOLOE-26）
- [x] 训练 / 推理 / 导出 / 格式兼容
- [x] 版本选型决策表

**可继续深入：**

- [ ] E2ELoss 源码逐行与 o2m/o2o 权重 schedule 数值
- [ ] STAL 相对 TAL 的小目标分配公式
- [ ] yolo26-p2 / p6 自定义 yaml 实验模板
- [ ] v11→v26 权重部分迁移可行性（层名对照）
