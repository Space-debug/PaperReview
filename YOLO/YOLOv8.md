# YOLOv8

> 本文档用于深入学习 YOLOv8：**核心网络结构**、**训练/推理全流程**、以及与 YOLOv5 的差异。  
> 代码：[ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

---

## 1. 基线：YOLOv8 概览

### 1.1 基本信息

| 项目 | 内容 |
|------|------|
| 作者/机构 | Ultralytics |
| 发表时间 | 2023 年 1 月（持续迭代 v8.1 / v8.2 / v8.3） |
| 任务 | 检测 / 分割 / 姿态 / 分类 / 旋转框（统一框架） |
| 代码 | `pip install ultralytics` |
| 论文 | **无正式论文**，工程实现；思想来自 YOLOX、TOOD、GFL 等 |
| 相对 v5 | **Anchor-free + 解耦头 + DFL + TaskAlignedAssigner** |

### 1.2 版本演进（v8.0 → v8.3）

| 版本 | 主要变化 |
|------|----------|
| **v8.0** | 首发：C2f + 解耦头 + TAL + DFL，统一 ultralytics 包 |
| **v8.1** | 新增 OBB（旋转框）；World 模型（开放词汇检测） |
| **v8.2** | C3k2 模块；更大模型 yaml 调整 |
| **v8.3** | 训练/导出优化；与 YOLO11 过渡 |

### 1.3 整体架构

```
Input (B, 3, 640, 640)
    │
    ▼
Backbone: CSPDarknet（C2f）→ P3 / P4 / P5
    │
    ▼
Neck: PAN-FPN（C2f + Upsample + Concat）
    │
    ▼
Head: Detect（Decoupled, Anchor-free）
    │   Reg 分支: 4×reg_max 通道（DFL 分布）
    │   Cls 分支: nc 通道
    ▼
Output（3 尺度）:
  P3: (B, 64+nc, 80, 80)    stride=8   小目标
  P4: (B, 64+nc, 40, 40)    stride=16  中目标
  P5: (B, 64+nc, 20, 20)    stride=32  大目标

  reg_max=16 → 64=4边×16档；无 obj 分支
```

**与 YOLOv5 输出对比：**

```
YOLOv5:  (B, 3×(5+nc), H, W)   每 cell 3 anchor × (tx,ty,tw,th,obj,cls)
YOLOv8:  (B, 64+nc, H, W)      每 cell 1 点 × (DFL四边分布, cls)
预测点总数: 25,200 → 8,400
```

### 1.4 官方精度（COCO val2017，640）

| 模型 | mAP@0.5 | mAP@0.5:0.95 | 参数量 | FLOPs | vs YOLOv5 同档 |
|------|---------|--------------|--------|-------|----------------|
| YOLOv8n | 52.0 | 37.3 | 3.2M | 8.7G | v5n +9.3 |
| YOLOv8s | 61.0 | 44.9 | 11.2M | 28.6G | v5s +7.5 |
| YOLOv8m | 67.2 | 50.2 | 25.9M | 78.9G | v5m +4.8 |
| YOLOv8l | 69.8 | 52.9 | 43.7M | 165.2G | v5l +3.9 |
| YOLOv8x | 71.1 | 53.9 | 68.2M | 257.8G | v5x +3.2 |

### 1.5 相对 YOLOv5 的核心变革

| 维度 | YOLOv5 | YOLOv8 | 好处 |
|------|--------|--------|------|
| Anchor | 9 个 K-Means anchor | **无** | 消除 anchor 超参 |
| Head | 1×1 耦合 | **解耦双分支** | 分类/定位任务解耦 |
| 置信度 | obj × cls | **cls max** | 结构更简单 |
| 框回归 | CIoU 直接回归 4 值 | **CIoU + DFL 分布** | 高 IoU 定位更准 |
| 标签分配 | build_targets | **TaskAlignedAssigner** | 分类与定位对齐 |
| Backbone | C3 | **C2f** | 更丰富梯度流 |
| 框架 | yolov5 独立仓库 | **ultralytics 统一包** | detect/seg/pose 一套 API |

---

## 2. 核心模块详解

### 2.1 Conv：Conv → BN → SiLU

与 YOLOv5 相同，`bias=False`，自动 padding。

### 2.2 C2f（替代 C3）

```
Input (C1)
  │
  Conv 1×1 → 2×C_hidden
  │
  Split → [part1 | part2]
  │
  part1 → Bottleneck → y1 → Bottleneck → y2 → ... → yn
  │
  Concat [part2, y1, y2, ..., yn]    ← 所有中间特征都保留
  │
  Conv 1×1 → C_out
```

| | C3 (v5) | C2f (v8) |
|---|---------|----------|
| 通路数 | 2 路 | **n+1 路** |
| 特征复用 | 仅 Bottleneck 链最终输出 | 每个 Bottleneck 输出都 concat |
| 效果 | 基线 | 梯度流更丰富，+0.5~1 mAP |

### 2.3 Bottleneck / SPPF

与 YOLOv5 相同：1×1→3×3 残差瓶颈；SPPF 串行 MaxPool 扩大感受野。

### 2.4 C2f 变体（v8.1+）

| 变体 | 说明 |
|------|------|
| C2f | 默认，内部标准 Bottleneck |
| C2fAtt | 带注意力 Bottleneck |
| C3k2 | Bottleneck 换 C3k（3×3 核堆叠），大模型用 |

---

## 3. Backbone 与 Neck

### 3.1 Backbone（YOLOv8s）

```
层   操作                 输出尺寸        说明
 0   Conv 3×3 s=2         320×320×64
 1   Conv 3×3 s=2         160×160×128
 2   C2f ×3               160×160×128
 3   Conv 3×3 s=2          80×80×256     → P3 源
 4   C2f ×6                80×80×256
 5   Conv 3×3 s=2          40×40×512     → P4 源
 6   C2f ×6                40×40×512
 7   Conv 3×3 s=2          20×20×1024    → P5 源
 8   C2f ×3                20×20×1024
 9   SPPF                  20×20×1024
```

### 3.2 Neck（PAN-FPN）

```
FPN: P5 → Conv → Upsample → Concat P4 → C2f → Upsample → Concat P3 → C2f → N3
PAN: N3 → Conv↓ → Concat → C2f → N4 → Conv↓ → Concat → C2f → N5
```

检测尺度：P3 (stride=8) / P4 (16) / P5 (32)。

### 3.3 模型缩放与 yaml

```yaml
# cfg/models/v8/yolov8.yaml 核心字段
depth_multiple: 0.33    # 控制 C2f 重复次数
width_multiple: 0.50    # 控制通道数
nc: 80                  # 类别数（数据集覆盖）

# reg_max 在 head 中默认 16
```

| 型号 | depth | width |
|------|-------|-------|
| n | 0.33 | 0.25 |
| s | 0.33 | 0.50 |
| m | 0.67 | 0.75 |
| l | 1.00 | 1.00 |
| x | 1.00 | 1.25 |

---

## 4. Head：Decoupled Detect（Anchor-free）

### 4.1 解耦头结构

```
输入 (B, C, H, W)
  │
  ├─ cv2（Reg 分支）:
  │    Conv 3×3 → Conv 3×3 → Conv 1×1 → (B, 4×reg_max, H, W)
  │
  └─ cv3（Cls 分支）:
       Conv 3×3 → Conv 3×3 → Conv 1×1 → (B, nc, H, W)
```

**解耦 vs YOLOv5 耦合头：**

- v5：1×1 同时输出 reg+cls+obj，梯度互相干扰
- v8：两分支独立 3×3 堆叠，分类学语义、回归学边界

### 4.2 Anchor-free：以 grid 中心为锚点

每个 cell 生成一个**参考点**（cell 中心），预测到框四边的距离：

```
         dist_t
            ↑
dist_l ←  ★ (cx,cy)  → dist_r
            ↓
         dist_b

x1 = cx - dist_l × stride    （训练时在 feature map 空间，不含 stride）
y1 = cy - dist_t × stride
x2 = cx + dist_r × stride
y2 = cy + dist_b × stride
```

**为何取消 anchor？**

| Anchor-based (v5) | Anchor-free (v8) |
|-------------------|------------------|
| 需 K-Means 聚类 9 个 anchor | 无 anchor 超参 |
| 每 cell 3 预测，NMS 前 25,200 候选 | 每 cell 1 预测，8,400 候选 |
| 宽高用 anchor 先验 + 偏移 | 直接用四边距离，更直观 |

### 4.3 DFL（Distribution Focal Loss）—— 详解

#### 4.3.1 核心思想

不回归单个距离值，而是预测 **reg_max=16 档的离散分布**，再取期望作为最终距离。

```
单边距离 d 的建模:
  网络输出 16 个 logit: [z₀, z₁, ..., z₁₅]
  概率:               pᵢ = softmax(z)ᵢ
  解码距离:            d = Σ(i × pᵢ),  i = 0..15

4 边 × 16 档 = 64 通道（reg 分支输出）
```

#### 4.3.2 训练：DFL Loss（软标签交叉熵）

GT 框转为四边距离 target（feature map 尺度），设某边 target = 5.7：

```
left_bin  = floor(5.7) = 5
right_bin = 6
weight_left  = 6 - 5.7 = 0.3
weight_right = 5.7 - 5 = 0.7

Loss = BCE(z₅, 0.3) + BCE(z₆, 0.7)    （左右 bin 软标签）
```

**好处**：比 one-hot 更平滑；比直接 MSE 回归在 **AP75/AP90** 上梯度更强。

#### 4.3.3 推理：Integral 层解码

```python
# 固定权重 [0,1,2,...,15]，对 softmax 后的分布做加权求和
d = Σ i · softmax(zᵢ)     # 等价于分布期望值
```

源码 `nn/modules/block.py` 中 `Integral` 模块实现，export 时会融合进图。

#### 4.3.4 为何 reg_max=16？

| reg_max | 效果 |
|---------|------|
| 太小 (8) | 距离量化粗糙，定位精度不足 |
| 16 | 默认，精度与速度平衡 |
| 太大 (32) | 通道数翻倍，收益递减 |

距离单位是 **grid 尺度**（不含 stride），再经 dist2bbox 乘 stride 映射像素。

### 4.4 去掉 Objectness

```
YOLOv5:  conf = σ(obj) × σ(cls)
YOLOv8:  conf = σ(cls_max)       无 obj 分支
```

**为何可去掉？**

1. TAL 保证正样本 cls 与 IoU 对齐，cls 分支已承担"有无物体"判别
2. Anchor-free 每 cell 仅 1 预测，无 multi-anchor 歧义
3. 负样本 cls target=0，背景抑制由 cls BCE 完成

### 4.5 参考点生成（make_anchors）

```python
# 每个尺度生成 grid 中心点（feature map 坐标，0.5 偏移对齐 pixel center）
for h, w, stride in zip(feats_h, feats_w, strides):
    sx = torch.arange(w) + 0.5
    sy = torch.arange(h) + 0.5
    sy, sx = torch.meshgrid(sy, sx)
    anchor_points.append(torch.stack([sx, sy], -1) * stride)  # 可选乘 stride
```

P3 共 80×80=6400 点，三尺度合计 **8400 个参考点**。

### 4.6 Detect 层 Bias 初始化

```python
# cls 分支最后一层 bias
b = log(5 / nc / (640/stride)²)    # 按 stride 和类别数调整先验

# reg 分支: bias=1.0（默认）
```

使训练初期 cls 预测接近低置信，避免 FP 爆炸（思想同 v5 的 obj bias 初始化）。

### 4.7 推理解码完整流程

```
1. 合并三尺度 reg (B,64,ΣHW) 和 cls (B,nc,ΣHW)
2. reg → reshape (B, 4, 16, N) → softmax(dim=2) → Integral → dist (B,4,N)
3. dist2bbox(dist, anchor_points) → xyxy（像素坐标）
4. conf = cls.sigmoid().max(-1),  cls_id = argmax
5. conf > conf_thres 过滤
6. per-class NMS（默认 iou=0.7）
7. 坐标从 letterbox 空间映射回原图
```

**dist2bbox / bbox2dist**（`utils/tal.py`）：

```python
def dist2bbox(dist, anchor_points, xywh=False):
    lt, rb = dist.chunk(2, dim)           # 左上下右
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    return cat([x1y1, x2y2], dim)          # xyxy

def bbox2dist(anchor, bbox, reg_max):
    # GT 框 → 四边距离 target（训练用，clamp 到 [0, reg_max-1]）
    x1y1, x2y2 = bbox.chunk(2, -1)
    return cat([anchor - x1y1, x2y2 - anchor], -1)
```

---

## 5. 标签分配：TaskAlignedAssigner（TAL）

YOLOv8 最核心的训练机制之一，源码 `utils/tal.py`。

### 5.1 设计动机

| YOLOv5 build_targets | YOLOv8 TAL |
|---------------------|------------|
| 宽高比匹配 anchor + 中心 cell | **预测质量驱动** |
| 正样本可能与 cls 预测差很远 | 要求 cls 高且 IoU 高 |
| 固定规则 | Top-K 动态选择 |

思想来源：**TOOD**（Task-aligned One-stage Object Detection）。

### 5.2 对齐度量（Alignment Metric）

```
align_metric = (cls_score)^α × (IoU)^β

v8DetectionLoss 实例化: α=0.5, β=6.0, topk=10
```

- `cls_score`：预测点对 **GT 所属类别** 的 sigmoid 得分（传入 assigner 前已 sigmoid）
- `IoU`：预测框与 GT 的 **CIoU**（`iou_calculation` 用 `bbox_iou(..., CIoU=True)`，训练时用 **detach 的预测框**）

**β=6 的含义**：IoU 占绝对主导，IoU 从 0.5→0.4，align 约降 44%（0.4^6 vs 0.5^6）→ 优先选**定位准**的点。  
**α=0.5**：cls 用平方根缓和，避免 cls 极低时 align 恒为 0 导致无法匹配。

### 5.3 完整分配流程

```
输入（每个 batch）:
  pd_scores: (B, N, nc)     预测 cls logits，N=8400
  pd_bboxes: (B, N, 4)     预测框 xyxy（decode 后，detach）
  anc_points: (N, 2)       参考点
  gt_labels, gt_bboxes     GT

Step 1: 中心点必须在 GT 框内
  mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)
  → 参考点 (cx,cy) 落在 GT 矩形内的才候选
  → 比 v5 "仅看 GT 中心所在 cell" 更灵活

Step 2: 计算 IoU 矩阵
  overlaps: (B, n_gt, N)   每个 GT 与每个预测点的 IoU

Step 3: 取 GT 类别的 cls score
  bbox_scores = pd_scores[gt_class]   (B, n_gt, N)

Step 4: 对齐度量
  align_metric = bbox_scores^α × overlaps^β

Step 5: Top-K 选择
  每个 GT 选 align_metric 最高的 K=10 个预测点 → 正样本
  → 10 个 GT 最多 100 个正样本（去重后更少）

Step 6: 冲突消解
  若一点匹配多个 GT → 分配给 IoU 最高的 GT

Step 7: 生成 target
  正样本: target_bbox=GT, target_score=归一化 align_metric（软标签）
  负样本: target_score=0
```

### 5.4 软标签 target_score（重要细节）

v8 的 cls target **不是硬 0/1**：

```
正样本 target_score = normalize(align_metric)   范围 (0, 1]
负样本 target_score = 0
```

**好处**：质量更高的正样本 cls 目标 closer to 1，质量一般的正样本目标 lower → **Quality Focal 思想**，减少低质量正样本的有害梯度。

### 5.5 与 YOLOv5 正负样本对比

| | YOLOv5 | YOLOv8 |
|---|--------|--------|
| 正样本判定 | 宽高比+中心 cell+相邻 cell | Top-K align_metric + 点在 GT 内 |
| 正样本数 | ~3~9 / GT / 尺度，不固定 | **最多 K=10 / GT** |
| cls target | 硬 one-hot 1.0 | **软标签** align 归一化 |
| box target | GT 偏移 | GT 框 → bbox2dist → DFL target |
| 负样本 cls | 不参与 loss | **参与 BCE，target=0** |
| obj loss | 有 | **无** |

### 5.6 为何 TAL 需要 detach 预测框？

分配阶段用 `pd_bboxes.detach()`：标签分配是**离散选择**（Top-K），不可导。detach 避免分配逻辑对梯度的干扰，只让 loss 回传优化网络。

### 5.7 源码逐行解读：`utils/tal.py`

以下基于当前 `ultralytics` 包中 `TaskAlignedAssigner`，与训练时 `v8DetectionLoss` 的调用链一致。

#### 5.7.1 类初始化

```python
class TaskAlignedAssigner(nn.Module):
    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0,
                 stride=[8,16,32], eps=1e-9, topk2=None):
        self.topk = topk          # v8DetectionLoss 传入 tal_topk=10
        self.topk2 = topk2 or topk
        self.alpha = alpha        # v8DetectionLoss 传入 0.5
        self.beta = beta          # 6.0
        self.stride_val = stride[1]  # 小 GT 框宽高下限用 16
```

| 参数 | v8 实际值 | 作用 |
|------|----------|------|
| topk | **10** | 每个 GT 最多选 10 个候选点 |
| alpha | **0.5** | cls 在 align 中的指数 |
| beta | **6.0** | IoU 在 align 中的指数 |
| eps | 1e-9 | 归一化防除零 |

#### 5.7.2 `forward` → `_forward` 主流程

```python
@torch.no_grad()   # 整个分配过程不参与梯度
def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
```

**输入张量形状：**

| 变量 | 形状 | 含义 |
|------|------|------|
| pd_scores | (B, N, nc) | 预测 cls，**已 sigmoid** |
| pd_bboxes | (B, N, 4) | 预测框 xyxy，**像素尺度**，detach |
| anc_points | (N, 2) | 参考点，**像素尺度** |
| gt_labels | (B, n_max, 1) | GT 类别 id |
| gt_bboxes | (B, n_max, 4) | GT xyxy，**像素尺度** |
| mask_gt | (B, n_max, 1) | 有效 GT 掩码（非 padding） |

**`_forward` 四步：**

```python
# Step A: 得到候选正样本 mask
mask_pos, align_metric, overlaps = self.get_pos_mask(...)

# Step B: 一点多 GT 冲突消解 + 可选 topk2 二次筛选
target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(...)

# Step C: 查表得到每个 anchor 的 GT 类别、框、one-hot cls
target_labels, target_bboxes, target_scores = self.get_targets(...)

# Step D: 软标签归一化（核心！）
align_metric *= mask_pos
pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)   # 每个 GT 的最大 align
pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + eps)).amax(-2).unsqueeze(-1)
target_scores = target_scores * norm_align_metric   # one-hot × 软权重
```

**Step D 数值含义：**

```
对每个正样本 anchor:
  1. 取该 GT 所有候选中 align 最大值 pos_align_metrics
  2. 用 IoU 最大值 pos_overlaps 做缩放
  3. norm = align × pos_overlaps / pos_align_metrics
  4. target_scores[c] = 1.0 × norm  →  软标签，非硬 1.0

质量最高的正样本 norm → 1.0，边缘正样本 norm → 0.3~0.7
```

**返回值：**

| 返回 | 形状 | 用途 |
|------|------|------|
| target_labels | (B, N) | 每个 anchor 分配的 GT 类别 |
| target_bboxes | (B, N, 4) | 回归目标 xyxy（像素） |
| target_scores | (B, N, nc) | cls BCE 的软 target |
| fg_mask | (B, N) | 正样本 bool 掩码 |
| target_gt_idx | (B, N) | anchor 对应第几个 GT |

#### 5.7.3 `get_pos_mask`：三重 mask 合并

```python
def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
    mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)
    align_metric, overlaps = self.get_box_metrics(
        pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
    mask_topk = self.select_topk_candidates(align_metric, topk_mask=...)
    mask_pos = mask_topk * mask_in_gts * mask_gt
    return mask_pos, align_metric, overlaps
```

```
mask_pos = mask_topk  AND  mask_in_gts  AND  mask_gt
           Top-K筛选      点在GT框内      有效GT
```

#### 5.7.4 `select_candidates_in_gts`：参考点须在 GT 内

```python
# gt_bboxes: (B, n_gt, 4) xyxy
# xy_centers: (N, 2) 参考点
lt, rb = gt_bboxes.view(-1,1,4).chunk(2, 2)   # 左上、右下
bbox_deltas = cat([xy_centers - lt, rb - xy_centers], dim=2)  # 到四边的距离
return bbox_deltas.amin(3).gt_(eps)   # 四边距离都 > 0 → 点在框内
```

**小 GT 框特殊处理：** 若 GT 宽或高 < stride[0]=8，强制宽高为 stride_val=16，避免极小 GT 匹配不到任何点。

#### 5.7.5 `get_box_metrics`：align 与 IoU

```python
# 取每个 GT 对应类别的 cls 分数
ind[0] = batch_idx          # (B, n_gt)
ind[1] = gt_labels.squeeze(-1)
bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]   # (B, n_gt, N)

# CIoU（非普通 IoU）
overlaps[mask_gt] = bbox_iou(gt_boxes, pd_boxes, CIoU=True).clamp_(0)

align_metric = bbox_scores.pow(alpha) * overlaps.pow(beta)
```

**为何用 CIoU 而非 IoU？** 分配阶段也考虑中心距离和宽高比，与 box loss 一致。

#### 5.7.6 `select_topk_candidates`：Top-K 实现

```python
topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1)  # 每 GT 取 topk 个 anchor 索引
count_tensor = zeros(B, n_gt, N)
for k in range(self.topk):
    count_tensor.scatter_add_(-1, topk_idxs[:,:,k:k+1], ones)  # 标记被选中的 anchor
count_tensor.masked_fill_(count_tensor > 1, 0)   # 重复选中置 0（去重）
return count_tensor   # 1.0=选中, 0.0=未选
```

#### 5.7.7 `select_highest_overlaps`：一点匹配多 GT

```python
fg_mask = mask_pos.sum(-2)   # (B, N) 每个 anchor 被几个 GT 选中

if fg_mask.max() > 1:   # 存在冲突
    max_overlaps_idx = overlaps.argmax(1)   # 每个 anchor 选 IoU 最大的 GT
    # 只保留 IoU 最大的那个 GT 的匹配，其余清零
    mask_pos = where(multi_gts, is_max_overlaps, mask_pos)

target_gt_idx = mask_pos.argmax(-2)   # (B, N) 每个 anchor 最终归属哪个 GT
```

#### 5.7.8 辅助函数 `make_anchors` / `dist2bbox` / `bbox2dist`

```python
# make_anchors: 生成 cell 中心 + stride 张量
sx = arange(w) + 0.5    # pixel center 对齐
sy = arange(h) + 0.5
anchor_points: (N, 2)   feature map 尺度（未乘 stride 或按调用方处理）
stride_tensor: (N, 1)   每个点对应 stride

# dist2bbox
x1y1 = anchor_points - lt
x2y2 = anchor_points + rb
return cat([x1y1, x2y2])   # xyxy

# bbox2dist（训练 DFL target）
dist = cat([anchor - x1y1, x2y2 - anchor])
dist = dist.clamp(0, reg_max - 0.01)
```

---

## 6. 损失函数

### 6.1 总损失

```
L = λ_cls · L_cls + λ_box · L_box + λ_dfl · L_dfl

默认: λ_cls=0.5, λ_box=7.5, λ_dfl=1.5
```

v5 对比：`L = λ_box·L_box + λ_obj·L_obj + λ_cls·L_cls`，v8 **无 obj，新增 dfl**。

### 6.2 L_cls：BCE with 软标签

```
L_cls = BCEWithLogits(pred_cls, target_score)

target_score: 正样本=归一化 align_metric，负样本=0
所有 8400×nc 个位置都参与（稀疏正样本，大量负样本）
```

**与 v5 差异**：v5 负样本不参与 cls loss；v8 **全参与**，背景分类监督更充分。

### 6.3 L_box：CIoU Loss

```
L_box = (1 - CIoU) × weight    仅 fg_mask 正样本

CIoU = IoU - ρ²/c² - α·v
```

pred 框来自 DFL 解码 + dist2bbox，与 GT xyxy 算 CIoU。

### 6.4 L_dfl：分布交叉熵

```
对每个正样本的每条边:
  target_dist = bbox2dist(anchor, gt_bbox)
  对 target 相邻两 bin 做软标签 BCE（见 §4.3.2）
```

三路 loss 都**仅正样本**算 box/dfl；cls 正负都算。

### 6.5 Loss 权重与归一化

| 参数 | 值 | 说明 |
|------|-----|------|
| box | 7.5 | 看似很大，loss 内部按正样本数归一化 |
| cls | 0.5 | |
| dfl | 1.5 | |

```python
# 典型归一化: loss × batch_size / 正样本总数
# 保证不同 batch 正样本数量变化时梯度量级稳定
```

### 6.6 源码逐行解读：`utils/loss.py`

训练时 `DetectionModel` 返回 `v8DetectionLoss(self)`，每个 batch 调用 `loss(preds, batch)`。

#### 6.6.1 调用链总览

```
trainer 前向
  → model(batch)  输出 preds dict: {boxes, scores, feats}
  → v8DetectionLoss.__call__(preds, batch)
       → get_assigned_targets_and_loss(preds, batch)
            ① preprocess GT
            ② bbox_decode 预测框
            ③ TaskAlignedAssigner 分配
            ④ BCE cls loss
            ⑤ BboxLoss (CIoU + DFL)
            ⑥ × hyp gain
       → return loss * batch_size
```

#### 6.6.2 `v8DetectionLoss.__init__`

```python
def __init__(self, model, tal_topk=10, tal_topk2=None):
    m = model.model[-1]              # Detect 头
    self.nc = m.nc
    self.reg_max = m.reg_max         # 默认 16
    self.stride = m.stride           # [8, 16, 32]
    self.bce = BCEWithLogitsLoss(reduction="none")
    self.assigner = TaskAlignedAssigner(
        topk=tal_topk,               # 10
        num_classes=self.nc,
        alpha=0.5, beta=6.0,
        stride=self.stride.tolist(),
        topk2=tal_topk2,
    )
    self.bbox_loss = BboxLoss(m.reg_max)
    self.proj = arange(m.reg_max)    # [0,1,...,15] 用于 DFL 积分解码
```

#### 6.6.3 `preprocess`：GT 格式整理

```python
# 输入 targets: (n_labels, 6) = [batch_idx, cls, cx, cy, w, h]  归一化 xywh
# 输出: (B, n_max_gt, 5) = [cls, x1, y1, x2, y2]  像素 xyxy

out[batch_idx, within_idx] = targets[:, 1:]
out[..., 1:5] = xywh2xyxy(out[..., 1:5] * scale_tensor)  # scale_tensor = [W,H,W,H]
```

同一 batch 内 GT 按图片索引填入二维张量，padding 位置 mask_gt=0。

#### 6.6.4 `bbox_decode`：DFL 积分 + dist2bbox

```python
def bbox_decode(self, anchor_points, pred_dist):
    # pred_dist: (B, N, 64)  raw logits
    pred_dist = pred_dist.view(b, a, 4, 16).softmax(3)          # 每边 16 档 softmax
    pred_dist = pred_dist.matmul(self.proj)                        # 加权求和 → 期望距离
    return dist2bbox(pred_dist, anchor_points, xywh=False)       # (B, N, 4) xyxy
```

**与推理一致：** 训练解码与推理使用相同的 softmax + proj 积分，保证 assign 用的框与 loss 用的框一致。

#### 6.6.5 `get_assigned_targets_and_loss` 核心

```python
# 1. 整理预测
pred_distri = preds["boxes"].permute(0, 2, 1)    # (B, N, 64)
pred_scores = preds["scores"].permute(0, 2, 1)  # (B, N, nc)
anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

# 2. 整理 GT
targets = cat([batch_idx, cls, bboxes])
targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1,0,1,0]])
gt_labels, gt_bboxes = targets.split((1, 4), 2)
mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

# 3. 解码预测框（feature map 尺度）
pred_bboxes = self.bbox_decode(anchor_points, pred_distri)   # xyxy, 未乘 stride

# 4. TAL 分配（注意尺度对齐！）
_, target_bboxes, target_scores, fg_mask, _ = self.assigner(
    pred_scores.detach().sigmoid(),                    # cls 概率
    (pred_bboxes.detach() * stride_tensor).type(...),  # 预测框 → 像素
    anchor_points * stride_tensor,                     # 参考点 → 像素
    gt_labels,
    gt_bboxes,                                         # GT 已是像素
    mask_gt,
)

target_scores_sum = max(target_scores.sum(), 1)        # 软标签总和，作归一化分母
```

**尺度对齐要点：**

| 变量 | 尺度 |
|------|------|
| pred_bboxes（decode 后） | feature map（÷ stride = 像素） |
| 传入 assigner 的 pd_bboxes | **× stride_tensor → 像素** |
| anchor_points 传入 assigner | **× stride_tensor → 像素** |
| target_bboxes 输出 | 像素 |
| 传入 BboxLoss 的 target_bboxes | **÷ stride_tensor → feature map** |

#### 6.6.6 Cls Loss

```python
bce_loss = self.bce(pred_scores, target_scores.to(dtype))  # logits vs 软标签
loss[1] = bce_loss.sum() / target_scores_sum
loss[1] *= self.hyp.cls   # × 0.5
```

- **所有 N×nc 个位置都参与**（负样本 target_scores=0）
- 分母 `target_scores_sum` 是所有软标签之和，非固定 N

#### 6.6.7 `BboxLoss.forward`：CIoU + DFL

```python
weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)   # 正样本软权重

# CIoU
iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], CIoU=True)
loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

# DFL
target_ltrb = bbox2dist(anchor_points, target_bboxes, reg_max - 1)
loss_dfl = self.dfl_loss(
    pred_dist[fg_mask].view(-1, 16),    # (n_pos*4, 16) 每条边 16 logits
    target_ltrb[fg_mask],                 # (n_pos, 4) 连续距离 target
) * weight
loss_dfl = loss_dfl.sum() / target_scores_sum
```

```python
loss[0] *= self.hyp.box   # × 7.5
loss[2] *= self.hyp.dfl   # × 1.5
return loss * batch_size  # 最终返回标量 × batch_size
```

#### 6.6.8 `DFLoss.__call__` 逐行

```python
def __call__(self, pred_dist, target):
    # pred_dist: (n, 16)  单边 logits
    # target:    (n_pos, 4) 或 broadcast 到每条边

    target = target.clamp_(0, reg_max - 1 - 0.01)   # [0, 14.99]
    tl = target.long()           # 左 bin  index
    tr = tl + 1                  # 右 bin  index
    wl = tr - target             # 左权重（小数部分给右 bin）
    wr = 1 - wl                  # 右权重

    return (
        CE(pred_dist, tl.view(-1)) * wl
      + CE(pred_dist, tr.view(-1)) * wr
    ).mean(-1, keepdim=True)
```

**数值例：** target=5.7 → tl=5, tr=6, wl=0.3, wr=0.7

```
Loss = 0.3 × CE(z, bin5) + 0.7 × CE(z, bin6)
```

等价于把 5.7 这个连续值用相邻两 bin 的软 one-hot 表示，比硬分类 bin5 梯度更平滑。

#### 6.6.9 完整数据流图（单 batch）

```
                    preds["feats"] P3,P4,P5
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
        pred_scores (B,N,nc)     pred_distri (B,N,64)
              │                         │
              │                  bbox_decode + make_anchors
              │                         │
              └────────┬────────────────┘
                       ▼
              TaskAlignedAssigner (@no_grad)
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
   target_scores  target_bboxes   fg_mask
   (软标签)        (像素xyxy)      (正样本)
         │             │             │
         ▼             └──────┬──────┘
    BCE cls loss              ▼
                        BboxLoss
                     CIoU + DFL
         │             │
         └──────┬──────┘
                ▼
    loss = (box×7.5 + cls×0.5 + dfl×1.5) × batch_size
                ▼
             backward
```

#### 6.6.10 与 YOLOv5 `ComputeLoss` 对照

| 步骤 | YOLOv5 `loss.py` | YOLOv8 `loss.py` |
|------|------------------|------------------|
| 分配 | `build_targets()` 宽高比+cell | `TaskAlignedAssigner` |
| 正样本 cls | 硬 1.0 | 软 norm_align |
| 负样本 cls | 不参与 | BCE target=0 |
| obj loss | BCE obj | **无** |
| box loss | CIoU | CIoU × 软权重 |
| 回归 | 直接 tx,ty,tw,th | DFL 分布 |
| 解码 | anchor+sigmoid | DFL 积分+dist2bbox |
| 梯度 | 分配可微（规则固定） | 分配 `@no_grad` detach |

---

## 7. 训练全流程

### 7.1 数据管道

```
原始图片 + YOLO 标签 (cls cx cy w h)
    │
    ▼
标签校验（越界、EXIF 旋转、损坏图片剔除）
    │
    ▼
Letterbox / 增强（Mosaic 等）
    │
    ▼
归一化 /255 → Tensor (B,3,H,W)
    │
    ▼
前向 → TAL 分配 → 计算 loss → 反向
```

#### data.yaml

```yaml
path: /data/custom
train: images/train
val: images/val
nc: 3
names: ['a', 'b', 'c']
```

### 7.2 数据增强详解

| 增强 | 参数 | 默认 | 说明 |
|------|------|------|------|
| Mosaic | `mosaic` | 1.0 | 4 图 2×2 拼接，同 v5 |
| MixUp | `mixup` | **0.0** | v8 默认关闭（v5 默认开） |
| HSV | `hsv_h/s/v` | 0.015/0.7/0.4 | 色调/饱和度/明度 |
| 翻转 | `fliplr` | 0.5 | 水平翻转 |
| 平移 | `translate` | 0.1 | ±10% |
| 缩放 | `scale` | 0.5 | 0.5~1.5× |
| 透视 | `perspective` | 0.0 | 默认关 |
| close_mosaic | `close_mosaic` | 10 | 最后 10 epoch 关 Mosaic |

**Mosaic 关闭原因**：末期在正常单图分布微调，减少 train-test gap（同 v5）。

### 7.3 训练循环

```
for epoch:
  1. LR: Warmup(3 epoch) → Cosine 退火 lr0 → lr0×lrf
  2. 可选 multi_scale: 每 batch 随机 imgsz ±50%
  3. 前向 + TAL assign + v8DetectionLoss
  4. AMP 混合精度反向
  5. EMA 更新（decay≈0.9999）
  6. val: EMA 权重算 mAP
  7. 保存 best.pt / last.pt
  8. patience 无提升则早停
```

| 机制 | 说明 |
|------|------|
| **AMP** | FP16 前向 + FP32 梯度，默认开 |
| **EMA** | best.pt 存 EMA 权重，非实时权重 |
| **DDP** | `device=0,1,2,3` 多卡 |
| **freeze** | `freeze=10` 冻结前 10 层 |
| **resume** | 恢复 epoch/optimizer/EMA |

### 7.4 关键超参（default.yaml）

```yaml
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
box: 7.5
cls: 0.5
dfl: 1.5
mosaic: 1.0
close_mosaic: 10
iou: 0.7              # val/NMS 默认 IoU
```

### 7.5 验证与 mAP

```
val 流程:
  1. EMA 模型推理验证集
  2. conf 过滤 + NMS (iou=0.7)
  3. 预测框与 GT 按 IoU 匹配（10 档 0.5~0.95）
  4. 算 AP → mAP@0.5, mAP@0.5:0.95

输出: confusion_matrix, PR_curve, F1_curve, predictions.json
```

**F1_curve 用途**：找 conf 阈值甜点，部署时可参考。

---

## 8. 推理全流程

### 8.1 整体流程

```
输入 → Letterbox(640) → /255 → 前向
  → DFL Integral → dist2bbox → cls sigmoid
  → conf 过滤(0.25) → NMS(0.7) → 坐标还原原图
```

### 8.2 NMS 详解

#### 为何默认 iou=0.7（v5 是 0.45）？

Anchor-free 预测框天然更**贴 GT**，同目标重复框 IoU 常在 0.5~0.7；iou=0.45 会过度抑制。0.7 是 v8 调参结果。

#### 算法（与 v5 相同逻辑）

```
1. 按 conf 降序
2. 取最高 conf 框 → 保留
3. 同类别 IoU > iou_thres 的框 → 删除
4. 重复至空
5. class-aware：不同类不互 suppress
```

#### 参数

| 参数 | 默认 | 作用 |
|------|------|------|
| conf | 0.25 | 进入 NMS 前的门槛 |
| iou | 0.7 | 去重力度 |
| max_det | 300 | 单图上限 |
| agnostic_nms | False | True=跨类 NMS |

| 现象 | 调整 |
|------|------|
| 重复框多 | 降低 iou（0.6） |
| 相邻目标被合并 | 提高 iou（0.75~0.8） |
| 漏检 | 降低 conf |
| FP 多 | 提高 conf |

### 8.3 推理优化

| 方式 | 命令 |
|------|------|
| FP16 | `half=True` |
| Conv+BN fuse | export 时自动 |
| TensorRT | `yolo export format=engine` |
| ONNX | `yolo export format=onnx simplify=True` |
| TTA | `yolo val augment=True`（慢，+0.5~1.5 mAP） |

---

## 9. 扩展任务概览

共享 Backbone+Neck，Head 不同：

| 任务 | Head | 额外输出 | 详解 |
|------|------|----------|------|
| detect | Detect | bbox+cls | 本文 §4~§8 |
| segment | Segment | +mask | **§13** |
| obb | OBB | xywhr | **§14** |
| pose | Pose / **Pose26** | +keypoints | **§16**, **§20** |
| classify | Classify | 类概率 | **§19** |

---

## 13. Instance Segmentation：Segment Head 与 Proto 详解

YOLOv8-seg 在 Detect 基础上增加 **Proto 模块**（共享掩码基）和 **cv4 掩码系数分支**（每实例 32 维），实现 YOLACT 风格的实例分割。

### 13.1 Segment Head 整体结构

```
P3/P4/P5 特征
    │
    ├─ cv2/cv3（继承 Detect）→ bbox + cls
    │
    ├─ cv4（mask 分支，每尺度独立）→ mask_coefficient (nm=32 维)
    │
    └─ P3 特征 x[0] → Proto 模块 → proto (32, H/4, W/4)
```

**Segment 类关键参数（`nn/modules/head.py`）：**

| 参数 | 默认 | 含义 |
|------|------|------|
| nm | 32 | 掩码系数维度 = Proto 输出通道数 |
| npr | 256 | Proto 中间通道数 |
| nc | 80 | 类别数 |

```python
class Segment(Detect):
    def __init__(self, nc=80, nm=32, npr=256, ...):
        super().__init__(nc, ...)
        self.proto = Proto(ch[0], npr, nm)      # 只用 P3 特征
        self.cv4 = ModuleList(...)               # 每尺度 mask 系数头
```

### 13.2 Proto 模块结构（逐层）

Proto 输入 **P3 特征**（最高分辨率检测层，如 80×80×256），输出低分辨率原型掩码：

```
P3 特征 (B, C, 80, 80)
    │
    cv1: Conv 3×3 → (B, 256, 80, 80)
    │
    Upsample: ConvTranspose2d stride=2 → (B, 256, 160, 160)   ← 4× 原图下采样
    │
    cv2: Conv 3×3 → (B, 256, 160, 160)
    │
    cv3: Conv 3×3 → (B, 32, 160, 160)    ← nm=32 个原型平面
    │
输出 proto: (B, 32, 160, 160)
```

**尺寸关系（imgsz=640）：**

| 层级 | 尺寸 | 相对原图 |
|------|------|----------|
| 原图 | 640×640 | 1× |
| P3 特征 | 80×80 | 1/8 |
| Proto 输出 | **160×160** | **1/4**（P3 上采样 2×） |

**为何只用 P3？** P3 分辨率最高、细节最丰富，适合生成精细 mask 原型；P4/P5 语义强但空间粗糙，不适合做 pixel-level 原型。

### 13.3 cv4 掩码系数分支

与 cv2/cv3 平行，每个检测尺度各一套：

```
输入 P3/P4/P5 特征
  Conv 3×3 → Conv 3×3 → Conv 1×1 → (B, nm=32, H, W)
  展平 → (B, 32, N_i)   N_i = H×W

三尺度 concat → mask_coefficient: (B, 32, 8400)
```

每个预测点输出 **32 维系数**，与 Proto 的 32 个通道一一对应。

### 13.4 掩码生成公式

**训练与推理核心（YOLACT 思想）：**

```python
# pred: (n_pos, 32)   正样本的 mask 系数
# proto: (32, Hm, Wm)  单张图的原型，Hm=Wm=160 @ 640输入

pred_mask = einsum('in,nhw->ihw', pred, proto)   # (n_pos, 160, 160)
mask = sigmoid(pred_mask)                         # 概率掩码
```

**直觉理解：**

```
Proto  = 32 张"基础纹理/形状模板"（共享，全图一份）
coeff  = 每个实例对 32 张模板的加权系数（实例独有）
mask   = 32 张模板按 coeff 线性组合 → 该实例的完整掩码
```

**推理后处理：**

```
1. Detect 分支 NMS 得到 K 个框 + 对应 32 维 coeff
2. mask = sigmoid(coeff @ proto)  → (K, 160, 160)
3. 按 bbox 裁剪 crop_mask（去掉框外区域）
4. upsample 双线性插值到原图 640×640
```

### 13.5 训练 Loss（v8SegmentationLoss）

在 v8DetectionLoss 三项基础上增加 **seg loss**：

```
L = L_box + L_cls + L_dfl + L_seg × hyp.box
```

**流程：**

```
1. get_assigned_targets_and_loss()  → 检测 loss + fg_mask + target_gt_idx
2. 对每个正样本 anchor:
     pred_coeff = pred_masks[fg_mask]     (n_pos, 32)
     gt_mask    = 按 target_gt_idx 取对应 GT 实例 mask
     proto      = preds["proto"]          (B, 32, 160, 160)

3. single_mask_loss():
     pred_mask = einsum('in,nhw->ihw', pred, proto)
     loss = BCE_with_logits(pred_mask, gt_mask)
     loss = crop_mask(loss, bbox)           只算框内像素
     loss = mean / area                   按框面积归一化
```

**overlap_mask 模式：**

| 模式 | GT mask 格式 | 说明 |
|------|-------------|------|
| overlap=True | (B, H, W) 整图，像素值=实例 id+1 | 重叠实例用 id 区分 |
| overlap=False | (N_inst, H, W) 每实例独立 | 更常见 |

**crop_mask 作用：** 只在 bbox 范围内计算 BCE，框外像素不参与，减少背景噪声梯度。

### 13.6 导出与推理输出

ONNX 导出 segment 模型有 **两个输出**：

```
output0: (1, 4+nc+nm, 8400)  = bbox + cls + mask_coeff  （与 detect 类似 +32 维）
output1: (1, 32, 160, 160)   = proto 原型图
```

部署时需同时取 output0 的 coeff 和 output1 的 proto 做矩阵乘生成 mask。

---

## 14. OBB 旋转框检测详解

YOLOv8-obb（v8.1+）在 Detect 基础上增加 **角度分支 cv4**，配合 **RotatedTaskAlignedAssigner** 和 **probiou** 实现旋转目标检测（DOTA 等遥感场景）。

### 14.1 OBB Head 结构

```
P3/P4/P5 特征
    │
    ├─ cv2 → DFL 四边距离（同 detect）
    ├─ cv3 → cls
    └─ cv4 → angle（ne=1，每点 1 个角度 logit）
```

```python
class OBB(Detect):
    def __init__(self, nc=80, ne=1, ...):
        self.cv4 = ModuleList(...)   # 角度分支

    def forward_head(...):
        angle = cat([angle_head[i](x[i]) ...])   # (B, 1, 8400)
        angle = (angle.sigmoid() - 0.25) * pi   # 映射到 [-π/4, 3π/4]
        preds["angle"] = angle
```

**角度解码范围：**

```
raw ∈ (0,1) 经 sigmoid
angle = (sigmoid(raw) - 0.25) × π

范围: [-π/4, 3π/4]  即 [-45°, 135°]
```

**为何减 0.25？** 使角度中心在 0 附近，与 DOTA 数据集目标主方向分布对齐，收敛更快。

### 14.2 旋转框表示与解码

**GT 标签格式（OBB 数据集）：**

```
class  x  y  w  h  angle     归一化 xywh + 弧度 angle
```

**预测框解码（dist2rbox）：**

```python
def bbox_decode(anchor_points, pred_dist, pred_angle):
    pred_dist = DFL_integral(pred_dist)           # 四边距离
    xywh = dist2rbox(pred_dist, pred_angle, anchor_points)  # 中心+宽高
    return cat([xywh, pred_angle], dim=-1)        # (B, N, 5)  xywhr
```

**dist2rbox 原理：**

```
1. 在 anchor 点建立局部坐标系，按 pred_angle 旋转
2. 四边距离 (l,t,r,b) 在该旋转系下还原出 xywh
3. 输出中心 (x,y)、宽高 (w,h)、角度 r
```

### 14.3 ProbIoU：旋转框 IoU

普通 axis-aligned IoU 无法衡量旋转框重叠，YOLOv8 用 **ProbIoU**（论文 [ArXiv 2106.06072](https://arxiv.org/pdf/2106.06072v1.pdf)）：

```
思路: 把旋转框建模为二维高斯分布，用 Hellinger 距离衡量相似度

1. xywhr → 协方差矩阵 (a, b, c)
2. 计算两高斯分布的 Bhattacharyya 距离 bd
3. hd = sqrt(1 - exp(-bd))
4. probiou = 1 - hd    范围 [0, 1]，越大越重叠
```

```python
# utils/metrics.py
def probiou(obb1, obb2, CIoU=False):
    # obb: (N, 5)  xywhr
    a1,b1,c1 = _get_covariance_matrix(obb1)
    a2,b2,c2 = _get_covariance_matrix(obb2)
    # ... 计算 t1, t2, t3 → bd → hd
    iou = 1 - hd
    return iou
```

**与 CIoU 对比：**

| | 水平框 CIoU | 旋转框 ProbIoU |
|---|------------|----------------|
| 输入 | xyxy | xywhr |
| 重叠判定 | 像素交并比 | 高斯分布 Hellinger 距离 |
| 可导 | ✓ | ✓ |
| 用于 | detect box loss | obb box loss + TAL assign |

### 14.4 RotatedTaskAlignedAssigner

继承 `TaskAlignedAssigner`，两处关键 Override：

#### （1）`iou_calculation` → probiou

```python
class RotatedTaskAlignedAssigner(TaskAlignedAssigner):
    def iou_calculation(self, gt_bboxes, pd_bboxes):
        return probiou(gt_bboxes, pd_bboxes).clamp_(0)
```

align_metric 中的 IoU 项变为 ProbIoU。

#### （2）`select_candidates_in_gts` → 旋转框内判定

水平版：判断点是否在 axis-aligned 矩形内。

旋转版：用**向量投影**判断点是否在旋转矩形内：

```python
# 将 OBB 四角转为向量 a,b（邻边）
# ap = 参考点 - 顶点a
# 条件: 0 <= ap·ab/|ab| <= |ab|  且  0 <= ap·ad/|ad| <= |ad|
return (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)
```

### 14.5 v8OBBLoss

```
L = λ_box·L_probiou + λ_cls·L_cls + λ_dfl·L_dfl + λ_angle·L_angle
```

| Loss | 说明 |
|------|------|
| L_probiou | `RotatedBboxLoss`，用 probiou 替代 CIoU |
| L_dfl | 用 `rbox2dist` 替代 `bbox2dist`（考虑角度旋转的 ltrb） |
| L_angle | `sin(2Δθ)²` 角度差损失，按宽高比加权 |

**角度 Loss 细节：**

```python
delta_theta = pred_theta - target_theta
delta_wrapped = delta_theta - round(delta_theta/π)*π   # 包裹到 [-π/2, π/2]
ang_loss = sin(2 * delta_wrapped)²

# 宽高比加权: 接近正方形的框 angle 难确定，降低 angle loss 权重
scale_weight = exp(-(log(w/h))² / λ²)    λ=3
ang_loss *= scale_weight
```

**为何用 sin(2Δθ)²？** 矩形旋转 π 周期等价（180° 对称），sin(2θ) 使 0° 与 180° 损失相同，避免歧义。

### 14.6 OBB 数据与训练

```bash
# 标签格式: class cx cy w h angle（归一化）
yolo obb train data=dota8.yaml model=yolov8n-obb.pt epochs=100

# 过滤极小框（稳定训练）
rw, rh = w*imgsz, h*imgsz
targets = targets[(rw >= 2) & (rh >= 2)]
```

**常见错误：** 用 detect 数据集训练 obb 模型 → 报 `OBB dataset incorrectly formatted`。

---

## 15. TensorRT 导出踩坑指南

YOLOv8 部署 GPU 生产环境常用 TensorRT（`.engine`）。导出链路：**PyTorch → ONNX → TensorRT engine**。

### 15.1 标准导出流程

```bash
# 推荐两步走
yolo export model=yolov8s.pt format=onnx simplify=True opset=12
yolo export model=yolov8s.pt format=engine half=True device=0

# 或一步（内部先 export ONNX 再 build engine）
yolo export model=yolov8s.pt format=engine half=True device=0 workspace=4
```

**内部流程（`engine/exporter.py` → `export/engine.py`）：**

```
1. 检查必须在 GPU 上（CPU 无法 build engine）
2. export_onnx()  生成 .onnx
3. onnx2engine()  TensorRT parser 解析 ONNX → build engine
4. 写入 metadata 到 engine 文件
```

### 15.2 关键参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `format=engine` | — | 导出 TensorRT |
| `device=0` | 无 GPU 时自动设 0 | **必须 GPU** |
| `half=True` | False | FP16，推荐开启，速度 ×1.5~2 |
| `int8=True` | False | INT8 量化，需 calibration 数据集 |
| `dynamic=True` | False | 动态 batch/尺寸 |
| `workspace=4` | 4 GB | TRT build 临时显存上限 |
| `simplify=True` | False | ONNX 简化（export ONNX 阶段） |
| `nms=True` | False | 导出含 NMS 的端到端模型 |
| `batch=1` | 1 | 固定 batch；dynamic 时可设 16 |

### 15.3 常见踩坑与解决

#### 坑 1：CPU 上导出 engine 失败

```
AssertionError: export running on CPU but must be on GPU
```

**解决：** 必须 `device=0`（或有 CUDA 的 GPU）。

---

#### 坑 2：TensorRT 版本不兼容

```
check_version(trt.__version__, '!=10.2.0')   # 10.2.0 有已知 bug
```

| 问题 | 解决 |
|------|------|
| TRT 10.2.0 | 升级到 ≠10.2.0（官方明确禁用） |
| TRT 10.3.0 + JetPack 6 + int8 | end2end 分支自动禁用 |
| CUDA 13 ARM (Jetson/DGX) | 需 TRT 10.15.x |

---

#### 坑 3：dynamic=True + batch=1

```
'dynamic=True' model with 'format=engine' requires max batch size, i.e. 'batch=16'
```

**原因：** 动态 batch 需要 optimization profile 的 max shape，batch=1 时 max=min，TRT 无法优化。

**解决：** `dynamic=True batch=16` 或关闭 dynamic。

---

#### 坑 4：half 与 int8 互斥

```
half=True and int8=True are mutually exclusive
```

**解决：** 二选一。追求速度用 `half=True`；极致压缩用 `int8=True data=your.yaml`。

---

#### 坑 5：INT8 校准数据不足

```
>300 images recommended for INT8 calibration, found N images
```

**解决：** 提供 `data=your.yaml`，确保验证集 >300 张；或用 `fraction=0.5` 增加采样。

---

#### 坑 6：Segment 模型两个输出

Segment 导出 ONNX 有 `output0`（检测+coeff）和 `output1`（proto）。

```
output0: (1, 4+nc+32, 8400)
output1: (1, 32, 160, 160)
```

**踩坑：** 部署代码只处理了 output0，忘记 proto → mask 全错。

**解决：** 推理时同时读取两输出，做 `coeff @ proto`。

---

#### 坑 7：OBB 导出需 simplify

```python
# exporter.py
if self.args.nms and self.model.task == "obb":
    self.args.simplify = True   # fix OBB runtime error related to topk
```

**解决：** OBB + nms 导出时加 `simplify=True`。

---

#### 坑 8：end2end 模型部分格式不支持

```
RKNN/NCNN/executorch/engine+int8 等可能自动禁用 end2end 分支
```

**原因：** end2end 含 TopK 算子，部分推理引擎不支持。

**解决：** 导出标准 detect 模型（非 end2end），NMS 在 CPU/GPU 后处理做。

---

#### 坑 9：max_det 与 anchor 数不兼容

```python
# 小 imgsz 时 anchor 数减少，max_det 需 clamp
m.max_det = min(m.max_det, anchor_count)   # TensorRT 要求 k 为常量
```

**踩坑：** imgsz=320 导出 engine 后推理报错。

**解决：** 减小 `max_det` 或增大 `imgsz`。

---

#### 坑 10：ONNX opset 与算子

```
推荐 opset=12~17，export 时自动 best_onnx_opset()
DFL softmax、SiLU 等在 opset>=12 兼容良好
```

**DFL Integral 在 TRT 中：** export 时 `arange_patch` 会把 `[0,1,...,15]` 固定为常量，避免 dynamic arange 问题。

---

### 15.4 推荐导出配置

**生产检测（平衡速度与精度）：**

```bash
yolo export model=best.pt format=engine device=0 half=True workspace=4 imgsz=640
```

**INT8 极致速度（需校准）：**

```bash
yolo export model=best.pt format=engine device=0 int8=True data=data.yaml fraction=0.5
```

**动态 batch 服务：**

```bash
yolo export model=best.pt format=engine device=0 half=True dynamic=True batch=16
```

**含 NMS 端到端（TensorRT 7.x+）：**

```bash
yolo export model=best.pt format=engine device=0 half=True nms=True
```

### 15.5 导出后验证

```bash
# 用 engine 直接推理
yolo predict model=best.engine source=test.jpg half=True

# 对比 PT vs Engine 精度
yolo val model=best.pt  data=data.yaml
yolo val model=best.engine data=data.yaml half=True
```

**正常情况：** FP16 engine 与 PT 模型 mAP 差距 < 0.5；INT8 可能差 1~2 mAP。

### 15.6 推理加载（DetectMultiBackend）

```python
from ultralytics import YOLO
model = YOLO("best.engine")   # 自动识别格式
results = model("image.jpg", half=True)
```

`nn/backends/tensorrt.py` 兼容 TRT 7~10+ API，支持 dynamic shape 绑定。

---

## 16. Pose 姿态估计详解

YOLOv8-pose 在 Detect 基础上增加 **cv4 关键点分支**，采用 **检测 + 关键点联合训练**：先用 TAL 分配检测正样本，再对正样本 anchor 回归 17 个 COCO 关键点。

### 16.1 Pose Head 结构

```
P3/P4/P5 特征
    │
    ├─ cv2/cv3（继承 Detect）→ bbox + cls
    │
    └─ cv4（pose 分支）→ kpts raw (nk 维)
```

**关键参数：**

| 参数 | COCO 默认 | 含义 |
|------|----------|------|
| kpt_shape | (17, 3) | 17 关键点，每点 (x, y, visibility) |
| nk | 51 | 17 × 3 = 总输出维度 |
| nc | 1（person） | 通常只检测人 |

```python
class Pose(Detect):
    def __init__(self, nc=80, kpt_shape=(17, 3), ...):
        self.nk = kpt_shape[0] * kpt_shape[1]   # 51
        self.cv4 = ModuleList(...)               # 每尺度关键点头
```

**输出（每尺度）：**

```
kpts: (B, 51, N_i)  → concat → (B, 51, 8400)
```

### 16.2 关键点解码

**训练时（`v8PoseLoss.kpts_decode`）：**

```python
y = pred_kpts.clone()
y[..., :2] *= 2.0
y[..., 0] += anchor_points[:, 0] - 0.5    # x 相对 grid 中心
y[..., 1] += anchor_points[:, 1] - 0.5    # y
# 再 × stride → 像素坐标
```

**推理时（`Pose.kpts_decode`）：**

```python
# ndim=3 时含 visibility
y[:, 0::3] = (y[:, 0::3] * 2.0 + (anchors[0] - 0.5)) * strides   # x
y[:, 1::3] = (y[:, 1::3] * 2.0 + (anchors[1] - 0.5)) * strides   # y
y[:, 2::3] = sigmoid(y[:, 2::3])                                  # vis ∈ (0,1)
```

与 bbox 解码类似：`×2 + (anchor-0.5)` 允许关键点偏移超出 cell。

**visibility 三态：**

| GT 值 | 含义 |
|-------|------|
| 0 | 未标注 / 不可见 |
| 1 | 标注但遮挡 |
| 2 | 可见 |

训练时 `kpt_mask = (gt[..., 2] != 0)`，vis=0 的关键点不参与 loss。

### 16.3 v8PoseLoss 总览

```
L = L_box + L_cls + L_dfl + L_kpt × hyp.pose + L_kobj × hyp.kobj

默认 hyp.pose=12.0, hyp.kobj=1.0
```

**流程：**

```
1. get_assigned_targets_and_loss()  → 检测三项 loss + fg_mask + target_gt_idx
2. kpts_decode → 预测关键点（feature map 尺度）
3. _select_target_keypoints()       → 按 target_gt_idx 取 GT 关键点
4. KeypointLoss                      → 位置 loss（OKS 形式）
5. BCE                               → visibility loss（kobj）
```

### 16.4 KeypointLoss 与 OKS

**OKS（Object Keypoint Similarity）** 是 COCO 姿态评估标准，YOLOv8 训练 loss 与其形式一致：

```
对每关键点 i:
  d_i = (pred_x - gt_x)² + (pred_y - gt_y)²

  e_i = d_i / (2 × σ_i)² × area × 2)

  OKS-like loss = 1 - exp(-e_i)     对每个有效关键点

σ_i = OKS_SIGMA[i]   COCO 17 点各有不同的尺度常数
area = GT bbox 面积（归一化到 feature map 尺度）
```

**OKS_SIGMA（COCO 17 点，源码中原始值 /10）：**

```
[0.026, 0.025, 0.025,   # 鼻、左眼、右眼
 0.035, 0.035,           # 左耳、右耳
 0.079, 0.079,           # 左肩、右肩  ← 大 σ，位置偏差容忍大
 0.072, 0.072,           # 左肘、右肘
 0.062, 0.062,           # 腕
 0.107, 0.107,          # 髋
 0.087, 0.087,          # 膝
 0.089, 0.089]          # 踝
```

**为何不同关键点 σ 不同？** 大关节（肩、髋）标注误差天然更大，σ 大 → 同样像素偏差惩罚更小；精细关节（眼、鼻）σ 小 → 要求更准。

**KeypointLoss 源码：**

```python
def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
    d = (pred_x - gt_x)² + (pred_y - gt_y)²
    kpt_loss_factor = n_kpts / (有效关键点数 + eps)   # 实例间平衡
    e = d / ((2 * sigmas)² * (area + eps) * 2)
    return mean(kpt_loss_factor * (1 - exp(-e)) * kpt_mask)
```

**kobj loss（visibility）：**

```python
kpts_obj_loss = BCE(pred_vis_logit, kpt_mask.float())   # 仅 ndim=3 时
```

### 16.5 验证指标：OKS mAP

验证时用 `kpt_iou`（即 OKS）匹配预测与 GT：

```python
def kpt_iou(kpt1, kpt2, area, sigma):
    d = (x1-x2)² + (y1-y2)²
    e = d / ((2*sigma)² * area * 2)
    oks = (exp(-e) * kpt_mask).sum(-1) / kpt_mask.sum(-1)
    return oks    # 范围 [0,1]
```

COCO pose mAP 在 OKS 阈值 0.5~0.95 下计算，类似 bbox mAP。

### 16.6 训练与数据格式

```bash
yolo pose train data=coco-pose.yaml model=yolov8n-pose.pt epochs=100 imgsz=640
```

**标签格式（每行）：**

```
class  cx  cy  w  h  x1 y1 v1  x2 y2 v2  ...  x17 y17 v17
       └─ bbox 归一化 ─┘  └──── 17 关键点 × 3 ────┘
```

**与 detect 的区别：** 在标准 5 列 bbox 后追加 51 列（17点×3）。

### 16.7 推理输出

```
results[0].keypoints.xy     # (N, 17, 2) 像素坐标
results[0].keypoints.conf   # (N, 17)     visibility
results[0].boxes            # 人体框
```

可视化时先画框，再在框内画 17 点骨架连线（COCO skeleton 定义）。

> YOLO26 姿态分支的 **Pose26 Head + RLE Loss** 演进见 **§20**。

---

## 17. INT8 校准流程源码

INT8 量化可将模型体积和推理延迟再降 ~2×，但需要**校准数据集**确定各层量化 scale。YOLOv8 主要在 **TensorRT / OpenVINO** 导出时使用。

### 17.1 整体流程

```
yolo export model=best.pt format=engine int8=True data=data.yaml
    │
    ├─ 1. get_int8_calibration_dataloader()   构建校准 DataLoader
    ├─ 2. export_onnx()                          先导出 FP32 ONNX
    └─ 3. onnx2engine(..., int8=True, dataset=calib_loader)
            │
            ├─ config.set_flag(INT8)
            ├─ EngineCalibrator(dataset)         逐 batch 喂数据
            └─ builder.build_serialized_network  TRT 统计 min/max 或 entropy
```

### 17.2 校准 DataLoader 构建

```python
# engine/exporter.py → get_int8_calibration_dataloader()
def get_int8_calibration_dataloader(self, prefix=""):
    data = check_det_dataset(self.args.data)    # 解析 yaml
    dataset = build_yolo_dataset(
        cfg, data["val"],           # 默认用 val 集
        batch_size=self.args.batch,
        data, mode="val",
        fraction=self.args.fraction,  # 可只取一部分，如 0.5
    )
    # 关闭 Letterbox 动态尺寸（固定 imgsz）
    dataset.transforms.transforms[0].new_shape = max(self.imgsz)
    return build_dataloader(dataset, batch=batch, workers=0, drop_last=True)
```

| 参数 | 作用 |
|------|------|
| `data=xxx.yaml` | **必须**，提供 val 图片路径 |
| `split=val` | 默认 val 集（也可用 train） |
| `fraction=0.5` | 只用 50% 数据加速校准 |
| `batch=8` | 校准 batch size |

**数据量建议：**

```
>300 张：常规 INT8
>100 张：最低可跑，精度可能降 1~2 mAP
Axelera：>100 张
```

### 17.3 EngineCalibrator 逐行

```python
class EngineCalibrator(trt.IInt8Calibrator):
    def __init__(self, dataset, cache=""):
        self.data_iter = iter(dataset)
        self.algo = ENTROPY_CALIBRATION_2 if dla else MINMAX_CALIBRATION
        self.cache = Path(cache)    # xxx.cache 缓存文件

    def get_batch(self, names):
        try:
            im0s = next(self.data_iter)["img"] / 255.0   # [0,1] float
            im0s = im0s.to("cuda")
            return [int(im0s.data_ptr())]   # GPU 内存指针给 TRT
        except StopIteration:
            return None   # 数据用完，校准结束

    def read_calibration_cache(self):
        if self.cache.exists():
            return self.cache.read_bytes()   # 复用已有 cache

    def write_calibration_cache(self, cache):
        self.cache.write_bytes(cache)         # 保存 cache 供下次加速
```

**校准算法：**

| 算法 | 使用场景 |
|------|----------|
| MINMAX_CALIBRATION | 默认 GPU，取各 tensor 运行 min/max |
| ENTROPY_CALIBRATION_2 | Jetson DLA 量化必须用此 |

**Cache 文件：** 与 ONNX 同目录的 `.cache` 文件，第二次 build 可跳过校准（数据分布不变时）。

### 17.4 校准时实际发生什么

```
1. TRT 解析 ONNX 图
2. 插入 Q/DQ（Quantize/Dequantize）节点
3. 用 calibrator 逐 batch 前向：
   - 记录每层激活值分布
   - 计算 INT8 scale = (max - min) / 255
4. 根据 scale 将 FP32 权重转为 INT8
5. 序列化 engine
```

**注意：** 校准数据应**代表真实部署场景**（光照、目标类型、分辨率），否则 INT8 精度损失大。

### 17.5 INT8 导出命令与验证

```bash
# 完整 INT8 导出
yolo export model=yolov8s.pt format=engine device=0 int8=True \
  data=coco.yaml fraction=0.5 batch=8 workspace=4

# 对比精度
yolo val model=yolov8s.pt data=coco.yaml
yolo val model=yolov8s.engine data=coco.yaml half=True   # engine 自动 INT8
```

**典型精度损失：**

| 精度 | mAP 变化 |
|------|----------|
| FP32 PT | 基线 |
| FP16 engine | -0.1~0.3 |
| INT8 engine | -0.5~2.0（取决于校准数据质量） |

### 17.6 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 未指定 data | INT8 必须校准 | 加 `data=xxx.yaml` |
| 校准图片太少 | 统计不充分 | 增加 fraction 或用完整 val |
| INT8 + end2end | JetPack6 TRT10.3 bug | 自动禁用 end2end |
| 第二次 build 慢 | 未用 cache | 保留 `.cache` 文件 |
| 精度掉太多 | 校准集与部署域差异大 | 用部署场景图片做 val |

---

## 18. ONNX 导出与 ONNX Runtime 部署

ONNX 是最通用的中间格式，适合 **Windows/Linux 跨平台、CPU/GPU 推理、进一步转 TRT/OpenVINO**。

### 18.1 导出 ONNX

```bash
yolo export model=yolov8s.pt format=onnx simplify=True opset=12 imgsz=640
```

**内部流程（`export_onnx`）：**

```
1. model.eval().float().fuse()     融合 Conv+BN
2. dummy input (1,3,640,640)
3. torch2onnx → .onnx
4. onnxslim.slim()  图简化（可选）
5. 写入 metadata（stride、names、task 等）
```

**detect 模型输出（默认 export，已含 DFL 解码）：**

```
output0: (1, 4+nc, num_anchors)   # COCO → (1, 84, 8400)

通道布局（Detect._inference）:
  [0:4]   解码后 xywh（letterbox 尺度，未映射原图）
  [4:84]  各类别 cls sigmoid 分数

训练 raw 输出为 (1, nc+reg_max×4, 8400) = (1, 144, 8400)，
export 走推理路径，已合并 DFL + dist2bbox + sigmoid。
```

**dynamic 导出：**

```bash
yolo export model=yolov8s.pt format=onnx dynamic=True batch=1 simplify=True
```

```python
dynamic = {
    "images": {0: "batch", 2: "height", 3: "width"},
    "output0": {0: "batch", 2: "anchors"},
}
```

### 18.2 ONNX Runtime 部署（Python）

**方式一：Ultralytics 内置（推荐）**

```python
from ultralytics import YOLO

model = YOLO("yolov8s.onnx")
results = model("image.jpg", conf=0.25, iou=0.7)
# 自动走 ONNXBackend，含 letterbox + NMS + 坐标还原
```

**方式二：原生 ONNX Runtime（理解底层）**

```python
import cv2
import numpy as np
import onnxruntime as ort

# 1. 创建 Session
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
session = ort.InferenceSession("yolov8s.onnx", providers=providers)
input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]

# 2. 预处理（Letterbox）
def letterbox(img, new_shape=640):
    h, w = img.shape[:2]
    scale = min(new_shape/h, new_shape/w)
    nh, nw = int(h*scale), int(w*scale)
    img = cv2.resize(img, (nw, nh))
    pad = ((new_shape-nw)//2, (new_shape-nh)//2)
    img = cv2.copyMakeBorder(img, pad[1], new_shape-nh-nw-pad[1],
                             pad[0], new_shape-nw-pad[0], cv2.BORDER_CONSTANT, value=(114,114,114))
    return img, scale, pad

img0 = cv2.imread("test.jpg")
img, scale, pad = letterbox(img0)
img = img[:, :, ::-1].transpose(2, 0, 1) / 255.0   # BGR→RGB, HWC→CHW, 归一化
img = img[np.newaxis].astype(np.float32)

# 3. 推理
outputs = session.run(output_names, {input_name: img})
pred = outputs[0]    # (1, 84, 8400)  detect

# 4. 后处理（需自行实现 NMS，或使用 ultralytics 后处理）
# pred[0,:4,:] = decoded xywh, pred[0,4:,:] = cls sigmoid
from ultralytics.utils.ops import non_max_suppression, scale_boxes
# ... 转置、过滤、NMS、映射回原图
```

**方式三：C++ ONNX Runtime（生产环境）**

```cpp
// 核心步骤相同：
// Ort::Session → Letterbox 预处理 → Run → 解析 (1,84,8400) → NMS
// 参考 Ultralytics examples/YOLOv8-ONNXRuntime-CPP
```

### 18.3 ONNXBackend 内部机制

```python
# nn/backends/onnx.py
class ONNXBackend:
    def load_model(self, weight):
        # 选择 Provider
        providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]

        self.session = ort.InferenceSession(weight, providers=providers)

        # IO Binding（静态 shape + GPU 时零拷贝加速）
        if cuda and not dynamic:
            self.io.bind_input(...)
            self.io.bind_output(...)
```

| Provider | 场景 |
|----------|------|
| CUDAExecutionProvider | NVIDIA GPU |
| CoreMLExecutionProvider | Apple MPS |
| CPUExecutionProvider | 无 GPU 兜底 |
| OpenVINOExecutionProvider | Intel CPU/iGPU |

### 18.4 Segment / Pose 的 ONNX 输出

| 任务 | 输出 | shape 示例 |
|------|------|-----------|
| detect | output0 | (1, 84, 8400) |
| segment | output0 + output1 | (1,116,8400) + (1,32,160,160) |
| pose | output0 | (1, 84+51, 8400) = (1,135,8400) |
| obb | output0 | (1, 85, 8400) 含 angle |

部署 segment 时必须同时处理 proto 输出（见 §13.6）。

### 18.5 ONNX 导出踩坑

| 问题 | 解决 |
|------|------|
| opset 过低算子不支持 | `opset=12` 或更高 |
| 动态 arange（DFL） | export 时 `arange_patch` 自动固化 |
| simplify 失败 | 可关 `simplify=False`，手动 onnxslim |
| NMS 在 ONNX 外 | 默认 export **不含 NMS**，需后处理或 `nms=True` |
| `nms=True` 需 torch≥1.13 | 升级 PyTorch |
| 输出需 transpose | ORT 输出 (1,C,N)，按 C 维解析 |
| half ONNX | `half=True` 导出 FP16 模型 |

### 18.6 推荐部署链路

```
开发验证:  YOLO("best.pt")                    最快上手
跨平台 CPU: YOLO("best.onnx")                 ORT CPU
NVIDIA GPU:  YOLO("best.engine") half=True    TensorRT 最快
Intel CPU:   YOLO("best_openvino_model")      OpenVINO
移动端:      YOLO("best.tflite") / NCNN        嵌入式
```

---

## 19. Classify 分类任务（简述）

YOLOv8-cls 去掉 Detect，改用 **Classify Head**：

```
Backbone → nn.AdaptiveAvgPool2d(1) → Conv 1×1 → 全连接式分类
```

```bash
yolo classify train data=imagenet model=yolov8n-cls.pt epochs=100
yolo classify predict model=yolov8n-cls.pt source=image.jpg
```

- 无 TAL、无 NMS、无 DFL
- Loss：CrossEntropy
- 适合整图分类，非目标检测

---

## 20. Pose26 与 RLE Loss（YOLO26 姿态分支）

YOLO26-pose 在 YOLOv8-pose 基础上引入 **Pose26 Head** 与 **RLE（Residual Log-Likelihood Estimation）Loss**，来自 MMPose 的不确定性建模思路，用于提升关键点定位精度，尤其是遮挡、模糊场景。

> 论文：[Human Pose Regression with Residual Log-Likelihood Estimation](https://arxiv.org/abs/2107.11291)（ICCV 2021）  
> 源码：`nn/modules/head.py::Pose26`、`utils/loss.py::PoseLoss26` / `RLELoss`、`nn/modules/block.py::RealNVP`

### 20.1 动机：v8 KeypointLoss 的局限

§16 的 `KeypointLoss` 使用 **固定 OKS_SIGMA** 衡量位置误差：

```
e = d / ((2σ)² × area × 2)
loss = 1 - exp(-e)
```

| 局限 | 说明 |
|------|------|
| σ 固定 | 所有样本、所有遮挡程度共用同一 σ，无法表达「这个点我不确定」 |
| 单峰假设 | 仅惩罚 L2 距离，不建模误差分布的尾部（遮挡时误差往往非高斯） |
| 无显式不确定性 | 推理时只输出 (x,y,vis)，没有 per-keypoint confidence 的统计含义 |

RLE 的思路：**网络同时预测坐标 μ 和不确定性 σ**，再用 **Normalizing Flow（RealNVP）** 学习归一化残差 `error = (μ - gt) / σ` 的真实分布，从而得到更合理的负对数似然 loss。

### 20.2 Pose26 vs Pose（v8）结构对比

```
                    Pose (v8)                         Pose26 (YOLO26)
                    ─────────                         ───────────────
cv4 分支            单路 Conv → nk=51                 共享 pose_head (Conv×2)
                                                        ├─ cv4_kpts  → nk=51  (x,y,vis)
                                                        └─ cv4_sigma → nk_sigma=34 (σx,σy)

解码公式            (pred×2 + anchor-0.5)×stride       (pred + anchor)×stride
额外模块            无                                RealNVP flow_model（6 层）
训练 loss           v8PoseLoss（5 项）                PoseLoss26（6 项，含 rle）
启用条件            yolov8n-pose.yaml                 yolo26n-pose.yaml + end2end=True
```

**Pose26 Head 结构：**

```
P3/P4/P5 特征
    │
    ├─ cv2/cv3（Detect）→ bbox + cls
    │
    └─ cv4（共享 Conv×2）
           ├─ cv4_kpts  → (B, 51, N_i)    关键点 raw
           └─ cv4_sigma → (B, 34, N_i)    每点 σx, σy（仅训练用）
```

```python
class Pose26(Pose):
    def __init__(self, nc, kpt_shape=(17, 3), ...):
        super().__init__(...)
        self.flow_model = RealNVP()                    # 归一化流，学习残差分布
        self.cv4 = ModuleList(Conv×2 for each scale)   # 共享特征
        self.cv4_kpts  = ModuleList(Conv2d → nk)       # 51 维
        self.nk_sigma = kpt_shape[0] * 2               # 34 维
        self.cv4_sigma = ModuleList(Conv2d → nk_sigma)
```

**训练 forward 额外输出：**

```python
preds["kpts"]       # (B, 51, 8400)
preds["kpts_sigma"] # (B, 34, 8400)  仅 training=True 时存在
```

### 20.3 关键点解码差异

| 阶段 | v8 Pose | Pose26 |
|------|---------|--------|
| 训练 decode | `×2 + (anchor-0.5)` 再 `×stride` | `+ anchor` 再 `×stride` |
| 推理 decode | 同左 | `(coord + anchor) × stride` |
| visibility | sigmoid | sigmoid（ndim=3 时） |

**Pose26 解码（源码）：**

```python
# PoseLoss26.kpts_decode — 训练用
y[..., 0] += anchor_points[:, [0]]   # x += anchor_x
y[..., 1] += anchor_points[:, [1]]   # y += anchor_y
# 外部再 ÷ stride_tensor 对齐 feature map 尺度

# Pose26.kpts_decode — 推理用
y[:, 0::ndim] = (y[:, 0::ndim] + self.anchors[0]) * self.strides
y[:, 1::ndim] = (y[:, 1::ndim] + self.anchors[1]) * self.strides
```

v8 的 `×2 - 0.5` 扩大偏移范围；Pose26 改为直接加 anchor，配合 RLE 的 σ 预测，由 loss 约束偏移幅度。

### 20.4 RLE Loss 原理

#### 20.4.1 核心公式

RLE 将关键点回归视为 **异方差高斯 + 流模型修正**：

```
1. 网络预测：μ = (x, y)，σ = (σx, σy)  （σ 经 sigmoid 映射到 (0,1)）
2. 归一化误差：error = (μ - gt) / (σ + ε)     shape (N, 2)
3. 流模型：log_φ = RealNVP.log_prob(error)    学习残差的真实分布
4. RLE loss（每个有效关键点）：
   L = log(σ) - log_φ + log(2σ) + |error|     （residual=True 时）
```

**各项含义：**

| 项 | 作用 |
|----|------|
| `log(σ)` | σ 大 → loss 增，惩罚「过度不确定」 |
| `-log_φ` | 残差符合流模型分布 → 奖励（负号变减 loss） |
| `log(2σ) + \|error\|` | L1 残差项 + 尺度校正，让 flow 学剩余偏差 |

**源码（RLELoss.forward）：**

```python
log_sigma = torch.log(sigma)
loss = log_sigma - log_phi.unsqueeze(1)
if self.residual:
    loss += torch.log(sigma * 2) + torch.abs(error)
if self.use_target_weight:
    loss *= target_weight   # RLE_WEIGHT 逐关节加权
return loss.sum()
```

#### 20.4.2 RealNVP 归一化流

`RealNVP` 是 6 层可逆变换，将 2D 误差映射到标准正态空间：

```
error (2D) ──backward_p──→ z ~ N(0,I)
log_φ = prior.log_prob(z) + log|det(Jacobian)|
```

```python
class RealNVP(nn.Module):
    # 6 组 (s, t) 仿射耦合层，交替 mask [0,1] / [1,0]
    def log_prob(self, x):
        z, log_det = self.backward_p(x)
        return self.prior.log_prob(z) + log_det
```

训练时 flow 参数与检测头**联合更新**，无需预训练；推理时 `fuse()` 会 **丢弃 flow_model**（仅训练辅助 loss）。

#### 20.4.3 RLE_WEIGHT 关节权重

COCO 17 点不同关节对 loss 贡献不同（难定位关节权重更高）：

```
[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,   # 头面部
 1.2, 1.2,                           # 肘
 1.5, 1.5,                           # 腕  ← 最高
 1.0, 1.0,                           # 髋
 1.2, 1.2,                           # 膝
 1.5, 1.5]                           # 踝  ← 最高
```

与 OKS_SIGMA 思路类似：腕、踝、肘、膝等远端关节标注噪声大、遮挡多，提高权重让 RLE 重点优化。

### 20.5 PoseLoss26 训练流程

#### 20.5.1 六项 Loss

```
L = L_box + L_cls + L_dfl
  + L_kpt × hyp.pose      (KeypointLoss，同 v8)
  + L_kobj × hyp.kobj     (visibility BCE)
  + L_rle × hyp.rle       (RLE，默认 hyp.rle=1.0)
```

| 索引 | 名称 | 说明 |
|------|------|------|
| loss[0] | box_loss | CIoU + DFL |
| loss[1] | pose_loss | KeypointLoss（OKS 形式） |
| loss[2] | kobj_loss | visibility BCE |
| loss[3] | cls_loss | 分类 BCE |
| loss[4] | dfl_loss | 分布回归 |
| loss[5] | rle_loss | RLE（仅 Pose26 + flow_model 存在时） |

**启用条件（源码）：**

```python
# nn/tasks.py → PoseModel.init_criterion()
return E2ELoss(self, PoseLoss26) if self.end2end else v8PoseLoss(self)

# PoseLoss26.__init__
self.flow_model = model.model[-1].flow_model
if self.flow_model is not None:
    self.rle_loss = RLELoss(use_target_weight=True)
```

即：**yolo26-pose.yaml 中 `end2end: True`** 时使用 `PoseLoss26`；普通 yolov8-pose 仍用 `v8PoseLoss`。

#### 20.5.2 calculate_rle_loss 逐步推演

```python
# 1. 取正样本、可见关键点
pred_kpt_visible = pred_kpt[kpt_mask]          # (M, 5) = x,y,vis,σx,σy
pred_coords = pred_kpt_visible[:, 0:2]
pred_sigma  = pred_kpt_visible[:, -2:].sigmoid()
gt_coords   = gt_kpt_visible[:, 0:2]

# 2. 归一化误差
error = (pred_coords - gt_coords) / (pred_sigma + 1e-9)
error = error.clamp(-100, 100)                 # 防 NaN

# 3. 流模型打分
log_phi = self.flow_model.log_prob(error)      # (M,)

# 4. 关节权重
target_weights = RLE_WEIGHT[kpt_index]         # 按关键点类型

# 5. RLELoss 聚合
return self.rle_loss(pred_sigma, log_phi, error, target_weights)
```

**与 KeypointLoss 的关系：** 两者**同时计算、同时反传**。KeypointLoss 提供 OKS 形式的坐标监督；RLE 额外约束 σ 与误差分布，二者互补而非替代。

#### 20.5.3 E2ELoss 双头训练

YOLO26 检测/姿态采用 **one2many + one2one** 双头（同 YOLO26 detect）：

```
E2ELoss:
  one2many: PoseLoss26(tal_topk=10)  × o2m  (初始 0.8 → 衰减至 0.1)
  one2one:  PoseLoss26(tal_topk=7)   × o2o  (初始 0.2 → 升至 0.9)
```

训练前期 one2many 主导（多正样本、收敛快）；后期 one2one 主导（NMS-free 推理对齐）。

### 20.6 推理与 fuse

**fuse() 时移除训练专用模块：**

```python
def fuse(self):
    super().fuse()
    self.cv4_kpts = self.cv4_sigma = self.flow_model = None
    # 保留 cv4 共享 backbone conv（若未 fuse）
```

推理路径与 v8 Pose 相同：只输出 `(x, y, vis)`，**不输出 σ**（σ 仅训练时辅助 RLE）。

**导出注意：** 部分后端（IMX 等）若检测到 5 维输出，会裁掉 σ 维：

```python
# utils/export/imx.py
kpt = kpt.view(bs, 17, 5, spatial)
kpt = kpt[:, :, :-2, :]   # 去掉 sigma_x, sigma_y → 3 维
```

### 20.7 v8 vs Pose26 总览

| 维度 | YOLOv8-pose | YOLO26-pose (Pose26) |
|------|-------------|----------------------|
| Head | Pose | Pose26 + RealNVP |
| 关键点 loss | KeypointLoss | KeypointLoss + RLELoss |
| Loss 类 | v8PoseLoss | PoseLoss26（end2end 时） |
| 解码 | `×2+anchor-0.5` | `+anchor` |
| σ 分支 | 无 | cv4_sigma（训练） |
| end2end | 否 | 是（yaml 默认） |
| reg_max | 16 | 1（YOLO26 简化 DFL） |
| 典型收益 | 基线 | COCO pose AP 提升（官方 YOLO26 报告） |

### 20.8 训练命令

```bash
# YOLO26-pose（自动启用 Pose26 + PoseLoss26 + RLE）
yolo pose train model=yolo26n-pose.pt data=coco-pose.yaml epochs=100 imgsz=640

# 对比：YOLOv8-pose（无 RLE）
yolo pose train model=yolov8n-pose.pt data=coco-pose.yaml epochs=100
```

**训练日志中可见：**

```
box_loss  pose_loss  kobj_loss  cls_loss  dfl_loss  rle_loss
  1.23      0.85       0.12      0.45      1.01      0.38      ← Pose26 多一项
```

**调参建议：**

| 参数 | 默认 | 说明 |
|------|------|------|
| `hyp.rle` | 1.0 | RLE loss 增益；过大可能导致 σ 塌缩 |
| `hyp.pose` | 12.0 | KeypointLoss 增益（与 v8 相同） |
| `hyp.kobj` | 1.0 | visibility loss 增益 |

### 20.9 数值示例（单关键点）

设某 wrist 关键点（RLE_WEIGHT=1.5）：

```
pred:  μ=(100, 200),  σ=(0.05, 0.08)  (sigmoid 后)
gt:    (102, 198)
area:  bbox 面积（feature map 尺度）= 400

── KeypointLoss（v8 共用）──
d = (100-102)² + (200-198)² = 8
e = 8 / ((2×0.062)² × 400 × 2) ≈ 2.06
L_kpt = 1 - exp(-2.06) ≈ 0.87

── RLE Loss ──
error = ((100-102)/0.05, (200-198)/0.08) = (-40, 25)
log_φ = flow.log_prob(error) ≈ -3.2  (假设)
L_rle = [log(0.05) - (-3.2) + log(0.1) + 40] × 1.5
      ≈ [-2.99 + 3.2 - 2.30 + 40] × 1.5 ≈ 56.9

最终: L_kpt×12 + L_rle×1  加入总 loss（经 batch 归一化）
```

RLE 对大 error 的惩罚随 σ 自适应：σ 大则 error 缩小，但 `log(σ)` 项阻止网络一味增大 σ 逃避惩罚。

---

| 模块 | YOLOv5 | YOLOv8 |
|------|--------|--------|
| 模块 | C3 | C2f |
| Head | 耦合 1×1 | 解耦双分支 |
| Anchor | 9 | 无 |
| Objectness | 有 | 无 |
| 回归 | CIoU | CIoU + DFL |
| 分配 | build_targets | TAL Top-K |
| cls target | 硬标签 | 软标签 |
| 负样本 cls | 不参与 | 参与 |
| 预测点 | 25,200 | 8,400 |
| NMS iou | 0.45 | 0.7 |
| conf | obj×cls | cls |

---

## 11. 精度优化方向

- **数据**：清洗标注、增大 imgsz、平衡类别
- **训练**：COCO 预训练、close_mosaic、multi_scale、调 box/cls/dfl
- **模型**：n→s→m→l→x，或 v8.2 C3k2 大模型
- **后处理**：调 conf/iou、TTA、集成

---

## 12. 相关论文与参考

| 资源 | 关联 |
|------|------|
| [Ultralytics 文档](https://docs.ultralytics.com/models/yolov8/) | 官方 |
| [YOLOv5 笔记](./YOLOv5.md) | 前代对比 |
| YOLOX (2021) | 解耦头、SimOTA |
| TOOD (2021) | Task-aligned 分配 |
| GFL (2020) | 分布回归、Quality Focal |
| YOLACT (2019) | Proto + coeff 实例分割思想 | §13 |
| ProbIoU (2021) | 旋转框 IoU | §14 |
| OKS / COCO Keypoints | 姿态评估与 loss | §16 |
| RLE (ICCV 2021) | 残差对数似然 + 不确定性 | §20 |
| RealNVP (2016) | 归一化流建模误差分布 | §20 |

---

## 附录

### A. 常用命令

```bash
yolo detect train data=data.yaml model=yolov8s.pt epochs=100 imgsz=640
yolo detect val model=best.pt data=data.yaml
yolo detect predict model=best.pt source=images/ conf=0.25 iou=0.7
yolo export model=best.pt format=onnx simplify=True
yolo segment train data=coco-seg.yaml model=yolov8s-seg.pt
yolo obb train data=dota8.yaml model=yolov8n-obb.pt epochs=100
yolo export model=best.pt format=engine device=0 half=True workspace=4
yolo pose train data=coco-pose.yaml model=yolov8n-pose.pt epochs=100
yolo pose train data=coco-pose.yaml model=yolo26n-pose.pt epochs=100  # Pose26 + RLE
yolo export model=best.pt format=onnx simplify=True dynamic=False
yolo export model=best.pt format=engine int8=True data=data.yaml fraction=0.5
```

### B. 关键源文件

| 路径 | 内容 |
|------|------|
| `nn/modules/block.py` | C2f, SPPF, DFL Integral, **RealNVP** |
| `nn/modules/head.py` | Detect, Segment, Pose, **Pose26**, OBB |
| `utils/loss.py` | v8DetectionLoss, v8PoseLoss, **PoseLoss26**, **RLELoss**, KeypointLoss |
| `utils/metrics.py` | OKS_SIGMA, kpt_iou, RLE_WEIGHT |
| `utils/tal.py` | TaskAlignedAssigner, dist2bbox, make_anchors |
| `cfg/models/v8/yolov8.yaml` | YOLOv8 检测模型结构 |
| `cfg/models/26/yolo26-pose.yaml` | YOLO26 姿态模型结构 |
| `cfg/default.yaml` | 超参 |
| `engine/trainer.py` | 训练主循环 |
| `engine/predictor.py` | 推理 |
| `engine/validator.py` | 验证与 mAP |
| `utils/export/engine.py` | ONNX→TensorRT、INT8 Calibrator |
| `nn/backends/onnx.py` | ONNX Runtime 推理后端 |
| `engine/exporter.py` | 导出总入口、校准 DataLoader |

### C. 文档覆盖清单

**已覆盖：**

- [x] C2f / Backbone / Neck / yaml 缩放
- [x] 解耦头 / Anchor-free / 参考点生成
- [x] DFL 训练软标签 + Integral 推理解码
- [x] TAL 完整流程 / 软标签 / detach / Top-K
- [x] **TaskAlignedAssigner 源码逐行（§5.7）**
- [x] **v8DetectionLoss + DFLoss + BboxLoss 源码逐行（§6.6）**
- [x] 正负样本与 v5 对比
- [x] 三项 Loss 与归一化
- [x] 数据增强 / 训练循环 / EMA / AMP
- [x] NMS 详解与 iou=0.7 原因
- [x] 扩展任务 segment/pose/obb
- [x] **Segment Proto 分支详解（§13）**
- [x] **OBB / probiou / RotatedTAL（§14）**
- [x] **TensorRT 导出踩坑（§15）**
- [x] **Pose / OKS / KeypointLoss（§16）**
- [x] **Pose26 / RLE Loss（§20）**
- [x] **INT8 校准流程源码（§17）**
- [x] **ONNX 导出与 ORT 部署（§18）**
- [x] Classify 简述（§19）
- [x] 验证输出与 mAP

**可继续深入：**

- [ ] OpenVINO / TFLite 部署
- [ ] 自定义 yaml 改结构实操

### D. 参考资料

- [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- [YOLOv5 本地笔记](./YOLOv5.md)
