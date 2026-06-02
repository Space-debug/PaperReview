# Mask R-CNN

> 本文档用于整理 Mask R-CNN 论文精读笔记。  
> 重点：**RoIAlign 量化误差与双线性插值**、**Mask 分支结构与 per-class sigmoid 解耦**，以及 **检测 + 实例分割** 的多任务训练/推理细节。

> 前置阅读：[RCNN.md](./RCNN.md) → [FastRCNN.md](./FastRCNN.md) → [FasterRCNN.md](./FasterRCNN.md)

---

## 1. 概览

### 1.1 基本信息

| 项目 | 内容 |
|------|------|
| 论文 | Mask R-CNN |
| 作者/机构 | Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick（Facebook AI Research） |
| 发表 | ICCV 2017 |
| 任务 | **实例分割**（Instance Segmentation）+ 目标检测 + 人体关键点（扩展） |
| 代码 | [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)、[Detectron](https://github.com/facebookresearch/Detectron) |

### 1.2 核心思想（一句话）

**在 Faster R-CNN 上并行加一个轻量 FCN Mask 分支，用 RoIAlign 替代 RoI Pooling 消除量化误差，并对每个 RoI 预测 K 张类专属 mask、仅对 GT 类通道算 loss，实现检测与分割解耦的多任务学习。**

Mask R-CNN 的两个关键创新：

| 创新 | 解决什么问题 |
|------|-------------|
| **RoIAlign** | RoI Pooling 的 **坐标量化** 导致 mask/bbox 定位不准（分割 AP 损失可达 10+ 点） |
| **Parallel Mask Head + per-class sigmoid** | 旧方法 mask 与 class 竞争（softmax）；Mask R-CNN **解耦**，mask 不影响检测 AP |

### 1.3 整体流水线

```
Input Image
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Backbone + FPN（论文主配置 ResNet-50/101-FPN）               │
│  多尺度特征 P2~P5（小物体用高分辨率层）                         │
└──────────────────────────────────────────────────────────────┘
    │
    ├─────────────────────────────┬────────────────────────────┐
    ▼                             ▼                            │
┌─────────────┐          ┌─────────────────────────────────┐   │
│    RPN      │          │  对每个 RoI 并行三个 Head        │   │
│  proposals  │─────────→│                                 │   │
└─────────────┘          │  ┌─ RoIAlign 7×7  → FC → cls    │   │
                         │  ├─ RoIAlign 7×7  → FC → bbox   │   │
                         │  └─ RoIAlign 14×14 → FCN → mask │   │
                         └─────────────────────────────────┘   │
                                    │                          │
                                    ▼                          │
              检测: NMS(box, class, score)                      │
              分割: 取预测类对应 mask → 阈值 → 贴回原图           │
```

> **Mask R-CNN 相对 Faster R-CNN**
>
> | 维度 | Faster R-CNN | Mask R-CNN |
> |------|--------------|------------|
> | 任务 | 检测 | 检测 + **实例分割** |
> | RoI 特征 | RoI **Pool**ing | RoI **Align** |
> | Head 数 | 2（cls + bbox） | **3**（cls + bbox + **mask**） |
> | 损失 | L_cls + L_box | L_cls + L_box + **L_mask** |
> | Backbone | 常 VGG/ResNet | 常 **ResNet-FPN** |

### 1.4 COCO 精度参考（test-dev）

| 模型 | bbox AP | segm AP | 速度 (V100) |
|------|---------|---------|-------------|
| Faster R-CNN ResNet-101-FPN | 36.2 | — | ~130 ms |
| **Mask R-CNN ResNet-101-FPN** | **39.8** | **36.4** | ~195 ms |
| Mask R-CNN ResNeXt-101-FPN | 41.8 | 37.5 | 更慢 |

- Mask 分支额外开销小（~+65 ms），分割 AP 达到当时 SOTA
- RoIAlign 单独贡献：mask AP **+10 点** 量级（相对 RoI Pool）

---

## 2. 任务定义：实例分割 vs 语义分割

```
语义分割:  每个像素一个类标签（不区分同类两个实例）
实例分割:  每个像素 → (类, 实例 id)，同类不同物体 mask 不合并

Mask R-CNN 输出:  每个检测实例一张二值 mask + 类别 + 框
```

**评估指标（COCO segm AP）**：

- 对每个预测实例：mask 二值化后与 GT mask 算 IoU
- 按 IoU 阈值 0.5:0.95 平均，类内 AP 再宏平均
- **mask AP 对定位精度极敏感** → RoIAlign 的价值在此体现

---

## 3. RoIAlign：论文第一关键细节

### 3.1 RoI Pooling 的量化问题（为何要改）

Fast/Faster R-CNN 的 RoI Pooling 有两处 **强制取整（quantization）**：

```
问题 1: RoI 边界映射到特征图时取 floor/ceil

  x1' = floor(x1 × spatial_scale)      ← 最多偏 1 个特征图像素
  x2' = ceil(x2 × spatial_scale)

问题 2: 划分 bin 时再次 floor/ceil

  bin 边界全部 snap 到整数像素
```

**数值例子（论文 Figure 2 精神）**：

```
原图 RoI 宽 10 px，spatial_scale = 1/16
理论特征图宽度: 10/16 = 0.625 px  → 实际被量化成 1 px 或 0 px

一个应输出 7×7 的 RoI，某 bin 理论宽 0.625 px
→ 量化后整个 bin 可能只占 1 个特征图像素，甚至为空
→ max pool 采样位置完全错位
```

| 影响 | 检测（bbox） | 分割（mask） |
|------|-------------|-------------|
| 量化误差 | 有，但 mAP 影响 ~1–2 点 | **致命**，mask AP 可掉 **~10 点** |
| 原因 | bbox 对亚像素误差不极度敏感 | mask 逐像素对齐，1 px 偏就 IoU 大降 |

### 3.2 RoIAlign 的核心原则

```
✓ 不做任何 spatial 量化（坐标保持浮点）
✓ 每个 bin 内用固定数量的采样点 + 双线性插值读特征
✓ 对采样点聚合（平均）得到 bin 输出
```

### 3.3 RoIAlign 算法（逐步）

**输入**：

- 特征图 F，spatial_scale = s（如 1/4 对 P2，1/16 对 P4）
- RoI = (x1, y1, x2, y2) **原图浮点坐标**
- 输出尺寸 H×W（bbox head: 7×7，mask head: 14×14）
- 每 bin 采样数：sample_ratio = n（论文/默认 **n=2**，即每维 2 点，共 4 点/bin）

**Step 1：RoI 映射到特征图（不取整）**

```
x1' = x1 × s
y1' = y1 × s
x2' = x2 × s
y2' = y2 × s
w'  = x2' - x1'
h'  = y2' - y1'
```

**Step 2：等分 bin（浮点边界）**

```
bin_w = w' / W
bin_h = h' / H

第 (i, j) bin 覆盖区域（0-indexed）:
  [x1' + i·bin_w,  x1' + (i+1)·bin_w) × [y1' + j·bin_h,  y1' + (j+1)·bin_h)
```

**Step 3：bin 内均匀采样 n×n 个点**

```
对 bin (i,j)，在 bin 内部均匀取 n×n 个采样点 (x, y)，通常取 bin 中心附近:

  例 n=2，bin [0.0, 0.625) 上采样点: 0.25, 0.75（相对 bin 内归一化后映射）

通用公式（Detectron 实现）:
  第 k 个采样点 (k = 0..n-1):
    offset = (k + 0.5) / n
    x_sample = x1' + i·bin_w + offset · bin_w
    y_sample = y1' + j·bin_h + offset · bin_h
```

**Step 4：双线性插值读特征**

采样点 (x, y) 通常落在 4 个整数像素之间，用 bilinear interpolation：

```
设 x0 = floor(x), x1 = x0+1,  y0 = floor(y), y1 = y0+1
dx = x - x0,  dy = y - y0

Q11 = F(y0, x0)    Q12 = F(y0, x1)
Q21 = F(y1, x0)    Q22 = F(y1, x1)

V = Q11·(1-dx)(1-dy) + Q12·dx(1-dy) + Q21·(1-dx)dy + Q22·dx·dy
```

**Step 5：聚合**

```
output(i, j, c) = mean{ V(x,y,c) | 所有 n×n 采样点 }

（论文用 average；与 max pool 不同，保留亚像素信息）
```

### 3.4 RoI Pooling vs RoIAlign 对比图

```
RoI Pooling                         RoIAlign
─────────────────                   ─────────────────
x1' = floor(x1×s)  ← 量化          x1' = x1×s        ← 浮点
bin 边界 snap 整数                   bin 边界浮点
每 bin 直接 max 整数像素             每 bin n×n 插值采样再 mean

特征图:  ·  ·  ·  ·                  采样点 × 落在 · 之间
         ·  ·  ·  ·                  用周围 4 个 · 加权
RoI:     [====]  → 偏了              [====]  → 精确对齐
```

### 3.5 反向传播

```
双线性插值对 4 邻域像素可微
→ 梯度按插值权重分配到周围 4 个特征图位置
→ backbone 能收到精确的亚像素级定位梯度
```

这也是 RoIAlign 同时 **小幅提升 bbox AP**（~1–3 点）的原因。

### 3.6 两个 Head 使用不同 RoIAlign 输出尺寸

| Head | RoIAlign 输出 | 后续 |
|------|---------------|------|
| **cls + bbox** | **7×7** | flatten → FC → 4096-d |
| **mask** | **14×14** | 4×Conv → Deconv → **28×28** mask |

mask 分支用更高分辨率 RoI 特征（14×14），保留更多空间细节。

---

## 4. Mask 分支：论文第二关键细节

### 4.1 网络结构（逐层）

```
RoIAlign(F, RoI) → 14 × 14 × 256
    │
    ▼
Conv 3×3, 256, ReLU  ─┐
Conv 3×3, 256, ReLU   ├─ ×4 层（论文 Figure 4）
Conv 3×3, 256, ReLU   │
Conv 3×3, 256, ReLU  ─┘
    │
    ▼  14 × 14 × 256
ConvTranspose 2×2, stride 2, 256  →  28 × 28 × 256
    │
    ▼
Conv 1×1, K (类别数)  →  28 × 28 × K
    │
    ▼
Sigmoid（每通道独立）  →  K 张 28×28 概率图 [0,1]
```

**设计要点**：

```
✓ 全是卷积，无 FC → 保持空间结构（FCN 思想）
✓ 与 cls/bbox head 完全并行，不共享 FC 层
✓ 轻量：4 层 3×3 + 1 次 2× 上采样
✓ 输出 28×28 低分辨率 mask，推理时再双线性放大到框大小
```

### 4.2 为何预测 K 个 mask，但只对 GT 类算 loss

**旧方法问题（FCIS 等 per-class softmax）**：

```
K 个类的 mask 做 softmax 竞争
→ 类 A 像素得分高，类 B 必低
→ mask 预测与 class 预测耦合，互相干扰
→ 检测 AP 下降
```

**Mask R-CNN 解耦方案**：

```
输出: K 个独立 sigmoid 通道，每个通道一张二值 mask
训练: 若 GT 类 = k，只对第 k 通道算 binary cross-entropy
      其余 K-1 个通道不参与 loss

→ 各通道互不竞争
→ mask 分支几乎不影响 bbox AP（论文 ablation 验证）
```

示意：

```
RoI 对应一只狗 (class=5):

  mask 输出: [M_1, M_2, ..., M_5, ..., M_K]   每个 28×28

  训练 loss 只算:  BCE(M_5, GT_mask)
  M_1, M_2, M_3, M_4, M_6... 忽略
```

### 4.3 Mask 损失 L_mask（公式与实现细节）

**定义**（仅对 **正样本 RoI**，即 IoU ≥ 0.5 且为前景类）：

```
L_mask = (1/N_mask) Σ_i BCE(M_i^{(k_i)}, GT_mask_i)

M_i^{(k_i)} = 第 i 个 RoI 的、GT 类 k_i 对应通道的 sigmoid 输出
GT_mask_i   = 该实例 GT mask 对齐到 28×28 的二值目标
N_mask      = mini-batch 中参与 mask loss 的 RoI 数
```

**Per-pixel BCE**：

```
BCE(p, y) = -[ y·log(p) + (1-y)·log(1-p) ]

y = 1  像素在实例 mask 内
y = 0  像素在实例 mask 外
p = sigmoid 预测概率
```

**GT mask 如何构造（训练时）**：

```
1. 取 GT 实例的 polygon / bitmap 全图 mask
2. 与当前 RoI 区域相交（只监督 RoI 内像素）
3. 将相交区域 resize / 采样到 28×28（与网络输出对齐）
4. 二值化（≥0.5 为 1）

注意: 低分辨率 28×28 监督 → 推理时 mask 放大到框尺寸，
      边界略粗，但 AP 仍高（COCO 评估允许一定误差）
```

### 4.4 推理时如何用 mask

```
对每个保留的检测 (box, class=k*, score):

1. 取 mask 分支输出的第 k* 通道: M_{k*} ∈ R^{28×28}
2. sigmoid 后阈值化:  M_{k*} > 0.5 → 二值 mask
3. 将 28×28 mask 双线性 resize 到 box 宽高
4. 贴回原图 box 位置（box 外像素置 0）
5. 可选: 与 box 做 element-wise 裁剪，去溢出

多个实例 mask 可重叠（实例分割允许）
```

---

## 5. 完整网络结构（Faster R-CNN + Mask + FPN）

### 5.1 带 FPN 的整体架构

Mask R-CNN 论文主实验用 **ResNet-FPN**（Lin et al., CVPR 2017），非单纯 ResNet：

```
Input
  │
  ▼
ResNet Backbone (C1~C5)
  │
  ▼
FPN 横向连接 + 自顶向下
  → P2 (1/4), P3 (1/8), P4 (1/16), P5 (1/32)  多尺度特征
  │
  ├─ RPN 在所有 P2~P5 上滑动（每 level 独立 anchor，共享参数）
  │
  └─ RoI 分配到某一 FPN level（按框面积）
         │
         ▼
     RoIAlign 从对应 level 取特征
         │
    ┌────┴────┬────────────┐
    ▼         ▼            ▼
 cls head  bbox head   mask head
 (7×7)     (7×7)       (14×14)
```

### 5.2 RoI 分配到 FPN 哪一层（关键细节）

论文公式（基于 RoI 面积）：

```
设 RoI 宽高为 w, h（原图像素）
k = floor( k0 + log2( sqrt(w·h) / 224 ) )

k0 = 4（ResNet 默认，可调）
k  clamp 到 [2, 5]  → 对应 P2 ~ P5

直觉:
  小物体 (面积小) → 大 k?  Let me recalculate...

Actually standard formula:
  levels = {P2, P3, P4, P5} with strides {4, 8, 16, 32}

  target_level = floor( k0 + log2( sqrt(w*h) / 224 + ε ) )

  Small object (area ~32²)  → P2 (stride 4, 高分辨率)
  Large object (area ~224²) → P4
  Very large → P5

Example:
  sqrt(wh)=32  → log2(32/224)=log2(0.14)≈-2.8 → level 1 → P2
  sqrt(wh)=224 → log2(1)=0 → level 4 → P4
```

| RoI 尺度 | FPN 层 | stride | 原因 |
|----------|--------|--------|------|
| 小 | P2 | 4 | 高分辨率特征，小 mask 需要对齐 |
| 中 | P3/P4 | 8/16 | 平衡语义与空间 |
| 大 | P5 | 32 | 大物体在低分辨率层仍有足够 pixels |

**同一 RoI 的 cls/bbox/mask 三个 head 从同一 FPN level 取特征。**

### 5.3 RPN + FPN 的 Anchor

```
每个 FPN level 使用单一 scale 的 Anchor（相对该层 stride）
常用: P2~P5 上 scale = {32², 64², 128², 256²} 对应各层
aspect ratio 仍为 {1:2, 1:1, 2:1}

总 anchor 数 = Σ (H_i × W_i × 3)
```

---

## 6. 多任务损失与训练

### 6.1 总损失

```
L = L_rpn_cls + L_rpn_box + L_cls + L_box + L_mask

RPN 部分: 与 Faster R-CNN 完全相同
检测部分: L_cls + L_box 与 Fast R-CNN 相同（RoIAlign 替换 Pooling）
新增:     L_mask（仅正样本 RoI，仅 GT 类通道）
```

**各项默认权重均为 1**（λ_mask = 1）。

### 6.2 哪些 RoI 参与哪个 loss

| RoI 类型 | IoU | L_cls | L_box | L_mask |
|----------|-----|-------|-------|--------|
| RPN 正/负 | 0.7/0.3 | — | — | — |
| 检测正样本 | ≥ 0.5 | ✓ | ✓ | **✓** |
| 检测负样本 | [0.1, 0.5) | ✓ | ✗ | ✗ |
| 忽略 | < 0.1 | ✗ | ✗ | ✗ |

```
mini-batch 128 RoI 中约 32 个正样本 → 约 32 个参与 L_mask
负样本只训练分类（学 background），不训练 mask
```

### 6.3 训练超参（COCO，ResNet-FPN）

| 超参 | 值 |
|------|-----|
| 优化器 | SGD, momentum=0.9, weight decay=1e-4 |
| 学习率 | 0.02（8 GPU × 2 img/GPU），step at 60k/80k |
| 迭代 | 90k |
| 输入 | 短边 800，长边 ≤ 1333 |
| RoIAlign sample_ratio | 2 |
| Mask 输出 | 28×28 |
| Anchor | FPN 各层 3 ratio × 1 scale |

### 6.4 训练流程（一步联合，比 Faster R-CNN 4-step 更简单）

```
每个 iteration:
  1. 整图 → FPN → {P2..P5}
  2. RPN → proposals, 算 L_rpn
  3. 采样 128 RoI → 分配 FPN level
  4. RoIAlign → cls/bbox head → L_cls + L_box
  5. 正样本 RoI → mask head → L_mask
  6. L 反向传播，更新 backbone + RPN + 三个 head
```

---

## 7. 推理流程（逐步）

```
输入: 测试图 I
────────────────────────────────────────────────────────────

1. 预处理
   resize（短边 800）→ 均值归一化

2. FPN + RPN
   多尺度特征 → proposals → NMS → Top-1000（COCO 常用）

3. 对每个 RoI（并行 batched）
   a. 按面积分配 FPN level
   b. RoIAlign 7×7  → cls score + bbox delta
   c. RoIAlign 14×14 → mask logits (28×28×K)

4. 检测后处理
   score 过滤 → apply bbox delta → 类内 NMS (IoU=0.5)

5. 分割后处理（对每个保留检测）
   mask = sigmoid(output[:, :, k*]) > 0.5
   resize mask 到 box 尺寸 → 贴回原图

6. 输出
   { bbox, class, score, mask_bitmap }

────────────────────────────────────────────────────────────
ResNet-101-FPN, 单卡 V100: ~195 ms/图
```

---

## 8. 消融实验与关键结论

### 8.1 RoIAlign 的影响（最重要 ablation）

| 方法 | bbox AP | mask AP |
|------|---------|---------|
| RoI Pool | 37.4 | 26.2 |
| **RoIAlign** | **38.7 (+1.3)** | **36.4 (+10.2)** |

**结论**：量化对 mask 是灾难性的；RoIAlign 是实例分割可用性的基础。

### 8.2 Mask 分支设计 ablation

| 配置 | bbox AP | mask AP |
|------|---------|---------|
| 无 mask 分支 | 38.7 | — |
| mask + RoI Pool | 38.7 | 26.2 |
| mask + RoIAlign | 38.7 | 36.4 |
| per-class **softmax** mask | **下降** | 下降 |
| per-class **sigmoid** + 只对 GT 类 loss | **38.7 不变** | **最高** |

**结论**：解耦式 sigmoid mask 不损害检测，是正确设计。

### 8.3 并行 vs 串行 mask head

```
并行（Mask R-CNN）: cls/bbox/mask 各用 RoIAlign，互不共享 FC
串行（mask 依赖 cls 特征）: 检测 AP 略降或 mask AP 略降

→ 并行是论文选择，简单且有效
```

### 8.4 Mask 分辨率

| 监督分辨率 | mask AP | 说明 |
|-----------|---------|------|
| 14×14 | 较低 | 边界太糊 |
| **28×28** | **默认最优** | 精度/速度平衡 |
| 56×56 | 略升 | 计算明显增加 |

---

## 9. 扩展：人体关键点检测（同框架）

Mask R-CNN 框架可 **几乎零改结构** 扩展到 COCO Keypoint：

```
把 mask head 换成 keypoint head:
  RoIAlign 14×14 → Conv 堆叠 → Deconv → 28×28 × K_keypoints

每个关键点一个 heatmap 通道
loss: 仅对可见关键点算 MSE / cross-entropy
```

说明 Mask R-CNN 本质是 **「Faster R-CNN + RoIAlign + 并行 FCN per-RoI head」** 的通用模板。

---

## 10. 与同期实例分割方法对比

| 方法 | 思路 | 问题 |
|------|------|------|
| **FCIS** | 位置敏感 score + mask 联合 softmax | 类间 mask 竞争，慢 |
| **MNC** | 多阶段 cascade mask | 训练复杂，慢 |
| **Mask R-CNN** | RoIAlign + 解耦 sigmoid mask | **简单、快、SOTA** |

---

## 11. 历史地位与局限

### 11.1 贡献

1. **RoIAlign** — 成为所有两阶段/部分单阶段方法的标配；
2. **实例分割范式** — 「检测 + per-RoI mask」沿用至今（Cascade Mask R-CNN 等）；
3. **多任务统一框架** — 检测/分割/关键点一套代码（Detectron）；
4. **FPN + Mask R-CNN** — COCO 基准长期霸榜基线。

### 11.2 局限

| 局限 | 表现 | 后续 |
|------|------|------|
| 两阶段 | 比单阶段慢 | YOLACT, SOLO, Mask2Former |
| 28×28 低分辨率 mask | 边界锯齿 | PointRend, Mask Scoring |
| per-RoI 串行 mask | 实例多时慢 | 全景/语义 Transformer |
| Anchor + RPN | 超参、小物体 | Anchor-free 实例分割 |

### 11.3 R-CNN 系列演进（更新）

```
R-CNN (2014)        SS + 独立 CNN + SVM
Fast R-CNN (2015)   RoI Pool + 联合训练
Faster R-CNN (2016) RPN + 共享 conv
Mask R-CNN (2017)   RoIAlign + mask head + FPN  ← 本文
Cascade Mask R-CNN  多级检测头提高高 IoU AP
```

---

## 12. 精读备忘：易混淆点

### 12.1 RoIAlign 的 sample_ratio

```
sample_ratio = 2  → 每个 bin 内 2×2 = 4 个采样点
不是「整个 RoI 只采样 2 个点」
```

### 12.2 bbox head 和 mask head 的 RoIAlign 尺寸不同

```
cls/bbox: 7×7  → 接 FC，要固定长度向量
mask:     14×14 → 接 Conv，要更大空间分辨率
两者从同一 FPN level、同一 RoI 坐标取特征，但 output_size 不同
```

### 12.3 训练 mask 为何不用 softmax

```
softmax: Σ_k p_k = 1，像素级类互斥 → 一个像素只能属于一个类
         但同一 RoI 内所有像素本就属于同一实例/GT 类
         应对「类 k 的 mask」做二分类，不是 K 类竞争

sigmoid + 仅监督 GT 类通道 = 实例内二分类 + 类间解耦
```

### 12.4 推理 mask 通道选择

```
训练: 用 GT 类 k 选通道
推理: 用检测头预测类 k* 选通道

若分类错 → mask 也会取错通道 → 分割错
因此 mask AP 依赖检测 AP（两阶段固有关联）
```

### 12.5 RoIAlign vs RoI Pool 在代码里的区别

```python
# RoI Pooling: 坐标 int() / round()
x1_feat = int(x1 * spatial_scale)   # 量化

# RoIAlign: 坐标 float，插值
x1_feat = x1 * spatial_scale       # 不量化
value = bilinear_interpolate(feat, y, x)
```

### 12.6 L_mask 是否回传到 RPN

```
可以。Mask loss 梯度经 RoIAlign → FPN → backbone
RPN 与检测 loss 一样共享 backbone 梯度
实践中 mask loss 权重 1 即可，无需特殊调参
```

---

## 13. R-CNN 系列四篇串联总结

```
┌──────────────┬────────────┬──────────────┬──────────────┬──────────────┐
│              │ R-CNN      │ Fast R-CNN   │ Faster R-CNN │ Mask R-CNN   │
├──────────────┼────────────┼──────────────┼──────────────┼──────────────┤
│ Proposal     │ SS         │ SS           │ RPN          │ RPN          │
│ RoI 特征     │ warp+独立  │ RoI Pool     │ RoI Pool     │ RoI Align    │
│ 分类         │ SVM        │ Softmax      │ Softmax      │ Softmax      │
│ 回归         │ 线性       │ FC           │ FC           │ FC           │
│ 分割         │ —          │ —            │ —            │ FCN mask head│
│ 多尺度       │ —          │ —            │ —            │ FPN          │
│ COCO segm AP │ —          │ —            │ —            │ ~36+         │
└──────────────┴────────────┴──────────────┴──────────────┴──────────────┘
```

**演进主线**：检测流水线成熟后，用 **RoIAlign 解决对齐** + **并行 head 解决多任务**，把两阶段框架扩展到像素级实例理解。

---

## 14. 参考资料

- 原论文：[Mask R-CNN](https://arxiv.org/abs/1703.06870)（ICCV 2017）
- 前置：[Faster R-CNN](./FasterRCNN.md)（NIPS 2015）
- 强相关：[Feature Pyramid Networks (FPN)](https://arxiv.org/abs/1612.03144)（CVPR 2017，Mask R-CNN 主配置 backbone）
- 后续：Cascade Mask R-CNN、PointRend、Mask2Former
- 官方实现：[Detectron / Detectron2](https://github.com/facebookresearch/detectron2)

---

*文档版本：初稿 | 对应论文 ICCV 2017 Mask R-CNN*
