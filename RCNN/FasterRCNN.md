# Faster R-CNN

> 本文档用于整理 Faster R-CNN（Towards Real-Time Object Detection with Region Proposal Networks）论文精读笔记。  
> 重点：**RPN 网络结构**、**Anchor 机制**，以及 **RPN + Fast R-CNN 检测头** 的联合训练与推理流程。

> 前置阅读：[RCNN.md](./RCNN.md) → [FastRCNN.md](./FastRCNN.md)

---

## 1. 概览

### 1.1 基本信息

| 项目 | 内容 |
|------|------|
| 论文 | Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks |
| 作者/机构 | Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun（Microsoft Research） |
| 发表 | NIPS 2015（常称 2016 版本） |
| 任务 | 目标检测（首个将 **Proposal 生成** 纳入深度网络的 two-stage 框架） |
| 代码 | [rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) |

### 1.2 核心思想（一句话）

**用 Region Proposal Network（RPN）在共享卷积特征图上滑动小网络，通过 Anchor 机制预测「有没有物体 + 框偏移」，替代 Selective Search，并与 Fast R-CNN 检测头共享 backbone，实现近实时的两阶段检测。**

Faster R-CNN 相对 Fast R-CNN 的本质改动只有一件事：**Proposal 从外部算法变成网络内部模块**。但这一步带来：

1. **Proposal 与检测共享 conv 特征** → 几乎零额外 conv 开销；
2. **RPN 可微、可训练** → proposal 质量随检测任务优化；
3. **SS ~2s/图（CPU）→ RPN ~10ms/图（GPU）** → 整体速度从 ~2.3s 降到 ~0.17s（VGG-16）。

### 1.3 整体流水线（统一网络）

```
Input: 整图 I（任意尺寸，如短边 600）
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Shared Conv Backbone（VGG-16 / ResNet 等）                   │
│  整图一次前向 → 共享特征图 F: (H/16 × W/16 × 512)             │
└──────────────────────────────────────────────────────────────┘
    │
    ├────────────────────────────────┬─────────────────────────┐
    ▼                                ▼                         │
┌─────────────────────┐    ┌─────────────────────────────┐    │
│  RPN（Proposal 分支） │    │  RoI Pooling + 检测头         │    │
│  3×3 conv + 双 1×1 头 │    │  （Fast R-CNN 同款）          │    │
│  → ~300 个 proposals │───→│  对每个 RoI: Softmax + bbox  │    │
└─────────────────────┘    └─────────────────────────────┘    │
    │                                │                         │
    │  objectness score              │  class score            │
    │  + anchor 偏移                   │  + class bbox 偏移      │
    └────────────────────────────────┴─────────────────────────┘
                                     │
                                     ▼
                    NMS → 最终检测框 + 类别 + 置信度
```

> **R-CNN 系列三代对比**
>
> | 方法 | Proposal | Backbone 前向次数 | 训练复杂度 | VOC07 mAP (VGG) | 速度 |
> |------|----------|------------------|-----------|-----------------|------|
> | R-CNN | SS | ~2000 次/图 | 4 阶段 | 66.0 | ~47 s |
> | Fast R-CNN | SS | 1 次/图 | 1 阶段 | 66.9 | ~2.3 s |
> | **Faster R-CNN** | **RPN** | **1 次/图** | 4 步交替 → 可联合 | **73.2** | **~0.17 s (~5 fps)** |

### 1.4 精度与速度参考

| 配置 | mAP (VOC07) | mAP (VOC12) | 测试速度 |
|------|-------------|-------------|----------|
| Fast R-CNN + SS | 66.9 | 65.7 | ~2.3 s/图 |
| Faster R-CNN + RPN (VGG-16) | **73.2** | **70.4** | **~0.17 s/图 (~5 fps)** |
| Faster R-CNN + RPN (ResNet-101) | 76.4 | 73.8 | 更慢但更准 |

- RPN 单独作为 proposal 方法：Recall@300 proposals 与 SS 相当甚至更好
- 端到端训练后 mAP 比 Fast R-CNN + SS **+6~7 点**（proposal 质量 + 特征共享双重收益）

---

## 2. 相对 Fast R-CNN：改了什么、保留了什么

### 2.1 保留的部分（检测头 = Fast R-CNN）

```
✓ RoI Pooling → FC6/FC7 → Softmax 分类 + 类专属 bbox 回归
✓ 检测头多任务损失 L_cls + L_loc（Smooth L1）
✓ RoI 采样：128/iter，25% 正样本，IoU 阈值 0.5 / 0.1
✓ 推理：类内 NMS（IoU 0.3）
✓ 预训练 backbone + 检测微调范式
```

### 2.2 新增 / 替换的部分

```
✗ Selective Search（外部、CPU、不可微）
    → RPN（网络内、GPU、可微、与检测共享 conv）

+ Anchor 机制：每个特征图位置预设 k 个参考框
+ RPN 双输出头：objectness 分类（2k）+ bbox 回归（4k）
+ 4-step 交替训练策略（论文默认）
+ 统一多任务损失 L = L_RPN + L_cls + L_loc
```

### 2.3 Feature Sharing 的意义

```
Fast R-CNN:
  Conv(I) → F ──→ [SS proposals 离线] → RoI Pool → Detect

Faster R-CNN:
  Conv(I) → F ──→ RPN(F) → proposals
              └──→ RoI Pool(F, proposals) → Detect

同一张 F 同时服务 RPN 和检测头，backbone 只算 1 次
```

| 指标 | 无共享（RPN 独立 conv） | 共享 conv |
|------|------------------------|-----------|
| 每图 conv 时间 | ~2× | **1×** |
| mAP | 略低 | **更高** |

---

## 3. Region Proposal Network（RPN）详解

### 3.1 RPN 在做什么

RPN 是一个**全卷积**子网络，在共享特征图上每个位置回答两个问题：

1. **这个 Anchor 里有没有物体？**（二分类 objectness）
2. **若有，框该怎么调整？**（4 维 bbox 回归，类无关）

```
本质：在特征图上 sliding window，每个位置放 k 个 Anchor，
      对每个 Anchor 预测 fg/bg + (tx, ty, tw, th)
```

### 3.2 RPN 网络结构

```
共享特征图 F: (H/16 × W/16 × 512)
    │
    ▼
┌─────────────────────────────────────────┐
│  3×3 Conv, 512 filters, padding=1        │  ← 滑动窗口感受野，整合局部上下文
│  输出: (H/16 × W/16 × 512)               │
└─────────────────────────────────────────┘
    │
    ├────────────────────────────┬────────────────────────────┐
    ▼                            ▼                            │
┌──────────────────┐    ┌──────────────────┐                 │
│  1×1 Conv         │    │  1×1 Conv         │                 │
│  2k filters       │    │  4k filters       │                 │
│  (cls 分支)       │    │  (reg 分支)       │                 │
└──────────────────┘    └──────────────────┘                 │
    │                            │                            │
    ▼                            ▼                            │
 每个位置 2k 分数              每个位置 4k 坐标               │
 (k 个 anchor × fg/bg)        (k 个 anchor × 4 维偏移)         │
```

**参数量极小**：仅 3×3 conv + 两个 1×1 conv，在共享特征上操作，几乎不增加推理时间。

**k 的含义**：每个 sliding 位置预设 **k 个 Anchor**（论文默认 k=9）。

### 3.3 Anchor 机制

#### 3.3.1 什么是 Anchor

Anchor 是特征图每个网格位置上预先定义的 **k 个参考框**，不随网络学习位置，只学习「相对 Anchor 的偏移」。

```
特征图位置 (i, j) 对应原图中心点:
  x = j × stride + stride/2
  y = i × stride + stride/2        (VGG-16: stride = 16)

在该 (x, y) 放置 k 个不同尺度、宽高比的 Anchor 框
```

#### 3.3.2 默认 Anchor 配置（论文）

```
3 种尺度（面积，相对短边 600 的参考图）:  128², 256², 512²
3 种宽高比:                              1:1, 1:2, 2:1

k = 3 × 3 = 9 个 Anchor / 位置

示例（中心在 (x,y)）:
  128×128, 128×64, 256×128, 256×256, 256×512, 512×512, 512×256, ...
  （具体 w,h 由 scale 和 ratio 组合决定）
```

**尺度 × 宽高比组合示意：**

```
              ratio=1:2 (瘦高)   ratio=1:1 (方)    ratio=2:1 (宽扁)
scale=128²      ┌──┐            ┌────┐           ┌────────┐
                │  │            │    │           │        │
                └──┘            └────┘           └────────┘
scale=256²      ┌──┐            ┌────┐           ┌────────┐
                │  │            │    │           │        │
                │  │            │    │           │        │
                └──┘            └────┘           └────────┘
scale=512²      ...             ...              ...
```

#### 3.3.3 为何需要 Anchor

| 无 Anchor（直接预测框） | 有 Anchor |
|------------------------|-----------|
| 每个位置需预测绝对 (x,y,w,h) | 只预测相对偏移，目标值更小、更稳定 |
| 多尺度物体难用固定感受野覆盖 | 多尺度/多比例 Anchor 覆盖不同大小 |
| 平移不变性需数据学 | Anchor 中心随网格平移，天然平移不变 |

#### 3.3.4 Anchor 总数

```
特征图大小: (H/16) × (W/16)
总 Anchor 数: (H/16) × (W/16) × k

例: 600×800 输入 → 特征图 ~38×50 → 38×50×9 ≈ 17,100 个 Anchor
```

### 3.4 RPN 标签分配（训练 Anchor）

对每个 Anchor，计算其与**所有 GT** 的最大 IoU：

| 条件 | 标签 | 回归目标 |
|------|------|----------|
| IoU > **0.7** 与某 GT | **正（fg=1）** | 有（相对匹配 GT） |
| IoU < **0.3** 与所有 GT | **负（bg=0）** | 无 |
| IoU ∈ [0.3, 0.7] | **忽略** | 无 |
| **额外**：与某 GT IoU **最高** 的 Anchor | **正（fg=1）** | 有（保证每个 GT 至少 1 个正样本） |

> 与检测头不同：RPN 用 **0.7 / 0.3** 阈值（更严格），检测头仍用 **0.5 / 0.1**。

**「最高 IoU 必为正」规则示意：**

```
某 GT 框周围多个 Anchor IoU 分别为 0.4, 0.55, 0.62
→ 仅 IoU=0.62 的那个强制标为正（即使 < 0.7）
→ 保证每个 GT 至少有一个回归正样本
```

### 3.5 RPN 损失函数

```
L({p_i}, {t_i}) = (1/N_cls) · L_cls + λ · (1/N_reg) · L_reg

p_i     = Anchor i 的 objectness 预测（2 类 softmax: bg / fg）
t_i     = Anchor i 的 bbox 回归预测
N_cls   = 256（mini-batch 中采样的 Anchor 数）
N_reg   = 128（参与回归的 Anchor 数，仅 fg）
λ       = 10（论文默认，回归 loss 权重）
```

**分类损失 L_cls**（二分类）：

```
对每个采样 Anchor，2 类 softmax 交叉熵（fg vs bg）
仅对参与采样的 256 个 Anchor 计算
正负比例约 1:1（随机采样，保证 mini-batch 平衡）
```

**回归损失 L_reg**（Smooth L1，类无关）：

```
L_reg = Σ smooth_L1(t_i - v_i)     仅对 fg Anchor 计算

v_i = 相对 Anchor 的 GT 偏移（与 Fast R-CNN 相同参数化）:
  v_x = (G_x - A_x) / A_w
  v_y = (G_y - A_y) / A_h
  v_w = log(G_w / A_w)
  v_h = log(G_h / A_h)

A = Anchor 框, G = 匹配的 GT 框
```

### 3.6 从 RPN 输出到 Proposals（推理）

```
1. 对全部 ~17k Anchor 预测 (p_fg, t)
2. 取 objectness score 最高的 Anchor（或全部）
3. 将回归偏移应用到 Anchor → 得到 refined box
4. 裁剪到图像边界
5. 去掉过小框（宽或高 < 16 px）
6. NMS（IoU 阈值 = 0.7，类无关，因 RPN 不分类别）
7. 保留 Top-N:
     训练时 N = 2000（供检测头采样 RoI）
     测试时 N = 300
```

```
~17,100 Anchors
    → apply delta + clip + filter small
    → NMS (IoU=0.7)
    → Top-300 proposals → 送入 RoI Pooling + 检测头
```

---

## 4. 检测头（Fast R-CNN 部分）

检测头结构与 [FastRCNN.md](./FastRCNN.md) 完全一致，此处只强调与 RPN 的衔接。

### 4.1 结构

```
共享特征图 F + RPN proposals (RoIs)
    │
    ▼
RoI Pooling (7×7×512) → FC6 → FC7
    │
    ├─ FC_cls:  4096 → (K+1)    Softmax 分类
    └─ FC_bbox: 4096 → K×4      类专属 bbox 回归
```

### 4.2 检测头损失（与 Fast R-CNN 相同）

```
L_det = L_cls + L_loc

L_cls = -log(p_u)                    u 为 RoI 真实类（0=bg, 1..K）
L_loc = λ · [u≥1] · smooth_L1(t^u - v)    λ=1, 仅前景 RoI
```

### 4.3 检测头 RoI 来源（训练时）

```
每张训练图的 RoI 来自:
  1. RPN 产生的 Top-2000 proposals（在线生成，非离线 SS）
  2. 所有 GT 框（保证高 IoU 正样本）

从中采样 128 个 RoI（25% 正）计算 L_det
```

### 4.4 RPN vs 检测头：两次 bbox 回归

| | RPN 回归 | 检测头回归 |
|--|---------|-----------|
| **输入** | Anchor | RPN proposal（已是 refined box） |
| **输出** | 类无关 4 维 | 类专属 K×4 维 |
| **目的** | 粗定位，产生 proposal | 精修 + 分类 |
| **IoU 阈值** | 0.7 / 0.3 | 0.5 / 0.1 |

推理时**两次回归串联**：Anchor → RPN delta → proposal → 检测头 delta → 最终框。

---

## 5. 统一网络与多任务损失

### 5.1 完整网络结构图

```
                        Input Image
                             │
                             ▼
              ┌──────────────────────────┐
              │   Shared Conv Backbone    │
              │   (VGG conv1_1 ~ conv5_3)│
              └──────────────────────────┘
                             │
                      特征图 F (共享)
                    ┌────────┴────────┐
                    ▼                 ▼
           ┌──────────────┐   ┌──────────────────┐
           │     RPN      │   │  RoI Pooling      │
           │  3×3 + 1×1×2 │   │  (proposals 来自  │
           │              │   │   RPN 输出)       │
           └──────────────┘   └──────────────────┘
                    │                 │
              proposals             ▼
              (~300/2000)    ┌──────────────────┐
                    │        │  FC6 → FC7        │
                    └───────→│  cls + bbox head  │
                             └──────────────────┘
                                    │
                                    ▼
                              Detections
```

### 5.2 总损失

联合训练时：

```
L = L_RPN + L_det
  = L_RPN_cls + λ_rpn · L_RPN_reg + L_det_cls + L_det_loc

λ_rpn = 10（RPN 回归权重）
检测头 λ = 1
```

**一个 iteration 的前向 + 反向：**

```
1. 整图 → F
2. RPN(F) → proposals + L_RPN（256 采样 Anchor）
3. RoI Pool(F, proposals) → Detect + L_det（128 采样 RoI）
4. L = L_RPN + L_det → 反向更新 backbone + RPN + 检测头
```

---

## 6. 训练策略

### 6.1 四步交替训练（论文默认）

因 RPN 与检测头联合训练不稳定，论文提出 **4-step alternating training**：

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: 训练 RPN（独立）                                      │
│   初始化: ImageNet 预训练 backbone                            │
│   训练: 仅 RPN 层（3×3 + 两个 1×1 head）                      │
│   数据: VOC trainval                                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: 训练 Fast R-CNN 检测头（RPN 固定）                    │
│   用 Step1 的 RPN 产生 proposals                              │
│   训练: RoI Pooling + FC + cls/bbox head                     │
│   backbone 与 RPN 权重固定                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: 微调 RPN（检测头固定）                                 │
│   共享 conv 用 Step2 的权重初始化                              │
│   训练: RPN（检测头固定，proposals 随 RPN 更新而变）            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: 微调检测头（RPN 固定）                                 │
│   用 Step3 的 RPN 产生 proposals                              │
│   训练: 检测头 + 共享 conv（RPN 固定）                         │
└─────────────────────────────────────────────────────────────┘
```

> 后续实现（如 py-faster-rcnn、Detectron）多采用 **近似联合训练**（一步 SGD 同时优化 L_RPN + L_det），效果与 4-step 接近且更简单。

### 6.2 超参数汇总

| 模块 | 超参 | 值 |
|------|------|-----|
| **RPN** | 采样 Anchor 数 N_cls | 256 |
| | 正负比例 | 1:1 |
| | IoU 正/负阈值 | 0.7 / 0.3 |
| | λ (reg 权重) | 10 |
| | NMS IoU | 0.7 |
| | Top-N proposals | 2000 (train) / 300 (test) |
| **检测头** | 采样 RoI 数 | 128 |
| | 正负比例 | 1:4 (25% 正) |
| | IoU 正/负阈值 | 0.5 / 0.1 |
| | λ (reg 权重) | 1 |
| **共用** | 优化器 | SGD, momentum=0.9, wd=5e-4 |
| | 学习率 | 0.001，step decay |
| | 输入短边 | 600 px |

### 6.3 Anchor 超参敏感性

| 变量 | 影响 |
|------|------|
| Anchor 尺度 `{128²,256²,512²}` | 需覆盖数据集中物体尺度；COCO 等大数据集常用更多尺度 |
| Anchor 比例 `{1:1,1:2,2:1}` | 覆盖常见形状；极端比例物体可能漏检 |
| k 过大 | 计算量增加，NMS 前候选更多 |
| k 过小 | 小/大/扁/长物体 recall 下降 |

---

## 7. 推理流程（逐步）

```
输入: 测试图 I
────────────────────────────────────────────────────────────

1. 图像预处理
   等比缩放（短边 600）→ 减均值

2. Backbone 前向（一次）
   F = Conv(I)                         # H/16 × W/16 × 512

3. RPN 前向
   对每个 Anchor: 预测 fg score + (tx,ty,tw,th)
   apply delta → clip → filter → NMS(0.7) → Top-300 proposals

4. 检测头前向
   对每个 proposal: RoI Pool → FC → cls scores + bbox deltas

5. 后处理
   k* = argmax cls (skip background)
   refined = apply_delta(proposal, bbox_delta[k*])
   类内 NMS (IoU=0.3)

6. 输出: {(box, class, score)}

────────────────────────────────────────────────────────────
耗时（VGG-16, GPU, 不含图像 IO）:
  Backbone:     ~80 ms
  RPN:          ~10 ms
  检测头:       ~80 ms
  总计:         ~170 ms → ~5 fps
```

**对比 Fast R-CNN**：SS ~2000ms → RPN ~10ms，**proposal 阶段加速 ~200×**。

---

## 8. 消融实验与关键结论

### 8.1 RPN vs Selective Search

| Proposal 方法 | mAP (VOC07) | 时间 | 可训练 |
|---------------|-------------|------|--------|
| Selective Search | 66.9 (Fast R-CNN) | ~2 s | 否 |
| RPN（无共享 conv） | ~69 | ~300 ms | 是 |
| **RPN + 共享 conv** | **73.2** | **~170 ms** | 是 |

### 8.2 共享特征的影响

```
不共享: RPN 与检测各跑一遍 backbone → 慢 + mAP 降 ~2 点
共享:   一次 backbone → RPN 与检测头复用 F → 快 + 更准
```

### 8.3 两次 bbox 回归的贡献

| 配置 | mAP |
|------|-----|
| 仅 RPN reg，检测头不回归 | 明显更低 |
| 仅检测头 reg，RPN 不回归 | 较低（proposal 差） |
| **RPN + 检测头都回归** | **最高** |

### 8.4 Anchor 设计消融

- 单尺度单比例（k=1）：mAP 显著下降
- k=9（3 scale × 3 ratio）：论文默认，效果与 k=12 接近
- 移除「最高 IoU 必为正」规则：部分 GT 无正样本，训练不稳定

### 8.5 跨数据集泛化

RPN 在 **COCO 上训练、VOC 上测** 仍优于 SS，说明 RPN 学到的是**类无关的 objectness**，而非类别特定模式。

---

## 9. 历史地位与后续影响

### 9.1 里程碑贡献

1. **Region Proposal Network** — 首个将 proposal 生成纳入 CNN 的方法；
2. **Anchor 机制** — 成为两阶段/单阶段检测的标准组件（SSD、RetinaNet、YOLO 系列均借鉴）；
3. **共享特征两阶段框架** — RPN + RoI Head 模板沿用至 Mask R-CNN、Cascade R-CNN 等；
4. **近实时两阶段检测** — VGG-16 上 ~5 fps，首次让 two-stage 具备实用速度；
5. **统一检测范式** — 后续 Detectron / MMDetection 等均以此为基础。

### 9.2 仍存在的局限

| 局限 | 表现 | 后续改进 |
|------|------|----------|
| **两阶段串行** | RPN → RoI Head 不能并行 | 单阶段 SSD / RetinaNet |
| **RoI Pooling 量化** | 定位有误差 | RoI Align（Mask R-CNN） |
| **Anchor 需手工设计** | 换数据集要调 scale/ratio | FPN + 多尺度 Anchor；Anchor-free（FCOS 等） |
| **小物体** | 单层特征图 stride=16 易漏 | FPN（Feature Pyramid Network） |
| **训练复杂** | 4-step 交替 | 近似联合训练；Cascade 多阶段 |

### 9.3 演进脉络

```
Faster R-CNN (2015/16)   RPN + RoI Head + 共享 backbone
    ↓
+ FPN (2017)             多尺度特征金字塔，小物体 AP 提升
    ↓
Mask R-CNN (2017)        RoI Align + mask 分支，实例分割
    ↓
Cascade R-CNN (2018)     多级检测头串行，提高 IoU 阈值 AP
    ↓
单阶段 / Anchor-free       追求速度与简洁
```

---

## 10. 精读备忘：易混淆点

### 10.1 RPN 与检测头的 IoU 阈值不同

| 模块 | 正样本 | 负样本 | 忽略 |
|------|--------|--------|------|
| **RPN** | IoU > 0.7，或 GT 最高 IoU | IoU < 0.3 | [0.3, 0.7]（非最高者） |
| **检测头** | IoU ≥ 0.5 | [0.1, 0.5) | < 0.1 |

RPN 阈值更严 → proposal 更「准」但 recall 靠「最高 IoU 必为正」保底。

### 10.2 RPN 分类是「类无关」二分类

```
RPN:     只分 fg / bg，不知道「是什么类」
检测头:  分 K+1 类（含 background），知道「是什么类」

RPN NMS 也是类无关的（所有 fg proposal 一起做 NMS）
检测 NMS 是类内独立的
```

### 10.3 训练 2000 proposals vs 测试 300

```
训练: Top-2000 → 检测头有足够负样本可采样
测试: Top-300  → 速度更快，recall 略降但 mAP 几乎不变
```

### 10.4 Anchor 坐标在哪个空间

```
Anchor 定义在原图像素空间（缩放后的 600 短边图）
RPN 在特征图上滑动，但通过 stride 映射回原图
回归目标 v 是 Anchor 与 GT 在原图坐标下的偏移
```

### 10.5 「Faster」相对谁

```
相对 R-CNN:   ~275× 加速（47s → 0.17s）
相对 Fast R-CNN: ~13× 加速（2.3s → 0.17s），主要省在 proposal
```

### 10.6 与 YOLO 的 Anchor 区别

| | Faster R-CNN RPN | YOLO v2+ |
|--|-----------------|----------|
| Anchor 用途 | 产生 proposal（两阶段第一阶段） | 直接预测 class+box（单阶段） |
| 分类 | 仅 fg/bg | 直接预测 K 类 |
| 数量 | ~17k Anchor → NMS → 300 | 固定 grid × k 个 |

---

## 11. R-CNN 系列三篇串联总结

```
┌─────────────┬──────────────────┬──────────────────┬──────────────────┐
│             │ R-CNN            │ Fast R-CNN       │ Faster R-CNN     │
├─────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Proposal    │ SS (外部)        │ SS (外部)        │ RPN (网络内)     │
│ 特征提取    │ 每 region 独立   │ 整图共享 + RoI   │ 整图共享 + RoI   │
│ 分类        │ SVM              │ Softmax          │ Softmax          │
│ 回归        │ 线性回归         │ FC 回归头        │ RPN reg + FC reg │
│ 训练        │ 4 阶段           │ 1 阶段           │ 4 步交替/联合    │
│ 速度        │ 47 s             │ 2.3 s            │ 0.17 s (~5 fps)  │
│ mAP (VGG)   │ 66.0             │ 66.9             │ 73.2             │
└─────────────┴──────────────────┴──────────────────┴──────────────────┘
```

**一条演进主线**：把「重复计算 → 共享计算 → proposal 也进网络」，同时「多阶段训练 → 端到端联合」。

---

## 12. 参考资料

- 原论文：[Faster R-CNN](https://arxiv.org/abs/1506.01497)（NIPS 2015）
- 前置：[R-CNN](./RCNN.md)（CVPR 2014）、[Fast R-CNN](./FastRCNN.md)（ICCV 2015）
- 后续：**Mask R-CNN**（2017，RoI Align + 实例分割）、**FPN**（2017，多尺度特征）
- 官方实现：[rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)

---

*文档版本：初稿 | 对应论文 NIPS 2015 Faster R-CNN*
