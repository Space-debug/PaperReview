# Fast R-CNN

> 本文档用于整理 Fast R-CNN（Fast Region-based Convolutional Network method for object detection）论文精读笔记。  
> 重点：**RoI Pooling 核心结构**、**多任务联合训练**，以及相对 R-CNN 在**推理/训练流程上的改造**。

> 前置阅读：[RCNN.md](./RCNN.md)

---

## 1. 概览

### 1.1 基本信息

| 项目 | 内容 |
|------|------|
| 论文 | Fast R-CNN |
| 作者/机构 | Ross Girshick（Microsoft Research） |
| 发表 | ICCV 2015 |
| 任务 | 目标检测（R-CNN 的直接改进，两阶段检测第二阶段雏形） |
| 代码 | [rbgirshick/fast-rcnn](https://github.com/rbgirshick/fast-rcnn)（Python + Caffe） |

### 1.2 核心思想（一句话）

**整图只跑一次 CNN 得到共享特征图，再用 RoI Pooling 把任意大小的候选区变成固定长度向量，在同一网络里同时做 Softmax 分类和边界框回归，端到端联合训练。**

Fast R-CNN 相对 R-CNN 的三大改造：

| R-CNN 痛点 | Fast R-CNN 解法 |
|-----------|----------------|
| 每个 region 独立 CNN 前向，极慢 | **整图共享卷积**，2000 个 region 共用一张特征图 |
| CNN、SVM、回归器分阶段训练 | **多任务损失**，Softmax + bbox reg 同一网络联合优化 |
| fc7 特征离线缓存，磁盘占用数百 GB | **在线 RoI Pooling**，训练/推理无需存特征 |

### 1.3 整体流水线（三阶段 → 实质两阶段）

```
Input: 整图 I + N 个 RoI proposals（SS / EdgeBoxes，仍来自外部）
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Region Proposal（外部，与 R-CNN 相同）             │
│  每张图 ~2000 个候选框，训练时可预计算离线存储                  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: 共享 CNN + RoI Pooling（Fast R-CNN 核心）          │
│  整图 → Conv Backbone → 特征图 (H/16 × W/16 × C)            │
│  每个 RoI → RoI Pooling → 固定 7×7×512 → flatten → FC        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: 双分支 Head（同一网络，联合训练）                   │
│  ├─ Softmax: K+1 类分类得分（含 background）                  │
│  └─ Bbox Regressor: 每类 4 维偏移 (tx, ty, tw, th)           │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Output: NMS 后的检测框 + 类别 + 置信度
```

> **与 R-CNN / Faster R-CNN 对比**
>
> | 方法 | 卷积次数/图 | 分类器 | 回归器 | Proposal 来源 | 训练阶段数 |
> |------|------------|--------|--------|---------------|-----------|
> | **R-CNN** | ~2000 次 | 21 个线性 SVM | 21 个线性回归 | Selective Search | 4（预训练+微调+SVM+回归） |
> | **Fast R-CNN** | **1 次** | Softmax（网络内） | FC 回归头（网络内） | SS / EdgeBoxes | **1**（预训练+联合微调） |
> | **Faster R-CNN** | 1 次 | Softmax | FC 回归头 | **RPN（网络内）** | 1（交替/联合） |

### 1.4 PASCAL VOC 精度与速度参考

| 方法 | Backbone | mAP (VOC07) | 测试速度 (GPU) |
|------|----------|-------------|----------------|
| R-CNN | VGG-16 | 66.0 | ~47 s/图 |
| SPP-net | VGG-16 | 63.1 | ~2.5 s/图 |
| **Fast R-CNN** | VGG-16 | **66.9** | **~0.22 s/图**（不含 SS） |
| Fast R-CNN | VGG-16 | 66.9 | ~2.3 s/图（含 SS ~2s） |

- 训练速度比 R-CNN 快 **~9×**（单 GPU）
- 测试速度比 R-CNN 快 **~213×**（含 SS）；不含 SS 约 **~61×**
- 精度略超 R-CNN，同时大幅提速

---

## 2. 相对 R-CNN：改了什么、保留了什么

### 2.1 保留的部分

```
✓ Region-based 范式：仍依赖外部 proposal，不在此阶段「找框」
✓ 预训练 + 检测微调范式（ImageNet → VOC）
✓ Bbox 回归参数化 (tx, ty, tw, th)（与 R-CNN 完全一致）
✓ RoI 正负样本 IoU 阈值思路（≥0.5 正，0.1~0.5 负）
✓ Mini-batch 128，25% 正样本
✓ 推理时类内 NMS（IoU 阈值 0.3）
```

### 2.2 革新的部分

```
✗ 去掉 per-region CNN 前向          → 整图共享 conv
✗ 去掉 21 个独立 linear SVM         → Softmax 分类头
✗ 去掉 21 个独立 linear bbox reg    → FC 回归头（仍类专属）
✗ 去掉 fc7 特征磁盘缓存             → RoI Pooling 在线提取
✗ 去掉 SVM hard negative mining     → 端到端 SGD 自动学
```

### 2.3 与 SPP-net 的关系

Fast R-CNN 借鉴了 **SPP-net**（He et al., ECCV 2014）的「整图卷积 + 空间金字塔池化」思想，但做了关键改进：

| 维度 | SPP-net | Fast R-CNN |
|------|---------|------------|
| 池化方式 | Spatial Pyramid Pooling（多尺度 bin） | **RoI Pooling**（单尺度 7×7，更简洁） |
| 微调范围 | 只能微调 **FC 层**（无法有效反传到 conv） | **所有层** 可端到端微调 |
| 检测头 | 仍用 SVM 分类 | Softmax + bbox reg 联合 |
| 训练 | 多阶段 | 单阶段 SGD |

> SPP-net 无法微调 conv 的根因：当时实现里 batch 中每张图的 proposal 数不同，导致反向传播到 conv 层时效率极低；Fast R-CNN 通过 **RoI Pooling + 截断反向传播（只回传 sampled RoI）** 解决了这个问题。

---

## 3. 网络结构详解

### 3.1 整体架构（以 VGG-16 为例）

```
Input Image: 任意尺寸 H×W×3（整图，不 warp）
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Conv Backbone: VGG-16（13 层 conv + 5 层 maxpool）           │
│  输出 conv5_3 特征图: (H/16) × (W/16) × 512                   │
│  ← 整张图只算这一次，所有 RoI 共享                              │
└──────────────────────────────────────────────────────────────┘
    │
    │  同时输入 N 个 RoI: (x1, y1, x2, y2) 像素坐标
    ▼
┌──────────────────────────────────────────────────────────────┐
│  RoI Pooling Layer                                           │
│  每个 RoI → 固定大小 7×7×512                                  │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  FC6: 4096-d + ReLU + Dropout                                │
│  FC7: 4096-d + ReLU + Dropout                                │
└──────────────────────────────────────────────────────────────┘
    │
    ├─────────────────────────────┬──────────────────────────────┐
    ▼                             ▼                              │
┌─────────────────┐    ┌─────────────────────────┐               │
│  FC_cls         │    │  FC_bbox                │               │
│  4096 → (K+1)   │    │  4096 → K×4             │               │
│  Softmax 分类   │    │  每类 4 维 bbox 偏移     │               │
└─────────────────┘    └─────────────────────────┘               │
    │                             │                              │
    ▼                             ▼                              │
 类别概率 p              各类回归输出 t = (tx,ty,tw,th)            │
 (含 background)         推理时取预测类的 4 维                     │
```

**VGG-16 的空间缩放因子**：5 次 stride=2 的 MaxPool → 特征图相对原图缩放 **1/16**。

### 3.2 RoI Pooling Layer（论文最核心创新）

#### 3.2.1 要解决什么问题

CNN 全连接层要求**固定长度输入**，但每个 proposal 大小不一。R-CNN 的做法是 warp 成 227×227（形变大、重复卷积）；Fast R-CNN 改为在**共享特征图**上对每个 RoI 做**自适应分 bin + max pool**。

#### 3.2.2 数学定义

给定 RoI（图像坐标 `(x1, y1, x2, y2)`），目标输出网格 `H_r × W_r`（VGG 默认 **7×7**）：

```
Step 1: 映射到特征图坐标（spatial_scale = 1/16）

  x1' = floor(x1 × spatial_scale)
  y1' = floor(y1 × spatial_scale)
  x2' = ceil(x2 × spatial_scale)
  y2' = ceil(y2 × spatial_scale)
  RoI 在特征图上宽 w' = x2' - x1'，高 h' = y2' - y1'

Step 2: 等分 RoI 为 H_r × W_r 个 bin（可能不等宽/高）

  bin_w = w' / W_r
  bin_h = h' / H_r

  第 (i, j) 个 bin 的边界（0-indexed）:
    x_start = x1' + ⌊i · bin_w⌋        x_end = x1' + ⌈(i+1) · bin_w⌉
    y_start = y1' + ⌊j · bin_h⌋        y_end = y1' + ⌈(j+1) · bin_h⌉

Step 3: 对每个 bin 做 max pooling → 1 个值/通道

  output(i, j, c) = max{ feature_map(y, x, c) | x∈bin_x, y∈bin_y }
```

#### 3.2.3 数值示例

```
原图 RoI: (112, 48, 352, 304)     宽 240, 高 256 像素
spatial_scale = 1/16
特征图 RoI: (7, 3, 22, 19)       宽 15, 高 16（特征图像素，已量化）

7×7 RoI Pooling:
  每个 bin 宽约 15/7 ≈ 2.14 像素，高约 16/7 ≈ 2.29 像素
  bin 边界取 floor/ceil → 相邻 bin 可能差 1 像素（量化误差）

输出: 7 × 7 × 512 = 25088 维 → flatten → 接 FC6
```

#### 3.2.4 RoI Pooling 示意

```
特征图 (H/16 × W/16)                某个 RoI 区域放大
┌─────────────────────────┐         ┌───┬───┬───┬───┬───┬───┬───┐
│  ┌──────┐               │         │max│max│max│max│max│max│max│ → 7×7
│  │ RoI  │   ...          │   →     ├───┼───┼───┼───┼───┼───┼───┤   每个 cell
│  └──────┘               │         │max│max│max│max│max│max│max│   对应该 bin
│                         │         ├───┼───┼───┼───┼───┼───┼───┤   内 max pool
└─────────────────────────┘         │ ... (7 rows) ...          │
                                    └───┴───┴───┴───┴───┴───┴───┘
```

#### 3.2.5 反向传播

RoI Pooling 的 backward 与 Max Pooling 相同：

```
每个 bin 的前向取 argmax 位置
反向时，梯度只回传给该 max 位置，其余为 0
→ 允许梯度从 FC 层一路传回 conv 特征图，进而更新 backbone
```

这正是 Fast R-CNN 能 **微调全部 conv 层**、而 SPP-net 当时做不到的关键。

#### 3.2.6 RoI Pooling 的局限（Mask R-CNN 的改进动机）

| 问题 | 原因 | 后续改进 |
|------|------|----------|
| **量化误差** | floor/ceil 把 RoI 边界 snap 到整数特征图像素 | RoI Align：双线性插值，不量化 |
| **两次量化** | 原图 RoI → 特征图坐标已量化一次 | 对定位敏感任务（mask）影响大 |
| **固定 7×7** | 小 RoI 的 bin 可能只有 1 像素 | 对检测尚可，分割不够 |

---

## 4. 双分支检测头

### 4.1 分类分支（Softmax）

```
输入: FC7 的 4096-d 向量
输出: p = (p_0, p_1, ..., p_K)

p_0     = background 概率
p_k     = 第 k 类物体概率 (k = 1..K)
K+1     = PASCAL VOC 为 21（20 类 + background）

损失: L_cls = -log(p_u)     u 为 RoI 的真实类别标签
```

**为何不再需要 SVM**：多任务联合训练下，Softmax 直接优化分类 log-loss，与检测目标一致；R-CNN 时代 SVM 更优是因为 CNN 微调与分类器分离训练。

### 4.2 回归分支（Class-specific Bbox Regression）

```
输入: 同上 4096-d
输出: 对每个类 k，预测 t^k = (t_x^k, t_y^k, t_w^k, t_h^k)

FC_bbox: 4096 → K×4（VOC: 4096 → 80）

训练: 仅对「类别为 u 且 u≥1」的 RoI，回归 t^u 与真值 v 之间的 Smooth L1
推理: 对预测类 k* = argmax(p_k)，取 t^k* 精修 proposal
```

**回归目标 v**（与 R-CNN 相同）：

```
给定 proposal (Px, Py, Pw, Ph) 与 GT (Gx, Gy, Gw, Gh):

v_x = (Gx - Px) / Pw
v_y = (Gy - Py) / Ph
v_w = log(Gw / Pw)
v_h = log(Gh / Ph)
```

推理反变换：

```
Ĝx = Pw · t_x + Px
Ĝy = Ph · t_y + Py
Ĝw = Pw · exp(t_w)
Ĝh = Ph · exp(t_h)
```

### 4.3 类专属 vs 类无关回归

论文默认 **class-specific**（每类独立 4 维）。也实验了 class-agnostic（所有类共享 4 维）：

| 模式 | 参数量 | mAP | 说明 |
|------|--------|-----|------|
| Class-specific | K×4 | 更高（默认） | 各类尺度/形状差异大 |
| Class-agnostic | 4 | 略低 | 参数少，某些场景够用 |

---

## 5. 多任务损失函数

### 5.1 总损失

对每个采样的 RoI，定义多任务损失：

```
L(p, u, t^u, v) = L_cls(p, u) + λ · [u ≥ 1] · L_loc(t^u, v)

其中:
  p      = Softmax 输出的 (K+1) 维概率
  u      = 真实类标签（0 = background, 1..K = 物体类）
  t^u    = 类 u 对应的回归预测（u=0 时不算回归）
  v      = 回归真值
  [u≥1]  = 指示函数，仅前景 RoI 参与回归
  λ      = 1（论文默认，平衡分类与回归）
```

### 5.2 分类损失 L_cls

```
L_cls(p, u) = -log(p_u)

标准 Softmax 交叉熵
background (u=0) 和各类物体都参与
```

### 5.3 定位损失 L_loc（Smooth L1）

```
L_loc(t^u, v) = Σ_{i∈{x,y,w,h}} smooth_L1(t_i^u - v_i)

smooth_L1(x) = {  0.5·x²,   if |x| < 1
               {  |x| - 0.5, otherwise
```

| 损失 | 特点 | 相对 L2 |
|------|------|---------|
| **L2 (x²)** | 对大误差惩罚过重 | 异常值主导梯度 |
| **L1 (|x|)** | 鲁棒，但 x=0 不可微 | 优化不稳定 |
| **Smooth L1** | 小误差 L2、大误差 L1 | **Fast R-CNN 默认**，Faster/Mask R-CNN 沿用 |

### 5.4 损失计算流程（单个 RoI）

```
RoI r 匹配到 GT 类 u=5（dog），IoU=0.7

分类: L_cls = -log(p_5)                    ← 所有采样 RoI 都算
回归: L_loc = smooth_L1(t^5 - v)           ← 仅 u≥1 且匹配到 GT 的正样本算
                                              负样本 (u=0) 只贡献 L_cls
```

---

## 6. 训练流程与样本处理

### 6.1 训练数据构成

```
每张训练图:
  1. 预计算的 SS proposals（离线存储，与 R-CNN 相同）
  2. 所有 GT 框也作为 proposal 加入（保证每个物体有高 IoU 正样本）
```

### 6.2 RoI 标签分配

对每个 RoI，计算与**所有 GT** 的最大 IoU：

| 最大 IoU | 类别标签 u | 回归目标 | 角色 |
|----------|-----------|----------|------|
| ≥ **0.5** | 匹配 GT 的类（1..K） | 有（相对匹配 GT 算 v） | **正样本** |
| ∈ [**0.1**, 0.5) | **0**（background） | 无 | **负样本** |
| < **0.1** | — | — | **忽略**（不参与训练） |

> 与 R-CNN 微调阶段一致；**不再有 SVM 阶段的 0.3~0.5 丢弃区间**。

**多 GT 匹配**：若 RoI 与多个 GT IoU 均 ≥ 0.5，赋给 **IoU 最大** 的那个 GT 类。

### 6.3 Mini-batch 采样

```
每个 SGD iteration:
  128 个 RoI = 从 2 张图各采 64 个（随机选图）

  正样本: 32 个（25%）  ← IoU ≥ 0.5
  负样本: 96 个（75%）  ← 0.1 ≤ IoU < 0.5

  若某图正样本不足，用该图全部正样本 + 随机负样本凑满 64
```

**为何从 2 张图采样**：单图 RoI 数可能上千，只取 64 个可加速；2 图 × 64 = 128 与 R-CNN 一致。

### 6.4 微调策略

| 超参 | 值 |
|------|-----|
| Backbone | VGG-16（ImageNet 预训练）或 AlexNet |
| 优化器 | SGD，momentum = 0.9，weight decay = 0.0005 |
| 初始学习率 | 0.001（预训练层的 1/10） |
| 学习率 schedule | 30k iter 后 ×0.1，再 30k iter |
| Batch size | 128 RoIs（2 images × 64 RoIs） |
| 迭代总数 | 60k（VOC） / 400k（VOC+VOC2012 trainval） |
| 数据增强 | 随机水平翻转（概率 0.5） |
| 输入 | **整图**，多尺度训练可选（短边 600 px 为常见设置） |

**分层学习率**（与 R-CNN 类似思想）：

```
conv 层:   lr × 1
fc 层:     lr × 1（新初始化的 cls/reg head 从随机权重开始，靠较大 lr 快速收敛）
```

### 6.5 单阶段训练流程

```
┌─────────────────────────────────────────────────────────────┐
│ Phase A: ImageNet 预训练 VGG-16（直接用现成权重）             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase B: 检测联合微调（唯一训练阶段）                          │
│                                                              │
│  for each iteration:                                         │
│    1. 随机取 2 张图 + 它们的 proposals                        │
│    2. 整图前向 → 特征图                                       │
│    3. 采样 128 RoI → RoI Pooling → FC → cls + bbox          │
│    4. 计算 L = L_cls + λ·L_loc                               │
│    5. 反向传播（梯度更新 conv + fc + 两个 head）               │
└─────────────────────────────────────────────────────────────┘
```

不再需要：离线 fc7 缓存、SVM 训练、独立回归器训练。

---

## 7. 推理流程（逐步）

```
输入: 测试图 I
────────────────────────────────────────────────────────────

1. Region Proposal
   RoIs = selective_search(I)          # ~2000 boxes，CPU ~2s

2. 整图前向（一次）
   feature_map = ConvBackbone(I)       # H/16 × W/16 × 512

3. 对每个 RoI r_i:
   feat_i = RoIPool(feature_map, r_i)   # 7×7×512
   (p, t) = Heads(FC(feat_i))          # p: K+1 维, t: K×4 维

4. 对每个 RoI:
   k* = argmax_{k≥1} p_k               # 选最高分的物体类（跳过 background）
   if p_{k*} < threshold: discard      # 可选置信度阈值

5. 对每个保留 RoI:
   refined_box = apply_delta(r_i, t^{k*})   # 用预测类的回归输出

6. 对每个类 k 独立 NMS:
   IoU 阈值 = 0.3（VOC 标准）

7. 输出: {(refined_box, class, score)}
────────────────────────────────────────────────────────────

耗时分布（VGG-16, GPU）:
  Selective Search:  ~2.0 s
  CNN + RoI Pooling: ~0.22 s  ← Fast R-CNN 的贡献
  总计:              ~2.3 s/图（R-CNN 约 47 s）
```

**瓶颈转移**：Conv 不再是瓶颈，**Selective Search 成为主要耗时** → 直接催生 Faster R-CNN 的 RPN。

---

## 8. 消融实验与关键结论

### 8.1 各组件贡献

| 配置 | VOC 2007 mAP | 说明 |
|------|--------------|------|
| R-CNN (VGG-16) | 66.0 | 基准 |
| Fast R-CNN，不微调 conv | ~63 | 类似 SPP-net 局限 |
| Fast R-CNN，微调 conv1-5 | **66.9** | 全层微调最优 |
| + bbox regression | 必要（+若干 mAP） | 与 R-CNN 结论一致 |
| Softmax vs SVM | Softmax 略优或持平 | 联合训练下 Softmax 足够 |
| 多任务 vs 分阶段 | 多任务更好、更简单 | 端到端优势 |

### 8.2 微调哪些层

| 微调范围 | mAP | 训练时间 |
|----------|-----|----------|
| 仅 FC | 较低 | 快 |
| conv4 + conv5 + FC | 高 | 中等 |
| **全部 conv + FC** | **最高** | 较慢但可接受 |

**结论**：RoI Pooling 使 conv 层可微，**微调 conv 层对 mAP 至关重要**（+3~4 mAP vs 只微调 FC）。

### 8.3 Proposal 来源对比

| Proposal 方法 | mAP (VOC07) | 速度 |
|---------------|-------------|------|
| Selective Search | 66.9 | SS 慢 |
| EdgeBoxes | 66.0 | 略快 |

Fast R-CNN **与 proposal 算法解耦**——换 proposal 不需改网络结构。

### 8.4 单尺度 vs 多尺度训练/测试

- **训练**：随机缩放短边（如 600px ± 范围）可小幅提升
- **测试**：多尺度测试 + 平均可 +1~2 mAP，代价是线性变慢

---

## 9. 仍存在的局限 → Faster R-CNN 的动机

| 局限 | 表现 | Faster R-CNN 解法 |
|------|------|-------------------|
| **Proposal 仍是外部 SS** | CPU 2s/图，占测试时间 ~87% | Region Proposal Network（RPN） |
| **Proposal 与检测网络分离** | 无法共享 proposal 阶段的 conv 特征 | RPN 与检测头共享 backbone |
| **非真正端到端** | SS 不可微，无法联合优化 proposal | RPN 可微，统一训练 |
| **RoI Pooling 量化** | 定位有亚像素误差 | RoI Align（Mask R-CNN） |

```
Fast R-CNN 解决了 R-CNN 的「重复卷积 + 多阶段训练」
但没有解决「慢 proposal」→ 这是 Faster R-CNN 的任务
```

---

## 10. 演进脉络（R-CNN → Fast → Faster）

```
R-CNN (2014)
  每 region warp + 独立 CNN，SVM 分类，四阶段训练
      ↓  痛点: 慢、训练繁琐、存特征
SPP-net (2014)
  整图 conv + SPP，但只能微调 FC
      ↓  痛点: conv 无法有效微调
Fast R-CNN (2015)  ← 本文
  整图 conv + RoI Pooling + Softmax/Bbox 多任务，单阶段训练
      ↓  痛点: SS proposal 慢且分离
Faster R-CNN (2016)
  RPN 生成 proposal，与 Fast R-CNN 检测头共享特征
      ↓
Mask R-CNN (2017)
  RoI Align + mask 分支
```

---

## 11. 精读备忘：易混淆点

### 11.1 Fast R-CNN vs R-CNN 样本阈值

| 阶段 | 正样本 | 负样本 | 忽略 |
|------|--------|--------|------|
| **R-CNN 微调** | IoU ≥ 0.5 | [0.1, 0.5) | < 0.1 |
| **R-CNN SVM** | IoU ≥ 0.5 | ≤ 0.3 | (0.3, 0.5) |
| **Fast R-CNN** | IoU ≥ 0.5 | [0.1, 0.5) | < 0.1 |

Fast R-CNN 只有一套阈值，**无 SVM 阶段**。

### 11.2 「整图输入」不等于「不做 resize」

- Fast R-CNN **不对每个 RoI warp 成 227×227**；
- 但训练/测试时通常会把整图 **等比缩放**（如短边 600px），RoI 坐标同步缩放；
- RoI Pooling 在**特征图**上完成「变固定长」，而非在原图像素上 warp。

### 11.3 回归分支对负样本不算 loss

```
u = 0 (background):  L = L_cls only
u ≥ 1 (foreground):  L = L_cls + λ · L_loc
```

负样本只教网络「这是背景」，不教「背景框该怎么移动」。

### 11.4 推理时回归用哪个类的输出

```
训练: 只对真实类 u 的 t^u 算回归 loss
推理: 用 Softmax 预测类 k* 对应的 t^{k*} 去精修框

若分类错了，回归也会用错类的参数 → 类专属回归的代价
```

### 11.5 RoI Pooling vs RoI Align

| | Fast R-CNN | Mask R-CNN |
|--|-----------|------------|
| 池化 | 量化 + max pool | 双线性插值，无量化 |
| 任务 | 检测够用 | 分割需要更准定位 |

---

## 12. 参考资料

- 原论文：[Fast R-CNN](https://arxiv.org/abs/1504.08083)（ICCV 2015）
- 前置：[R-CNN](./RCNN.md)（CVPR 2014）
- 并行参考：[SPP-net](https://arxiv.org/abs/1406.4729)（ECCV 2014）
- 后续：**Faster R-CNN**（2016）—— 建议继续精读
- 官方实现：[rbgirshick/fast-rcnn](https://github.com/rbgirshick/fast-rcnn)

---

*文档版本：初稿 | 对应论文 ICCV 2015 Fast R-CNN*
