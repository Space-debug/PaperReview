# SSD（Single Shot MultiBox Detector）

> 本文档用于整理 SSD 论文精读笔记。  
> 重点：**多尺度特征图检测**、**Default Box（Anchor）生成与匹配**、**Hard Negative Mining**，以及相对两阶段 / YOLO v1 的**单阶段处理方式**。

---

## 1. 概览

### 1.1 基本信息

| 项目 | 内容 |
|------|------|
| 论文 | SSD: Single Shot MultiBox Detector |
| 作者/机构 | Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy 等（Google 等） |
| 发表 | ECCV 2016 |
| 任务 | **单阶段、多尺度、Anchor-based** 目标检测 |
| 代码 | [weiliu89/caffe:ssd](https://github.com/weiliu89/caffe/tree/ssd)、[amdegroot/ssd.pytorch](https://github.com/amdegroot/ssd.pytorch) |

### 1.2 核心思想（一句话）

**在 VGG 等 backbone 的多个卷积特征图上，为每个网格位置预设多组 Default Box，一次性预测「类别 + 框偏移」，用 Hard Negative Mining 平衡正负样本，无需 RPN、RoI Pooling 或第二阶段的分类网络。**

SSD 相对当时主流方法的定位：

| 对比对象 | SSD 的切入点 |
|----------|-------------|
| **R-CNN / Fast / Faster** | 去掉 proposal + RoI 二阶段 → **单次前向** |
| **YOLO v1** | 从 7×7 单尺度 → **6 层特征图多尺度** + 更多 default box |
| **DPM 等** | 用深度 CNN 特征替代手工特征 |

### 1.3 整体流水线

```
Input: 固定尺寸图像（300×300 或 512×512）
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Backbone: VGG-16（截断）+ Extra Conv Layers                   │
│  得到 6 个不同分辨率的特征图 {m₁×m₁, ..., m₆×m₆}              │
└──────────────────────────────────────────────────────────────┘
    │
    ▼  每个特征图每个位置 × k 个 default box
┌──────────────────────────────────────────────────────────────┐
│  检测头（每尺度小型卷积 predictor）                            │
│  输出: 类别分数 c + 4 维位置偏移 (Δcx, Δcy, Δw, Δh)          │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
  训练: 匹配 GT ↔ default box → L_conf + L_loc + Hard Negative Mining
  推理: 阈值过滤 → 类内 NMS → Top-K 检测
```

### 1.4 精度与速度参考（PASCAL VOC2007 test）

| 模型 | 输入 | mAP | FPS (GTX 980) | 默认框总数 |
|------|------|-----|---------------|-----------|
| Faster R-CNN (VGG) | ~1000短边 | 73.2 | ~7 | ~300 proposals |
| YOLO v1 | 448×448 | 63.4 | 45 | 7×7×2=98 |
| **SSD300** | 300×300 | **74.3** | **59** | **8732** |
| **SSD512** | 512×512 | **76.8** | 22 | **24564** |

- **SSD300**：速度与精度兼顾，实时性优于两阶段
- **SSD512**：更高 mAP，小目标更好，更慢

---

## 2. 动机：为何需要 SSD

### 2.1 两阶段检测的瓶颈

```
Faster R-CNN 流程:
  Backbone → RPN → ~300 RoI → RoI Pool → cls + reg

瓶颈:
  · RoI 相关操作（Align/Pool）逐区域处理，难极致加速
  · 两阶段训练与推理逻辑复杂
  · 对小物体：单尺度特征图（如 stride 16）上 RoI 像素少
```

### 2.2 YOLO v1 的不足（SSD 要解决的）

| 问题 | YOLO v1 | SSD |
|------|---------|-----|
| 检测尺度 | 仅 7×7 一层 | **6 层**特征图 |
| 每格预测数 | 2 box | 4~6 default box/格 |
| 小目标 | 差 | conv4_3 大特征图专门检小目标 |
| 背景误检 | 全图 grid 负样本极多 | **Hard Negative Mining** |
| 全连接 | 有 FC 层 | **全卷积** head |

### 2.3 SSD 的设计原则

```
1. 多尺度特征图  → 不同 receptive field 覆盖不同大小物体
2. Default Box   → 每位置多先验框，减轻回归难度
3. 单次前向      → 所有尺度、所有框同时预测
4. Hard Negative Mining → 控制易分负样本主导梯度
```

---

## 3. 网络结构详解（SSD300 + VGG-16）

### 3.1 整体架构图

```
Input 300×300×3
    │
    ▼
┌─────────────────────────────────────────┐
│  VGG-16（到 conv4_3，去掉 fc 大层）      │
│  conv4_3 → 38×38×512   ← 检测层 1（小目标）│
└─────────────────────────────────────────┘
    │
    ▼ MaxPool + 继续 conv → conv7
┌─────────────────────────────────────────┐
│  fc7 改为 conv7: 19×19×1024  ← 检测层 2  │
└─────────────────────────────────────────┘
    │
    ▼ Extra Feature Layers（降采样卷积堆叠）
┌─────────────────────────────────────────┐
│  conv8_2  → 10×10×512    ← 检测层 3      │
│  conv9_2  →  5×5×256     ← 检测层 4      │
│  conv10_2 →  3×3×256     ← 检测层 5      │
│  conv11_2 →  1×1×256     ← 检测层 6（大目标）│
└─────────────────────────────────────────┘
    │
    ▼ 每个检测层接 Predictor（3×3 conv 或 1×1）
  输出 tensor:  (H×W×k×(4+num_classes))  per layer
```

**Extra layers 典型构造**（逐步缩小特征图）：

```
conv8:  1×1 conv → 3×3 conv stride=2  (降采样)
conv9:  1×1 conv → 3×3 conv stride=2
conv10: 1×1 conv → 3×3 conv stride=1
conv11: 1×1 conv → 3×3 conv stride=1
```

### 3.2 六个检测层一览（SSD300）

| 层 | 特征图 | 步长≈ | 每格 default 数 k | 感受野角色 |
|----|--------|-------|------------------|-----------|
| **conv4_3** | 38×38 | 8 | **4** | 小物体 |
| **conv7** | 19×19 | 16 | **6** | 中小 |
| **conv8_2** | 10×10 | 32 | **6** | 中 |
| **conv9_2** | 5×5 | 64 | **6** | 中大 |
| **conv10_2** | 3×3 | ~100 | **4** | 大 |
| **conv11_2** | 1×1 | 300 | **4** | 全图级 |

**默认框总数**：

```
N = 38²×4 + 19²×6 + 10²×6 + 5²×6 + 3²×4 + 1²×4
  = 5776 + 2166 + 600 + 150 + 36 + 4
  = 8732
```

### 3.3 检测头（Prediction Module）

对每个检测层 `l`，共享或独立的小卷积：

```
输入:  feature_l  (m_l × m_l × C_l)

分支（可合并为一个 3×3 conv 输出多通道）:
  · 位置:  m_l × m_l × (4 × k_l)     → 每个 default box 4 维偏移
  · 分类:  m_l × m_l × ((C+1) × k_l)  → C 类物体 + background，每 box 一组

C = 20（VOC）→ 每 box 输出 4 + 21 = 25 维（实现上常拆成两 tensor）
```

**全卷积、无 RoI、无 FC** → 适合 GPU 并行，是「单 shot」的工程基础。

### 3.4 SSD512 与 SSD300 的差异

| 项目 | SSD300 | SSD512 |
|------|--------|--------|
| 输入 | 300×300 | 512×512 |
| 检测层数 | 6 | **7**（多一层 conv12_2: 2×2） |
| 默认框总数 | 8732 | **24564** |
| conv4_3 步长 | 8 | 更早层、更细（小目标更好） |
| mAP (VOC07) | 74.3 | **76.8** |
| FPS | 59 | 22 |

```
SSD512 思路: 更大输入 + 更多尺度层 → 小目标 AP 明显提升
```

---

## 4. Default Box（先验框）生成 — 关键细节

### 4.1 符号与尺度 schedule

设检测层索引 `l = 1..L`（SSD300 时 L=6），每层特征图边长 `m_l`。

**归一化尺度**（相对输入边长 1.0）：

```
s_l = s_min + (s_max - s_min) · (l - 1) / (L - 1)

SSD300 常用:  s_min = 0.15,  s_max = 0.9
SSD512 常用:  s_min = 0.10,  s_max = 0.95
```

**每层还使用「额外尺度」** 取相邻层几何平均，增加一组 default box：

```
s'_l = sqrt(s_l · s_{l+1})     用于扩展不同大小的框
```

### 4.2 宽高比（aspect ratio）

每层除 scale 外，设宽高比集合 `r ∈ {1, 2, 0.5, 3, 1/3, ...}`（实现按层不同）。

**单个 default box 宽高**（归一化）：

```
设当前尺度为 s，宽高比为 r:

  w_box = s · sqrt(r)
  h_box = s / sqrt(r)

例: s=0.5, r=2  → 宽长条框
    s=0.5, r=1  → 正方形
    s=0.5, r=0.5 → 高长条框
```

### 4.3 中心位置（网格映射）

特征图位置 `(i, j)`，`i = 0..m_l-1`，`j = 0..m_l-1`：

```
cx = (j + 0.5) / m_l
cy = (i + 0.5) / m_l

→ 每个 cell 中心均匀铺在 [0,1]×[0,1] 图像上
```

**SSD300 conv4_3 例子**（m=38）：

```
cell (0,0) 中心 ≈ (0.013, 0.013)
cell (19,19) 中心 ≈ (0.5, 0.5)
```

### 4.4 每层 default box 数量为何不同

```
conv4_3: 4 个  → 1 个 scale s_l + 3 个 ratio {1,2,0.5} 等组合
conv7 等:  6 个  → s_l 与 s'_l 各配 3 个 ratio → 2×3=6
conv10/11: 4 个  → 大物体层减少 ratio 种类，避免冗余
```

> 先验框**手工设计**（按数据集统计调节），后续 YOLOv2+ 常用 k-means 聚类 anchor；SSD 开创了「多尺度手工 anchor schedule」范式。

---

## 5. 训练：匹配策略（比 RPN 更「宽松」）

### 5.1 为何匹配规则是 SSD 的核心

每个 default box 需贴标签：**正（某类）** 或 **负（background）**。  
SSD 的匹配比 Faster R-CNN **更宽松** → 正样本多 → 有利于小目标、密集场景。

### 5.2 匹配算法（逐步）

对每个 GT 框 `g`、每个 default box `d`：

```
Step 1: 计算 Jaccard IoU(d, g)（即 IoU）

Step 2: 对每个 GT，选 IoU 最大的 default box 作为正样本
        （保证每个 GT 至少被 1 个 default 覆盖）

Step 3: 对所有 default box，若 IoU(d, g) ≥ 0.5（与某 GT）
        → 将该 d 标为正样本，类别 = 该 GT 的类

Step 4: 其余 default box → 负样本（background）
```

**与 Faster R-CNN RPN 对比**：

| | SSD default box | Faster R-CNN Anchor |
|--|-----------------|---------------------|
| 正样本条件 | IoU ≥ **0.5**（可多 GT 多对一） | IoU > 0.7 或 best anchor |
| 每 GT 保证 | best match + 所有 ≥0.5 | best + IoU>0.7 |
| 结果 | **正样本远多于 RPN** | 正样本更稀疏 |

```
示意: 一个大 GT 可能同时匹配 几十个 default box（都 ≥0.5）
→ 分类器反复见到「这类外观」，回归有多个监督信号
```

### 5.3 边框编码（与 Faster R-CNN 一致）

匹配到 GT `g=(g_x,g_y,g_w,g_h)` 的 default `d=(d_x,d_y,d_w,d_h)`，回归目标：

```
t_x = (g_x - d_x) / d_w
t_y = (g_y - d_y) / d_h
t_w = log(g_w / d_w)
t_h = log(g_h / d_h)
```

网络预测 `t̂`，Smooth L1 监督；推理时反变换到原图坐标。

---

## 6. 损失函数与 Hard Negative Mining

### 6.1 总损失

```
L(x, c, l, g) = (1/N) · ( L_conf(x, c) + α · L_loc(x, l, g) )

N   = 正样本 default box 数量（匹配到的个数）
α   = 1（论文默认，平衡 loc 与 conf）
```

**仅正样本参与 L_loc**；**L_conf 对正 + 挖掘后的负样本计算**。

### 6.2 定位损失 L_loc

```
L_loc = Σ_{i∈正样本} Σ_m smooth_L1( t_i^m - t̂_i^m )

smooth_L1(x) = { 0.5x²,  |x|<1
               { |x|-0.5, otherwise

与 Fast R-CNN / Faster R-CNN 相同
```

### 6.3 分类损失 L_conf

```
Softmax over (C+1) 类，包含 background

对每个参与训练的 default box i:
  L_conf,i = -log(p_i(c_i))

c_i = 正样本类别，或 0（background）
```

### 6.4 Hard Negative Mining（关键细节）

**问题**：8732 个框中绝大多数是易分负样本（天空、草地），若全算 loss，梯度被背景淹没。

**做法**：

```
1. 对所有 default box 前向，算分类 loss（或按 confidence loss 排序）
2. 正样本: 全部保留
3. 负样本: 按 loss 从高到低排序，取 Top-K 难负例
4. 使 负:正 ≈ 3:1（论文固定比例 neg:pos = 3:1）

即: 若正样本 30 个，则最多选 90 个 hardest negatives 参与 L_conf
```

**流程图**：

```
8732 boxes
    │
    ├─ 正样本 N_pos（全部用于 L_conf + L_loc）
    │
    └─ 负样本 ~8700+
           │
           ▼ 按 conf loss 排序，取 Top (3 × N_pos)
           难负例进入 L_conf（不进 L_loc）
```

> YOLO v1 用全图 grid 权重；SSD 显式 **挖掘难例**，是两阶段里 OHEM 思想在单阶段的落地。

### 6.5 训练超参（VOC，SSD300）

| 超参 | 值 |
|------|-----|
| 优化器 | SGD，momentum 0.9，weight decay 5e-4 |
| 学习率 | 10⁻³ → 10⁻⁴ → 10⁻⁵（step decay） |
| Batch size | 32（或 16，视 GPU） |
| 输入 | 300×300 |
| 初始化 | VGG-16 ImageNet；extra layers Xavier |
| α | 1 |
| 匹配 IoU | 0.5 |
| neg:pos | 3:1 |

---

## 7. 数据增强（对小目标很重要）

SSD 在 VOC/COCO 上强依赖 **数据增强**，论文与开源实现均包含：

### 7.1 随机裁剪（核心）

```
1. 从原图随机裁一块，要求与某 GT 的 IoU ∈ {0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0} 之一
   （随机选目标 IoU 下限，保证裁切块里仍有物体）
2. 裁切后 resize 到 300×300
3. 以 0.5 概率执行

作用: 人工制造「放大物体」→ 等效增加小目标训练样本
```

### 7.2 其它增强

```
· 随机水平翻转
· 颜色抖动（亮度、对比度、饱和度、色相）
· （可选）随机缩放原图再 crop
```

**不做增强时 SSD300 小目标 AP 明显下降** —— 与多尺度 head 设计配套使用。

---

## 8. 推理流程（逐步）

```
输入: 图像 → resize 到 300×300（保持均值归一化）
────────────────────────────────────────────────────────────

1. 一次前向
   6 层特征图 → 8732 组 (class_logits, loc_offsets)

2. 解码
   对每个 default box d:
     score_c = softmax(class_logits)_c
     box = Decode(d, loc_offsets)   # 反变换到 300×300 坐标

3. 过滤
   去掉 background 或 score < 0.01 的框
   （可对各类别保留 Top-400 候选）

4. 类内 NMS
   IoU 阈值 = 0.45（VOC 常用）
   每类独立 NMS

5. Top-K
   每张图最多保留 200 个检测（按 score 排序）

6. （可选）映射回原图尺寸

输出: {box, class, score}
────────────────────────────────────────────────────────────
SSD300 @ GTX980: ~59 FPS（含前后处理略有出入）
```

---

## 9. 与两阶段 / YOLO 的对比

### 9.1 SSD vs Faster R-CNN

| 维度 | Faster R-CNN | SSD |
|------|--------------|-----|
| 阶段 | 2（RPN + Head） | **1** |
| 候选 | ~300 RoI | **8000+ default box** |
| 区域特征 | RoIAlign 逐区域 | **无 RoI**，网格共享特征 |
| 小目标 | 依赖 FPN（后期） | **大特征图 conv4_3** 原生多尺度 |
| 速度 | ~5–15 fps | **~59 fps**（SSD300） |
| mAP VOC07 | 73.2 | **74.3** |

### 9.2 SSD vs YOLO v1

| 维度 | YOLO v1 | SSD |
|------|---------|-----|
| 网格 | 7×7 单层 | **6 层** 38×38…1×1 |
| 框/格 | 2 | **4~6** |
| 置信度 | objectness × class | **softmax 含 background** |
| 负样本 | 全 grid | **Hard Negative Mining** |
| mAP VOC07 | 63.4 | **74.3** |
| 速度 | 45 fps | 59 fps |

### 9.3 SSD 的局限（后续工作改进点）

```
· Default box 需按数据集手工调 scale/ratio
· 浅层（conv4_3）语义弱，小目标仍不如 FPN+两阶段
· 密集场景：一个 GT 匹配过多 default → 训练冗余
· 无显式 feature fusion（DSSD 用 deconv 融合弥补）
· 大输入 SSD512 慢，实时性下降
```

---

## 10. 扩展与后续影响

| 工作 | 相对 SSD 的改动 |
|------|----------------|
| **DSSD** | Deconv + skip 融合，增强语义 |
| **SSD + ResNet / MobileNet** | 更强/更轻 backbone |
| **RetinaNet** | FPN + focal loss，解决类不平衡（不用 hard neg mining） |
| **YOLOv2/v3** | k-means anchor + 多尺度，继承 SSD 多尺度思想 |

**历史地位**：

1. 确立 **「多尺度特征图 + 密集 default box + 单 shot」** 范式；
2. 证明单阶段可在 **精度上超过** 早期 Faster R-CNN（VOC）；
3. **Hard Negative Mining** 成为 anchor-based 单阶段标配直至 Focal Loss；
4. 直接影响 YOLO 系列与 RetinaNet 的设计。

---

## 11. 精读备忘：易混淆点

### 11.1 「Single Shot」指什么

```
不是「只检测一个物体」
而是: 一次 network forward 输出所有尺度所有框
      无 cascaded RoI stage、无第二遍全图 conv
```

### 11.2 正样本很多不是 bug

```
同一 GT 匹配多个 IoU≥0.5 的 default 是刻意设计
→ 增加监督密度，尤其利于学习不同形状的框
副作用: 训练略冗余，靠 mining 控负样本即可
```

### 11.3 L_loc 与 L_conf 参与的样本不同

```
L_loc: 仅正样本（匹配到 GT 的 default）
L_conf: 正样本 + 难负样本（3:1），不含全部 8000+ 负例
```

### 11.4 conv4_3 为何能检小目标

```
stride≈8，38×38 网格更密
+ 该层 default scale 小（s_min=0.15 附近）
→ 小物体在特征图上仍占足够多 cell / 匹配的 default
```

### 11.5 SSD 与 RetinaNet 的 anchor

```
SSD:     手工 scale schedule + 每层固定 ratio 组合
RetinaNet: FPN 每层 9 anchor（3 scale × 3 ratio），focal loss 替代 mining
```

### 11.6 分类是 softmax 不是 sigmoid

```
SSD 每 default box: (C+1) way softmax，含 background 类
YOLOv1: 类条件概率 × objectness
多类互斥假设（VOC 每框一类）→ 与检测任务一致
```

---

## 12. 单阶段检测脉络（SSD 位置）

```
YOLO v1 (2016)     7×7 grid，单尺度，快但 AP 低
    ↓
SSD (2016)         多尺度 feature map + default box + mining  ← 本文
    ↓
YOLOv2/v3 (2017)   anchor + 多尺度 + passthrough/FPN 思想
    ↓
RetinaNet (2017)   FPN + focal loss，超越两阶段 AP
    ↓
YOLOv5/v8、RT-DETR 等  工程化 / Transformer 检测
```

**与 `2D Detection/RCNN` 系列关系**：

```
RCNN 系列:  「proposal → RoI 特征 → 分类回归」两阶段高质量
SSD 路线:   「密集先验 + 单次卷积」实时单阶段

现代检测: 两阶段仍强（Cascade、Mask R-CNN）；单阶段靠 FPN、focal、大模型追平
```

---

## 13. 参考资料

- 原论文：[SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)（ECCV 2016）
- 对比：[Faster R-CNN](../RCNN/FasterRCNN.md)、[YOLOv5](../YOLO/YOLOv5.md)
- 改进：[DSSD](https://arxiv.org/abs/1701.06659)（Deconvolutional Single Shot Detector）
- 实现：[weiliu89/caffe:ssd](https://github.com/weiliu89/caffe/tree/ssd)

---

*文档版本：初稿 | 对应论文 ECCV 2016 SSD*
