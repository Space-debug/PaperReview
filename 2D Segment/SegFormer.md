# SegFormer

> 本文档用于整理 SegFormer（Simple and Efficient Design for Semantic Segmentation with Transformers）论文精读笔记。  
> 重点：**MiT（Mix Transformer）编码器**、**序列缩减高效自注意力**、**Mix-FFN**、**全 MLP 轻量解码器**，相对 SETR / Swin+UPerNet 的**效率—精度**权衡。

> 相关：[OneFormer.md](./OneFormer.md) · [SAM.md](./SAM.md)（提示式基础模型）· [Mask2Former.md](./Mask2Former.md)（mask 分类）· [../backbone/SwinTransformer.md](../backbone/SwinTransformer.md)（层次 Transformer）· [HRNet.md](./HRNet.md) · [PSPNet.md](./PSPNet.md) · [DeepLab.md](./DeepLab.md)

---

## 1. 概览

### 1.1 基本信息

| 项目 | 内容 |
|------|------|
| 论文 | SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers |
| 作者/机构 | Enze Xie, Wenhai Wang, Zhiding Yu, …（NVIDIA、港中文 等） |
| 发表 | **NeurIPS 2021**（arXiv 2021.05） |
| 任务 | **语义分割**（ADE20K、Cityscapes 等） |
| 代码 | [NVlabs/SegFormer](https://github.com/NVlabs/SegFormer)（MMSegmentation 集成） |

### 1.2 核心思想（一句话）

**用层次化 MiT 编码器（重叠 patch + 序列缩减注意力 + 带 3×3 卷积的 Mix-FFN）提取 1/4~1/32 多尺度 token 特征，再用仅含 1×1 MLP 的轻量头把四档特征对齐到 1/4 分辨率 concat 融合后分类，无需 ASPP/UPerNet/位置编码，以小参数量达到接近重型 Swin+UPerNet 的 mIoU。**

| 对比 | SETR | Swin + UPerNet | **SegFormer** |
|------|------|----------------|---------------|
| 编码器 | ViT 单尺度 | Swin 层次窗口 | **MiT 层次 + SR Attention** |
| 解码器 | 重、上采样复杂 | **UPerNet/FPN 重** | **All-MLP（极轻）** |
| 位置编码 | 需要 | 相对偏置 | **不需要**（Mix-FFN 隐式） |
| ADE20K mIoU | ~48 | **53.9**（Swin-L） | **51.0**（B5，**参数量约 1/4**） |

### 1.3 整体流水线

```
Input: H×W×3
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  MiT Encoder（Mix Vision Transformer）B0~B5                   │
│  Stage1 → 1/4   Stage2 → 1/8   Stage3 → 1/16   Stage4 → 1/32 │
│  每 stage: Overlap Patch Embed + Transformer×N               │
│    · Efficient Self-Attention（SR 降 K/V 序列长）              │
│    · Mix-FFN（Linear + 3×3 DWConv + GELU + Linear）          │
└──────────────────────────────────────────────────────────────┘
    │  F₁ F₂ F₃ F₄  多尺度特征
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Lightweight All-MLP Decoder                                │
│  各 Fi → 1×1 MLP 到 256 维 → 上采样到 F₁ 尺寸（1/4）         │
│  → Concat → 1×1 融合 → 分类 → 4× 上采样到 H×W              │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
训练: 逐像素 CE（ignore index）；AdamW + poly LR
```

### 1.4 精度与效率（ADE20K val, mIoU / Params）

| 模型 | mIoU | 参数量（约） | GFLOPs（512×512） |
|------|------|--------------|-------------------|
| PSPNet-ResNet-101 | 41.96 | 65M | — |
| HRNetV2-W48 + [OCR](./OCRNet.md) | ~48 | 70M+ | — |
| SETR-MLA | 48.05 | 310M | — |
| Swin-L + UPerNet | **53.9** | **310M** | — |
| **SegFormer-B0** | 37.4 | **3.7M** | 8.4 |
| **SegFormer-B2** | 46.0 | 27.5M | — |
| **SegFormer-B5** | **51.0** | **84.7M** |  — |
| SegFormer-B5 + MS | **51.7** | 84.7M | — |

#### Cityscapes（val mIoU）

| 模型 | mIoU |
|------|------|
| HRNetV2-W48 + [OCR](./OCRNet.md) | 84.9 |
| **SegFormer-B5** | **84.0** |
| Swin-L + UPerNet | 84.9 |

- **卖点**：B5 用 **约 1/4 参数量** 逼近 Swin-L+UPerNet 的 ADE20K 精度；推理更快、结构更简单

---

## 2. 动机：Transformer 分割要「省」什么

### 2.1 SETR / ViT 分割的代价

```
SETR (ViT-L):
  · 固定 patch 16×16 → 序列长 N = HW/256
  · **全局 self-attention** O(N²) → 高分辨率图爆炸
  · 需 **重型 decoder**（PUP、MLA）把 token 拉回像素
  · 参数量 300M+，训练慢
```

### 2.2 Swin + UPerNet 的代价

```
Swin 提供层次特征（好）
但分割常用 **UPerNet / FPN**:
  · PPM + FPN + 多重 conv 融合
  · 解码器参数与计算仍可观
  · 完整 pipeline 复杂，调参组件多

见 [SwinTransformer.md](../backbone/SwinTransformer.md)
```

### 2.3 SegFormer 的设计目标

```
1. 编码器:  **比全局 ViT 省算力**（序列缩减注意力）
2. 编码器:  **比纯 MLP 缺局部** 的问题 → Mix-FFN 加 3×3
3. 解码器:  **比 UPerNet 极简** → 只要 MLP + 上采样
4. 去掉:    绝对/相对 **位置编码**（依赖 overlap patch + conv FFN）
```

---

## 3. MiT 编码器（Mix Vision Transformer）

### 3.1 四阶段层次结构（与 CNN/Swin 对齐）

| Stage | 输出 stride | 作用 | B5 深度（blocks）示意 |
|-------|-------------|------|----------------------|
| 1 | **1/4** | 高分辨率 token，边界细节 | 3 |
| 2 | 1/8 | 中层语义 | 6 |
| 3 | 1/16 | 上下文 | 40 |
| 4 | 1/32 | 全局语境 | 3 |

```
相邻 stage 之间:
  **Overlap Patch Embedding**（重叠卷积切块）
  → 下采样并升维，进入下一 stage Transformer 堆叠
```

### 3.2 Overlapping Patch Embedding

```
非 ViT 式非重叠 16×16:
  使用 **kernel > stride** 的 Conv2d 分 patch
  例: k=7, s=4, p=3  → 相邻 patch **有重叠**

好处:
  · 减少块效应、利于 **边界连续**
  · 类似 CNN 的局部平滑，弥补无显式 PE
```

### 3.3 Efficient Self-Attention（序列缩减 SR）

**问题**：Stage1 仍有 (H/4)×(W/4) 个 token，全局 attention 仍贵。

**做法**：对 K、V 做 **空间降采样**（Sequence Reduction）：

```
输入 token 序列 X ∈ R^{N×C},  N = h×w

Q = Linear(X)                    — 保持 **全分辨率** N
X' = SR(X):  Conv2d stride=R 重排为序列  — 长度 N' = N / R²
K = Linear(X'),  V = Linear(X')

Attention(Q, K, V)  复杂度 ~ O(N · N') 而非 O(N²)
```

**各 stage 典型缩减比 R**（B5 配置示意）：

| Stage | SR ratio R |
|-------|------------|
| 1 | 8 |
| 2 | 4 |
| 3 | 2 |
| 4 | 1（不再缩减） |

```
直觉:
  · 浅层: N 大，**强缩减** K/V
  · 深层: N 已小，R=1 等价标准 attention
  · Q 始终高分辨率 → **定位信息不丢**
```

### 3.4 Mix-FFN（替代标准 ViT MLP）

```
标准 ViT FFN:
  Linear → GELU → Linear   （无 2D 局部性）

Mix-FFN:
  Linear(扩维 C → C_ffn)
  → **reshape 到 2D**
  → **3×3 Depthwise Conv**（same padding）  ← 关键
  → GELU
  → Linear(压回 C)

作用:
  ① 注入 **局部空间归纳偏置**（邻域交互）
  ② **隐式位置编码** → 论文 **去掉** sin/cos 与 learned PE
  ③ 与 SR Attention 互补（全局稀疏 + 局部卷积）
```

### 3.5 MiT-B0 ~ B5

```
B0~B5:  各 stage **block 数、通道宽、head 数** 缩放
  · B0:  极轻（3.7M），移动端/快速实验
  · B5:  最深（stage3 达 40 blocks），精度最高

编码器可 **单独** 用于分类/检测预训练（后续工作多）
```

---

## 4. 轻量 All-MLP 解码器

### 4.1 设计原则

```
**不用**:
  · ASPP / PPM（见 [DeepLab.md](./DeepLab.md)、[PSPNet.md](./PSPNet.md)）
  · UPerNet 式 heavy FPN
  · 额外 Transformer decoder

**只用**:
  1×1 Conv（= per-pixel MLP）+ 双线性上采样
```

### 4.2 逐步流程

```
编码器输出 F₁…F₄，空间尺寸分别为 H/4, H/8, H/16, H/32

Step 1:  对每个 Fi，1×1 Conv → 统一维度 C_dec（默认 **256**）

Step 2:  F₂,F₃,F₄ 双线性上采样到 **F₁ 的 H/4 × W/4**

Step 3:  Concat[F₁', F₂', F₃', F₄']  → 4×256 通道

Step 4:  1×1 Conv 融合 → 256

Step 5:  1×1 Conv → K 类 logits（仍在 1/4）

Step 6:  4× 双线性上采样 → H×W
```

**为何足够**：

```
MiT 已在 **四层** 注入多尺度:
  decoder 仅负责 **对齐 + 线性融合**
  语义语境主要靠 encoder，无需 PPM 式再池化一遍
```

### 4.3 与 HRNet / U-Net 对比

| | HRNet | U-Net | SegFormer |
|--|-------|-------|-----------|
| 融合 | 全程双向 fusion | 每层 concat skip | **末端一次 concat** |
| 高分辨率 | 专用 1/4 分支 | 对称解码 | **F₁ 为最高档** |
| 复杂度 | 高 | 中 | **低（decoder 极简）** |

见 [HRNet.md](./HRNet.md)、[UNet.md](./UNet.md)。

---

## 5. 损失函数与训练

### 5.1 损失

```
标准 **Cross-Entropy**，忽略 label = 0（ADE20K 背景/ignore 依 MMSeg 配置）

**无** aux loss（相对 PSPNet 深监督更简）
**无** CRF
```

### 5.2 优化（论文 / MMSeg 典型）

| 超参 | 值 |
|------|-----|
| 优化器 | **AdamW** |
| LR | 6e-5（B0）~ 6e-5（B5），依 scale 微调 |
| LR policy | **Poly** decay |
| Weight decay | 0.01 |
| Batch | 16（4 GPU×4）等 |
| Crop | 512×512（ADE20K） |
| 迭代 | 160k |
| 增强 | 随机缩放、裁剪、翻转、光度扰动 |

### 5.3 推理

```
单尺度:  直接 forward + 上采样
多尺度:  {0.5, 0.75, 1.0, 1.25, 1.5} + flip → 概率平均（+0.7 mIoU 量级）

导出:  常合并 encoder+decoder 为单网络
```

---

## 6. 与 SETR / Swin+UPerNet / CNN 系列对比

### 6.1 Transformer 分割路线

```
ViT/SETR:     单尺度 token + 重 decoder
Swin+UPerNet:  强 encoder + **重 neck/decoder** → SOTA 精度，重
SegFormer:     高效 encoder + **最轻 decoder** → 接近精度，省参数量
```

### 6.2 与 CNN 多尺度模块

```
PSPNet PPM / DeepLab ASPP:
  在 **单一深层特征** 上做多尺度池化/空洞

SegFormer:
  **四个 stage 本身** 已是多尺度
  decoder 不再叠 PPM/ASPP
```

### 6.3 总表

| 维度 | DeepLabv3+ | PSPNet | HRNet | **SegFormer-B5** |
|------|------------|--------|-------|------------------|
| 骨干类型 | CNN+空洞 | CNN | CNN 并行 | **Transformer** |
| 全局建模 | ASPP | PPM | 多分支 fusion | **Attention** |
| 参数量 | 中 | 中 | 大 | **中（< Swin-L pipeline）** |
| ADE20K | — | 41.96 | ~48 OCR | **51.0** |
| 部署友好 | 中 | 中 | 差 | **B0/B1 优** |

---

## 7. 消融与论文结论（要点）

| 设计 | 结论 |
|------|------|
| 去掉 SR（全局 attention） | 算力↑，精度略变，不划算 |
| 去掉 Mix-FFN 中 3×3 | mIoU **明显下降**（局部性丧失） |
| 去掉 overlap patch | 边界变差 |
| 加 sin PE | **几乎无增益** → 验证「不需要 PE」 |
| Heavy decoder 换 UPerNet | 参数↑，精度略升但违背「简单」目标 |
| All-MLP vs 单用 F₄ | 多尺度融合 **必要** |

---

## 8. 精读备忘：易混淆点

### 8.1 SegFormer ≠ Swin

```
Swin:     窗口 attention + shift，**无** SR on K/V
MiT:      **全图 attention 但 K/V 序列缩减** + Mix-FFN
二者都可出 4 档特征，但 **block 内算子不同**
```

### 8.2 「无位置编码」不是无位置信息

```
Overlap patch + 3×3 DWConv FFN → **隐式** 编码相对位置
```

### 8.3 All-MLP 解码器里仍有「上采样」

```
MLP 指 **1×1 卷积** 做通道变换
  空间对齐靠 **双线性插值**，非学习 deconv
```

### 8.4 B5 的 stage3 很深（40 blocks）

```
主要计算量在 **encoder stage3**（1/16 分辨率）
  不是 decoder；读 profiling 时注意
```

### 8.5 MMSeg 配置名

```
实际训练常用 `mit_b5` + `SegformerHead`
  论文名 SegFormer = MiT + Lightweight decoder 组合
```

### 8.6 与 Mask2Former 关系

```
SegFormer:  **语义分割** 专用高效 pipeline
Mask2Former:  mask 分类（分任务训）          → [Mask2Former.md](./Mask2Former.md)
OneFormer:    单权重三任务联合训            → [OneFormer.md](./OneFormer.md)
  可视为 SegFormer 之后的 **任务统一** 方向，非简单升级
```

---

## 9. 后续与变体

| 方向 | 代表 |
|------|------|
| 更强 MiT | MiT v3、EfficientFormer 融合思想 |
| 实时 | SegFormer-B0/B1、TensorRT 部署 |
| 检测 | MiT 作 backbone + DETR 系 |
| 统一分割 | [Mask2Former](./Mask2Former.md)、[OneFormer](./OneFormer.md) |
| 多模态 | 扩展至遥感、医学（需重新训 MiT） |

---

## 10. 语义分割脉络（SegFormer 位置）

```
CNN 时代:  FCN → U-Net → DeepLab → PSPNet → HRNet  → 各 .md
Transformer 分割:
  SETR (2021)        ViT + 重 decoder
  Swin+UPerNet       层次 ViT + 重 neck
  SegFormer (2021)   **MiT + All-MLP**              ← 本文
  Mask2Former        mask 分类（分任务训）        → [Mask2Former.md](./Mask2Former.md)
  OneFormer          单模型三任务联合训          → [OneFormer.md](./OneFormer.md)
```

---

## 11. 参考资料

- 原论文：[SegFormer](https://arxiv.org/abs/2105.15203)（NeurIPS 2021）
- 代码：[NVlabs/SegFormer](https://github.com/NVlabs/SegFormer)、[MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- 骨干对比：[SwinTransformer.md](../backbone/SwinTransformer.md)、[ViT.md](../backbone/ViT.md)
- 对比：[HRNet.md](./HRNet.md)、[PSPNet.md](./PSPNet.md)、[DeepLab.md](./DeepLab.md)
- 前置 Transformer 分割：[SETR](https://arxiv.org/abs/2101.06175)

---

*文档版本：初稿 | 对应论文 NeurIPS 2021 SegFormer*
