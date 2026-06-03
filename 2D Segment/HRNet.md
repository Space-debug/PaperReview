# HRNet（High-Resolution Network）

> 本文档用于整理 HRNet（Deep High-Resolution Representation Learning for Visual Recognition）及分割常用变体 **HRNetV2 / HRNet-OCR** 精读笔记。  
> 重点：**全程维持高分辨率特征流**、**多分辨率并行子网 + 反复跨尺度融合**，相对 [FCN.md](./FCN.md) / [PSPNet.md](./PSPNet.md)「先压低再抬高」路线的结构差异。

> 相关：[OCRNet.md](./OCRNet.md)（HRNet+OCR 物体语境模块）· [PSPNet.md](./PSPNet.md)（单点 PPM）· [DeepLab.md](./DeepLab.md)（ASPP / OS=8）· [UNet.md](./UNet.md)（编解码 skip）

---

## 1. 概览

### 1.1 基本信息

| 项目 | 内容 |
|------|------|
| 论文 | Deep High-Resolution Representation Learning for Visual Recognition |
| 作者/机构 | Jingdong Wang, Ke Sun, Tianheng Cheng, …（微软亚洲研究院 MSRA） |
| 发表 | **CVPR 2019**（arXiv 2018.12） |
| 任务 | **人体姿态估计**、**语义分割**、**图像分类**（统一骨干） |
| 代码 | [HRNet/HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation)、[HRNet/HRNet-Human-Pose-Estimation](https://github.com/HRNet/HRNet-Human-Pose-Estimation) |

### 1.2 核心思想（一句话）

**不再把高分辨率特征仅留在浅层、语义压到最低层再强行上采样，而是自始至终保留一条「全分辨率」卷积分支，并并行多条低分辨率分支提取语义，通过模块内**反复**高↔低双向融合，使高分辨率流在全程都注入多尺度语境。**

| 对比 | ResNet / PSPNet / DeepLab | **HRNet** |
|------|---------------------------|-----------|
| 高分辨率特征 | 浅层有，**深层丢失** | **全程维持** 1/4 输入分辨率流 |
| 多尺度 | 单尺度深特征 + ASPP/PPM | **并行多分支** + 每层融合 |
| 融合时机 | 末端一次性 | **每个 stage 内多次** exchange |
| 分割头 | 1/8 或 1/32 再上采 | 默认 **1/4 stride** 聚合 |

### 1.3 整体流水线（语义分割 HRNetV2）

```
Input: H×W×3
    │
    ▼
Stem: 2× stride-2 conv3×3 → 特征 stride = 4（1/4 分辨率）
    │
    ▼
Stage 1: 单分支 ResNet-style bottleneck × N（仍在 1/4）
    │
    ▼
Stage 2: 分裂为 2 个分辨率分支（1/4, 1/8）→ **Fusion Modules** × M
    │
    ▼
Stage 3: 3 分支（1/4, 1/8, 1/16）→ Fusion × M
    │
    ▼
Stage 4: 4 分支（1/4, 1/8, 1/16, 1/32）→ Fusion × M
    │
    ▼
分割头（HRNetV2）:
  各分支 → 上采样到 1/4 → **Concat** → 1×1 conv → K 类
  → 4× 双线性上采样 → H×W
    │
    ▼
（可选 HRNet-OCR）: 对象上下文模块进一步聚合语义
```

### 1.4 精度参考

#### Cityscapes（val / test, mIoU）

| 模型 | mIoU |
|------|------|
| PSPNet-ResNet-101 | 78.4（test） |
| DeepLabv3+ | ~82 量级 |
| **HRNetV2-W48** | **81.1** |
| **HRNetV2-W48 + OCR** | **84.9**（val） / **82.3**（test 报告） |

#### PASCAL VOC 2012（test, mIoU）

| 模型 | mIoU |
|------|------|
| DeepLabv3+ | 89.0 |
| **HRNetV2-W48** | 87.4 |
| **HRNetV2-W48 + OCR** | **87.6+**（论文/后续报告） |

#### COCO 人体姿态（AP）

| 模型 | AP |
|------|-----|
| CPN ResNet | 72.1 |
| **HRNet-W32** | **75.5** |
| **HRNet-W48** | **76.4** |

- 姿态任务上 HRNet 是论文 **主战场**；分割上 **HRNet-OCR** 把 Cityscapes 推到当时 SOTA

---

## 2. 动机：高分辨率表征为何不能只靠上采样

### 2.1 经典骨干的「分辨率漏斗」

```
ResNet / VGG 用于分割（FCN、PSPNet、DeepLab）:

  1/4 → 1/8 → 1/16 → 1/32
           ↑
  语义最强，但 **空间最粗**

恢复手段:
  · 双线性 / 转置卷积上采样
  · skip connection（FCN、U-Net、DeepLabv3+）
  · ASPP / PPM 在 **低分辨率** 图上做多尺度

问题:
  高分辨率细节主要在 **浅层** 出现一次
  深层网络继续 stack 后，**高分辨率语义流被中断**
  → 上采样只能「猜」边界，小物体、薄结构仍弱
```

### 2.2 HRNet 的范式转换

```
旧范式:  High-Res ──pool──→ Low-Res ──upsample──→ High-Res 预测
新范式:  High-Res 分支 **一直活着**，Low-Res 分支并行提供语境
         二者 **每个 module 都互相交换信息**

类比:
  不是「先模糊再锐化」
  而是「高清轨 + 语义轨 **持续混音**」
```

### 2.3 与 U-Net skip 的差异

| | U-Net | HRNet |
|--|-------|-------|
| 高分辨率信息 | 编码器 **一层** 传给 decoder | **专用分支** 贯穿全网络 |
| 融合次数 | 每层 decode 一次 | **每个 fusion module** 都跨尺度 |
| 低分辨率角色 | 仅解码路径 | **并行常驻** 子网 |

见 [UNet.md](./UNet.md)。

---

## 3. 网络结构详解

### 3.1 Stem 与 Stage 1

```
Stem:
  Conv 3×3, stride 2  →  H/2
  Conv 3×3, stride 2  →  H/4
  BN + ReLU

Stage 1（单分支）:
  4 个 Bottleneck（与 ResNet bottleneck 同构，base width W）
  输出:  **高分辨率流** X₁，stride = 4
```

### 3.2 逐级增加并行分支

| Stage | 分支数 | 相对输入分辨率 | 典型通道（W48） |
|-------|--------|----------------|-----------------|
| 2 | 2 | 1/4, **1/8** | 48, 96 |
| 3 | 3 | 1/4, 1/8, **1/16** | 48, 96, 192 |
| 4 | 4 | 1/4, 1/8, 1/16, **1/32** | 48, 96, 192, 384 |

**Transition（阶段切换）**：

```
Stage1 → Stage2:
  · 分支1: 接 Stage1 输出，接若干 Fusion Module
  · 分支2: 从分支1 经 **stride=2 的 3×3 conv** 新建，分辨率减半

Stage2 → Stage3 / Stage3 → Stage4:
  · 保留已有分支
  · 从 **最低分辨率分支** 再 stride=2 分出更低一支
```

### 3.3 Fusion Module（核心算子）

每个 stage 含 **M 个** 重复 Fusion Module（如 4 个）。  
设当前有 `B` 条分支，分辨率从低到高索引为 `0 … B-1`（或实现中反过来，逻辑一致即可）。

对第 `i` 条分支，输出为 **所有分支 j 对 i 的贡献之和**：

```
y_i = Σ_j  f(x_j → resolution_i)

f 的规则:

① j = i（同分辨率）:
     x_j 经 **4 个 Residual Block**（Basic/Bottleneck）

② j < i（j 比 i **更高分辨率**，要降到 i）:
     x_j 经 **stride=2 的 3×3 conv** 重复 (i-j) 次下采样
     → 1×1 conv 对齐通道

③ j > i（j 比 i **更低分辨率**，要升到 i）:
     x_j 经 **1×1 conv** 升通道
     → **双线性上采样** 到 i 的尺寸
```

**要点**：

```
· **高→低**: 卷积下采样（带语义）
· **低→高**: 上采样 + 1×1（注入全局/区域语境到高分辨率流）
· 每个 y_i 是 **多源求和**，不是 U-Net 式单次 concat

→ 高分辨率分支在 **每一层** 都被低分辨率更新
→ 低分辨率分支也持续接收高分辨率细节，避免语义漂移
```

### 3.4 示意图（Stage 4 四分支）

```
分辨率:  1/4 ─────── 1/8 ─────── 1/16 ────── 1/32
          │           │            │            │
          └───── Fusion Module（双向箭头）──────┘
                    × 重复 M 次

高分辨率轨 (1/4)  ←── 不断接收来自 1/8、1/16、1/32 的上采样语义
低分辨率轨      ←── 接收来自 1/4 的下采样细节
```

### 3.5 宽度命名：W18 / W32 / W48

```
W 表示 **最高分辨率分支** 的基础通道宽度
  · W18: 轻量
  · W32: 姿态/检测常用
  · W48: 分割 SOTA 配置，显存大

各低分辨率分支通道按倍数递增（实现表见官方 config）
```

---

## 4. HRNetV2 分割头

### 4.1 与 HRNetV1 分类头的区别

```
HRNetV1（分类）:
  各分支上采样到 1/4 → concat → 1×1 conv → **Global Average Pool** → FC

HRNetV2（分割）:
  各分支上采样到 **最高分辨率（1/4）**
  → Concat 全部通道
  → 1×1 conv → K 类
  → **4× 双线性** 到原图（output stride = 4）

不再做 GAP，保留空间维度
```

### 4.2 输出步长 OS=4 的含义

```
比 DeepLab OS=8、PSPNet OS=8 **更细** 的原生预测网格
  → 边界与小物体通常更好
  → 显存与计算 ↑（高分辨率流全程维护的代价）
```

### 4.3 可选轻量头

```
部分实现: 仅用最 **高** 分辨率分支 + 1×1 cls
  → 更快，精度略降
官方语义分割默认 **多分支 concat** 头
```

---

## 5. HRNet-OCR（分割增强）

> 物体语境模块 **OCRNet** 专篇见 **[OCRNet.md](./OCRNet.md)**；本节仅保留与 HRNet 的衔接。

```
组合:  HRNetV2-W48（OS=4） + OCR 头 → **HRNet+OCR**
效果:  Cityscapes val **~84.9** mIoU；test 提交 **84.5%**（+ SegFix，ECCV 2020 榜首）

相对纯 HRNetV2-W48（81.6 test）:
  OCR 在 **高分辨率特征** 上做 **按类软区域聚合**，与 PPM/ASPP 叠 HRNet 反而掉点形成对比

三步摘要:  粗分割 M_k → 区域向量 f_k → 像素–区域注意力得 OCR → 与 x_i 融合分类
```

---

## 6. 损失函数与训练

### 6.1 分割损失

```
标准 **逐像素 Cross-Entropy**（忽略 void label）
Cityscapes / VOC 与 contemporaries 相同

HRNet-OCR:  可能含 **辅助分割 loss** 训练中间粗分割（实现见官方 repo）
```

### 6.2 优化（Cityscapes / VOC 典型）

| 超参 | 典型值 |
|------|--------|
| 优化器 | SGD momentum 0.9 |
| LR | 0.01 → poly decay |
| Weight decay | 1e-4 |
| Batch | 8~16 crop |
| Crop | 512×1024（Cityscapes）、512×512（VOC） |
| 迭代 | 90k~110k |
| 增强 | 缩放、翻转、颜色 |

### 6.3 推理

```
多尺度 + 左右翻转（与 PSPNet / DeepLab 同协议）
单尺度已较强（高 OS=4）

姿态任务: 热图回归 + 高分辨率特征，无需上采样到全图分类
```

---

## 7. 与 FCN / PSPNet / DeepLab / U-Net 对比

| 维度 | PSPNet | DeepLabv3+ | U-Net | **HRNetV2** |
|------|--------|------------|-------|-------------|
| 高分辨率流 | 无（1/8） | decoder 补一层 | 编码器浅层 | **全程 1/4 分支** |
| 多尺度 | PPM 单点 | ASPP 单点 | U 形层级 | **并行 4 尺度反复融合** |
| 参数量/速度 | 中 | 中 | 中 | **高**（W48 重） |
| 边界/小物体 | 中 | 好 | 医学最好 | **自然图像优** |
| 姿态 / 关键点 | 不适配 | 不适配 | 少见 | **原生 SOTA 级** |

```
HRNet 优势:
  · **密集预测** 任务（姿态、分割、面部关键点）一站式骨干
  · 高分辨率表征不依赖「最后一次上采样」

HRNet 劣势:
  · 结构复杂，实现与调试成本高
  · 比 ResNet-101+ASPP **更耗显存**（多分支并行）
  · 检测 backbone 普及度低于 ResNet/Swin（仍可用于高分辨率 mask head）
```

---

## 8. 其它任务与变体

### 8.1 人体姿态（论文主实验之一）

```
输出:  K 个关键点热图，与最高分辨率分支同尺寸
优势:  亚像素定位需要 **持续高分辨率特征**，HRNet 比 Hourglass/CPN 更契合

COCO val AP 76.4（W48）→ 2019 前后 SOTA
```

### 8.2 图像分类

```
HRNetV1 头: concat 多尺度 → GAP → FC
ImageNet top-1 约 78%（W48）— 证明表征通用，但分类不如 EfficientNet 等专门设计
```

### 8.3 后续变体（读 HRNet 之后）

| 变体 | 说明 |
|------|------|
| **Lite-HRNet** | 条件通道权重、Shuffle，提速 |
| **HRFormer** | 将部分块换 Transformer |
| **HigherHRNet** | 多分辨率监督，自下而上姿态 |
| **Mask R-CNN + HRNet** | 作 backbone 实例分割 |

---

## 9. 消融与论文结论

| 设计 | 结论 |
|------|------|
| 仅单尺度高分辨率（无低分辨率分支） | 语义/context 不足 |
| 无高分辨率分支（类 ResNet） | 定位 AP / mIoU 降 |
| 减少 fusion 次数 | 多尺度交换不够，性能降 |
| W48 vs W18 | 宽度↑，分割与姿态均↑ |

```
核心 ablation 信息:
  **维持 + 反复融合** 高分辨率流，优于仅末端 skip 或 ASPP
```

---

## 10. 精读备忘：易混淆点

### 10.1 HRNet ≠ 只是「多尺度特征金字塔」

```
FPN/PSP:  **单路** 主干到底层，再外挂多尺度模块
HRNet:    **从 Stage2 起就分裂** 多路，且高分辨率路 **从不消失**
```

### 10.2 Fusion 是 **求和** 不是 concat

```
各分支贡献 **相加** 到目标分辨率（通道先对齐）
  与 U-Net decoder **concat** 不同
```

### 10.3 最高分辨率是 **1/4 输入**，不是全分辨率

```
仍需 4× 上采样得到 H×W 分割图
  但比 1/8、1/32 原生网格更细
```

### 10.4 HRNetV1 vs V2

```
V1:  分类头（GAP）
V2:  密集预测头（保留空间）
  读分割论文/代码默认 **V2**
```

### 10.5 OCR 不是 Optical Character Recognition

```
此处 OCR = **Object-Contextual Representations**
```

### 10.6 姿态与分割共用骨干

```
同一 HRNet body，换不同 **head** 即可
  理解结构时勿与「分割专用网络」割裂
```

---

## 11. 语义分割脉络（HRNet 位置）

```
FCN / U-Net / SegNet / DeepLab / PSPNet  → 见各文档
HRNet (2019)         高分辨率并行 + 跨尺度融合   ← 本文
HRNet-OCR (2020)     + 对象语境模块            → [OCRNet.md](./OCRNet.md)
Swin + UPerNet       Transformer 骨干 + 多尺度 neck
SegFormer            轻量 Transformer 分割  → [SegFormer.md](./SegFormer.md)
Mask2Former          统一 mask 分类三任务    → [Mask2Former.md](./Mask2Former.md)
OCRNet               类级区域语境（+HRNet）  → [OCRNet.md](./OCRNet.md)
SAM                  提示式基础模型          → [SAM.md](./SAM.md)
OneFormer            单权重三任务            → [OneFormer.md](./OneFormer.md)
```

---

## 12. 参考资料

- 原论文：[Deep High-Resolution Representation Learning](https://arxiv.org/abs/1908.07919)（CVPR 2019）
- 分割实现：[HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation)
- OCR 扩展：[OCRNet.md](./OCRNet.md)
- 姿态：[HRNet-Human-Pose-Estimation](https://github.com/HRNet/HRNet-Human-Pose-Estimation)
- 对比：[PSPNet.md](./PSPNet.md)、[DeepLab.md](./DeepLab.md)、[UNet.md](./UNet.md)

---

*文档版本：初稿 | 对应论文 CVPR 2019 HRNet + ECCV 2020 OCR 扩展*
