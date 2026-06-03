# PSPNet（Pyramid Scene Parsing Network）

> 本文档用于整理 PSPNet 论文精读笔记。  
> 重点：**金字塔池化模块（PPM）**、**全局—局部场景语境**、**深监督（Deep Supervision）**，以及与 [DeepLab.md](./DeepLab.md) **ASPP**、[FCN.md](./FCN.md) 的多尺度设计差异。

> 相关：[OCRNet.md](./OCRNet.md)（类级物体语境）· [HRNet.md](./HRNet.md)（高分辨率并行融合）· [DeepLab.md](./DeepLab.md)（空洞卷积 / ASPP）· [FCN.md](./FCN.md) · [UNet.md](./UNet.md)

---

## 1. 概览

### 1.1 基本信息

| 项目 | 内容 |
|------|------|
| 论文 | Pyramid Scene Parsing Network |
| 作者/机构 | Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, Jiaya Jia（香港中文大学 等） |
| 发表 | **CVPR 2017** |
| 任务 | **场景解析**（Scene Parsing）= 密集语义分割；强调 **ADE20K** 等复杂场景 |
| 代码 | [hszhao/PSPNet](https://github.com/hszhao/PSPNet)（Caffe / PyTorch 社区复现多） |

### 1.2 核心思想（一句话）

**在 ResNet 提取的深层特征上并行多档自适应平均池化（1×1、2×2、3×3、6×6 网格），将不同「区域粒度」的上下文压成向量再双线性上采样回原分辨率，与原特征 concat 后预测像素类，使网络显式利用「飞机停在跑道上」这类全局场景对局部像素的约束。**

| 对比 | DeepLab ASPP | **PSPNet PPM** |
|------|--------------|----------------|
| 多尺度机制 | **空洞卷积** 不同 rate | **自适应池化** 不同 bin 数 |
| 全局语境 | v3 加 image-level pool 分支 | **1×1 bin** 即整图全局 |
| 骨干改造 | atrous 保持 OS | ResNet + **dilated** OS=8 |
| 训练技巧 | 多尺度推理 | PPM + **深监督** 辅助 loss |

### 1.3 整体流水线

```
Input: H×W×3
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Backbone: ResNet-101（conv1~4 常规，conv5 **空洞化**）        │
│  输出 stride = 8 的特征 F:  H/8 × W/8 × 2048                 │
└──────────────────────────────────────────────────────────────┘
    │
    ├──────────────────────────────┐
    │                              │ 深监督分支（训练时）
    ▼                              ▼
┌─────────────────────────┐   conv4 特征 → 1×1 conv → 上采样
│  Pyramid Pooling Module │   → 辅助 loss L_aux（权重 0.4）
│  (PPM) on F             │
└─────────────────────────┘
    │
    ▼
  原 F 与 4 档池化分支 concat → 3×3 conv 降维 → 1×1 → K 类
    │
    ▼
  双线性上采样 ×8 → H×W logits
    │
    ▼
训练: L = L_main + λ·L_aux（λ=0.4）
推理: 多尺度 + 左右翻转 平均概率（可选）
```

### 1.4 精度参考

#### PASCAL VOC 2012（test，mIOU）

| 模型 | mIOU |
|------|------|
| FCN-8s | 67.2（+CRF） |
| DeepLabv2-ASPP | 79.7 |
| DeepLabv3-ResNet-101 | 85.7 |
| **PSPNet-ResNet-101** | **85.4** |

- VOC 上与 DeepLabv3 同级；论文卖点之一是 **ADE20K** 大幅领先

#### ADE20K（150 类场景，val mIOU）

| 模型 | mIOU | pixel acc |
|------|------|-----------|
| FCN-8s | 29.39 | 71.83 |
| DilatedNet | 32.65 | — |
| **PSPNet-ResNet-101** | **41.96** | **80.64** |
| **PSPNet-ResNet-269** | **44.94** | **81.39** |

- **ADE20K Challenge 2016 冠军**；复杂室内/室外大类集上证明 PPM 价值

#### Cityscapes（test, mIOU）

| 模型 | mIOU |
|------|------|
| **PSPNet-ResNet-101** | **78.4** |

---

## 2. 动机：场景解析需要全局语境

### 2.1 局部看不够的 failure case

```
场景图特点（ADE20K）:
  · 150 类，物体共现复杂（沙发+墙+画）
  · **同类局部纹理可对应不同语义**（水面 = 河 / 泳池 / 海）

仅依赖局部感受野:
  · 把「跑道上的灰色块」误判为路面而非飞机部件
  · 把「室内的床」一角误判为沙发

需要:
  **全局类别先验** + **区域级上下文** 约束像素标签
```

### 2.2 现有方法的上下文不足

| 方法 | 上下文来源 | 局限 |
|------|------------|------|
| FCN | 逐层卷积 RF 线性增大 | 最深仍有限；skip 偏局部 |
| DeepLabv1/v2 | 空洞 + ASPP | 2017 前 ASPP 未含显式全局 bin |
| 图像分类再分割 | 整图标签弱监督 | 非端到端像素级 |

### 2.3 PSP 的直觉

```
人看场景:
  先感知 **整图场景**（机场 / 卧室）
  再细化 **区域**（天空、跑道、机身）
  最后 **像素** 归属

PPM 用 1×1、2×2、3×3、6×6 池化网格 **显式模拟** 由整到局部的金字塔语境
```

---

## 3. 金字塔池化模块（PPM）详解

### 3.1 放置位置

```
在 ResNet **最后一层卷积特征** F 上操作:
  · 空间尺寸: H'×W'（通常 OS=8，H'=H/8）
  · 通道: 2048（ResNet-101）

**不在** 每层都建金字塔（与 FPN 不同）— 仅 **单点、深层** 融合多尺度语境
```

### 3.2 四档并行分支（默认）

对同一输入 F，**并行** 四路：

```
Level 1: AdaptiveAvgPool2d(output_size=(1, 1))
         → 1×1 conv（降维到 C/n）→ BN → ReLU
         → 双线性上采样到 H'×W'

Level 2: AdaptiveAvgPool2d((2, 2))
         → conv → BN → ReLU → upsample → H'×W'

Level 3: AdaptiveAvgPool2d((3, 3))
         → 同上

Level 4: AdaptiveAvgPool2d((6, 6))
         → 同上
```

**融合**：

```
PPM_out = Concat[ F ; branch_1×1 ; branch_2×2 ; branch_3×3 ; branch_6×6 ]
         → 3×3 conv + BN + ReLU（融合、降通道）
         → 1×1 conv → K 类 logits（仍在 H'×W'）
         → 最后 **×8 双线性** 到原图
```

### 3.3 各档物理含义

| Bin 尺寸 | 覆盖语义 | 作用 |
|----------|----------|------|
| **1×1** | 整图一条向量 | **全局场景类**（天空主导→蓝多） |
| **2×2** | 四分块 | 粗略布局（上天下地） |
| **3×3** | 九宫格 | 中等区域（床/桌/墙区） |
| **6×6** | 36 块 | 更细区域语境，仍比像素粗 |

```
与 ASPP 对比（见 [DeepLab.md](./DeepLab.md) §3.2）:

ASPP:  同一像素位置上，用不同 **空洞率** 采样邻域 → 连续 RF
PPM:   先 **池化整块区域** 再铺满 → 离散 **几何尺度** 的区域统计

DeepLabv3 的 image-level 分支 ≈ PPM 的 1×1 分支；
PPM 额外提供 2×2、3×3、6×6 的中间粒度 — **论文主要创新点**
```

### 3.4 通道与参数

```
每路池化后常用 1×1 conv 压到 **512 维**（论文实现细节）
Concat 后通道 ≈ 2048 + 4×512 → 3×3 conv 压回 512 再分类

参数量相对 ASPP 可比；计算主要在 **大分辨率 H'×W'** 上的 concat conv
```

### 3.5 示意图（逻辑）

```
                    F (H'×W'×2048)
                         │
     ┌───────┬───────┬───┴───┬───────┐
     ▼       ▼       ▼       ▼       ▼
   1×1     2×2     3×3     6×6      F
   pool    pool    pool    pool    (identity)
     │       │       │       │       │
     └───────┴───────┴───┬───┴───────┘
                         ▼
                    Concat → Conv → cls
```

---

## 4. 骨干网络：Dilated ResNet

### 4.1 为何用空洞 ResNet（与 DeepLab 同族）

```
若 ResNet .pool 到 OS=32:
  · 特征图太小，PPM 的 6×6 bin 意义弱
  · 边界定位差

PSPNet 采用 **output stride = 8**:
  · res4:  stride 1 + dilation 2  （保持 1/16）
  · res5:  首个 block stride 1，内部 3×3 dilation=4
  · 最终 F 为 **1/8** 输入尺寸
```

### 4.2 ResNet-269 变体

```
论文在 ADE20K 上使用 **更深的 ResNet-269**（更多 bottleneck）
  → 44.94% mIOU，说明场景解析仍受益于 **容量**
  · 训练更慢，显存更大
```

### 4.3 与 FCN-8s 的 stride

```
FCN-8s:  skip 从 pool3 拉到 1/8
PSPNet:  无 skip，靠 **PPM + OS=8 单路** 到全分辨率

→ PSP 更依赖 **强全局模块** 而非浅层特征拼接
（边界细节略弱于 FCN-8s / DeepLabv3+ decoder，靠后处理与 MS 推理补）
```

---

## 5. 深监督（Deep Supervision）

### 5.1 动机

```
ResNet-101 很深 → 仅最顶层 loss，**低层/中层** 梯度弱、收敛慢
场景类多（ADE20K 150 类）→ 需要中间层也学判别特征
```

### 5.2 做法

```
在 **res4 输出**（OS=16 的特征 G）上接:
  · 1×1 conv → K 类
  · 上采样到 H×W（双线性）
  · 与 GT 算 **辅助交叉熵** L_aux

总损失:
  L = L_main + λ · L_aux
  论文 λ = **0.4**
```

### 5.3 训练 vs 推理

```
训练:   主分支 + 辅助分支 **同时** 反传
推理:   **丢弃** 辅助头，仅用最深层 + PPM 输出

与 FCN 深监督、Hourglass 中间监督 **同族 trick**
```

---

## 6. 损失函数与训练

### 6.1 主损失

```
逐像素 **Softmax Cross-Entropy**
忽略 label = 0 或数据集定义的 void（ADE20K 用 0 作 ignore 等，以实现为准）

无 CRF 后处理（与 DeepLabv3 路线一致）
```

### 6.2 优化与 schedule

| 超参 | PASCAL VOC / 通用 | ADE20K（论文） |
|------|-------------------|----------------|
| 优化器 | SGD momentum **0.9** | 同左 |
| 初始 LR | **1e-2**（batch 16） | **1e-2** |
| LR 策略 | **Poly**: lr = base×(1 - iter/max_iter)^0.9 | 同左 |
| Weight decay | 1e-4 | 1e-4 |
| 迭代 | 60k~100k | 150k 量级 |
| Crop | 473×473 随机 | 473×473 |
| Batch | 16（8 GPU×2 等） | 较大 batch 有利 BN |

### 6.3 数据增强

```
随机缩放（0.5 ~ 2.0）
随机裁剪到固定尺寸
左右翻转
颜色扰动（数据集相关）

ADE20K:  150 类长尾，增强对泛化重要
```

### 6.4 推理

```
1. **Multi-scale**: 如 {0.5, 0.75, 1, 1.25, 1.5} 缩放输入
2. **Flip**: 原图 + 水平翻转，对 **softmax 概率图** 平均
3. 上采样到原图尺寸 → argmax

VOC 85.4% / ADE20K 41.96% 均含 MS+flip（与 DeepLab 评测协议可比）
```

---

## 7. 与 DeepLab / FCN / U-Net 对比

### 7.1 多尺度模块正交性

```
实践中可 **ASPP + PPM 并联**（后续工作常见）
论文时代二选一对比:

  DeepLabv3:  atrous + 全局池化分支 → 逼近 PPM 的 1×1
  PSPNet:     显式 4 档 **几何金字塔** → ADE20K 优势明显
```

### 7.2 总表

| 维度 | FCN-8s | U-Net | DeepLabv3+ | **PSPNet** |
|------|--------|-------|------------|------------|
| 浅层细节 | skip 相加 | concat | 单层 decoder | **弱**（无 skip） |
| 全局上下文 | 弱 | 中 | ASPP+全局 | **PPM 专精** |
| 场景大类集 | 一般 | 医学为主 | 强 | **ADE20K SOTA** |
| 深监督 | 无 | 无 | 可选 | **有（λ=0.4）** |
| CRF | 可选 | 少 | 无 | 无 |

### 7.3 何时优先读 PSPNet

```
· 做 **场景解析 / ADE20K / 室内** 大类共现
· 需要 **可解释的多粒度区域** 先验（bin 可视化）
· 与 ASPP 对照设计 **自定义 neck**（Panoptic FPN 等）
```

---

## 8. 消融与论文结论

| 去掉 PPM | ADE20K mIOU 明显下降（约 4~6+ 点量级） |
| 去掉某一 bin | 1×1 最关键；缺 6×6 区域语境变差 |
| 无深监督 | 收敛慢、精度略降 |
| OS=16 vs 8 | OS=8 更好，与 DeepLab 经验一致 |

```
结论:
  **全局—区域金字塔池化** 对复杂场景分割有效；
  与空洞多尺度 **互补** 而非完全替代。
```

---

## 9. 精读备忘：易混淆点

### 9.1 PSP ≠ Pyramid Pooling in FPN

```
FPN:  多层特征 **横向** 融合，服务检测/分割多尺度实例
PPM:  **单层深层特征** 上多档 pool，服务 **语义语境**
```

### 9.2 Adaptive Average Pooling

```
输入 H'×W' 任意，指定 output (k,k) 即可
  → 每 bin 覆盖原图一块 **自适应区域**
  与固定 kernel 的 average pool 不同
```

### 9.3 PPM concat 后必须再 conv

```
直接 concat 通道暴涨 → 需 3×3 conv 融合，否则参数与过拟合压力大
```

### 9.4 深监督头推理时不用

```
部署只导出 **main branch**；aux head 仅训练存在
```

### 9.5 1×1 bin 与 Global Average Pooling

```
PPM 的 1×1 分支 = 对 F 做 global average pool + 1×1 conv
  与 DeepLabv3 ASPP 最后一支 **同功能**
  PSP 的差异在 **2×2、3×3、6×6** 中间档
```

### 9.6 ResNet-269 不是标准 torchvision 权重

```
复现时多用 ResNet-101；269 需专用预训练与更长训练
```

---

## 10. 后续影响

| 方向 | 关系 |
|------|------|
| **UPerNet / Semantic FPN** | 将 PPM 思想接入多尺度特征金字塔 |
| **DeepLabv3 ASPP+image pool** | 收敛「全局分支」设计 |
| **OCNet / CCNet** | 用 attention 替代/增强区域上下文 |
| **SegFormer / Mask2Former** | Transformer 分割 → [SegFormer.md](./SegFormer.md)、[Mask2Former.md](./Mask2Former.md) |

---

## 11. 语义分割脉络（PSPNet 位置）

```
FCN (2015)           → [FCN.md](./FCN.md)
DeepLab v2 ASPP (2016) → [DeepLab.md](./DeepLab.md)
PSPNet (2017)        **PPM + 深监督**              ← 本文
DeepLab v3/v3+ (2017–18)
HRNet                高分辨率并行融合          → [HRNet.md](./HRNet.md)
OCRNet               类级物体语境              → [OCRNet.md](./OCRNet.md)
SegFormer            MiT + All-MLP 解码器      → [SegFormer.md](./SegFormer.md)
Mask2Former          mask 分类统一分割         → [Mask2Former.md](./Mask2Former.md)
SAM                  提示式基础模型            → [SAM.md](./SAM.md)
OneFormer            单模型三任务              → [OneFormer.md](./OneFormer.md)
```

---

## 12. 参考资料

- 原论文：[Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105)（CVPR 2017）
- 实现：[hszhao/PSPNet](https://github.com/hszhao/PSPNet)
- 数据集：[ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)、PASCAL VOC、Cityscapes
- 对比：[DeepLab.md](./DeepLab.md)（ASPP）、[FCN.md](./FCN.md)
- 后续：[UPerNet](https://arxiv.org/abs/1807.10221)（统一 perceptual parsing）

---

*文档版本：初稿 | 对应论文 CVPR 2017 PSPNet*
