# DeepLab 系列

> 本文档用于整理 Google DeepLab **v1 → v2 → v3 → v3+** 系列论文精读笔记。  
> 重点：**空洞（Atrous/Dilated）卷积**、**ASPP 多尺度上下文**、**输出步长（output stride）控制**、**Dense CRF 后处理（v1/v2）与轻量 decoder（v3+）**，以及与 [FCN.md](./FCN.md) / [UNet.md](./UNet.md) 的路线差异。

> 相关：[PSPNet.md](./PSPNet.md)（金字塔池化 PPM）· [FCN.md](./FCN.md) · [UNet.md](./UNet.md) · [SegNet.md](./SegNet.md)

---

## 0. 系列地图（先读此节）

| 版本 | 论文 / 年份 | 核心增量 | CRF | 代表 VOC2012 mIOU |
|------|-------------|----------|-----|-------------------|
| **v1** | [DeepLabv1](https://arxiv.org/abs/1412.7062) 2014/2015 | 空洞卷积 + LargeFOV | **必选** 精修边界 | LargeFOV **62.25%** |
| **v2** | [DeepLabv2](https://arxiv.org/abs/1606.00915) 2016/TPAMI | **ASPP** + ResNet 骨干 | 常用 | 79.7%（ResNet-101） |
| **v3** | [DeepLabv3](https://arxiv.org/abs/1706.05501) 2017 | 改进 ASPP、**去掉 CRF** 仍 SOTA | 不用 | 85.7%（JFT 预训） |
| **v3+** | [DeepLabv3+](https://arxiv.org/abs/1802.02611) 2018 | **Encoder-Decoder** + 深度可分离空洞卷积 | 不用 | **89.0%**（Xception） |

```
共同主线:
  分类 CNN 做密集预测时，不靠「再堆 pool」换感受野
  → 用 **atrous conv** 在较高分辨率特征图上拉大 RF
  → 用 **ASPP** 在同一层并行多尺度空洞采样
  → v3+ 再补 **浅层特征 decoder** 修边界

与 FCN:  FCN 用 skip+deconv 恢复分辨率；DeepLab 长期 **单尺度高分特征图 + 空洞**（v3+ 才加 decoder）
与 U-Net: U-Net concat 浅层；DeepLab v3+ decoder 只融合 **最浅一层**（stride 4）
```

---

## 1. 共享基础：为何需要空洞卷积

### 1.1 分类网直接做分割的三重矛盾

```
VGG / ResNet 用于分割时:

① 分辨率骤降
   多次 MaxPool / stride conv → 1/32 特征图
   → 边界定位糊、小物体消失

② 感受野不足（若过早停 pool）
   浅层 RF 小 → 分不清「局部纹理是猫还是草地上的猫」

③ 全连接或大核卷积的参数冗余
   DeepLabv1 指出: 7×7 等效大核可用 **级联空洞卷积** 参数更高效地实现
```

### 1.2 空洞（Atrous / Dilated）卷积定义

```
标准 3×3 conv:  采样间隔 = 1（连续邻域）

空洞率 rate = r 的 3×3 atrous conv:
  · 核仍是 3×3，但采样点之间插入 (r-1) 个「洞」
  · 有效感受野扩大，**特征图 H×W 不变**（stride=1, same padding）

例: rate=2 的 3×3 实际覆盖 5×5 原图邻域，输出分辨率不降
```

**与下采样的对比**：

| 手段 | 分辨率 | 感受野 |
|------|--------|--------|
| MaxPool / stride=2 | **降低** | 相对原图变大 |
| Atrous rate↑ | **保持** | 在同层 feature map 上变大 |

### 1.3 输出步长（Output Stride, OS）

```
OS = 输入图像尺寸 / 最终 score map 边长

OS=32:  快，定位差（FCN-32s 同类问题）
OS=16:  平衡，DeepLab 常用训练配置
OS=8:   边界更好，显存↑（需在骨干中把 stride 换空洞）

改法（ResNet）:
  block3 输出 stride 16 → 保持
  block4 原 stride=2 下采样 → 改为 **rate=2 空洞**，OS 保持 16
  若要 OS=8 → block4 rate=2，block5 再 rate=2，不再额外 pool
```

---

## 2. DeepLab v1 — 空洞卷积 + Dense CRF

### 2.1 基本信息

| 项目 | 内容 |
|------|------|
| 论文 | Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs |
| 作者 | Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, Alan L. Yuille（Google） |
| 发表 | arXiv 2014.12；ICLR 2015 |
| 骨干 | VGG-16（ImageNet 预训练） |

### 2.2 两条技术支柱

#### （1）前端：空洞卷积替换 FC + 控制 OS

**DeepLab-Conv**：

```
VGG 最后 pool 后特征 1/8（相对原图 OS=8 的一种配置）
  → 用 **atrous conv** 替代后续全连接
  → 保持空间网格，每点输出 K 类 score
```

**DeepLab-LargeFOV**（论文主模型名）：

```
在 score 层之前串联多层 **空洞卷积**，扩大感受野（Large Field-of-View）:

  例: 3×3 atrous, rate = 12（或 6,12,18,24 等多尺度组合实验）
  → 近似用大核看更大上下文，参数少于 giant FC

相对普通 DeepLab-Conv:
  LargeFOV 在 VOC 上 **mIOU 明显更高**（62.25 vs 约 59 量级）
```

#### （2）后端：全连接 CRF 精修

```
CNN 输出 coarse score map（即使 OS=8 仍偏糊）
  → 当作 unary potential

Dense CRF [Krahenbühl & Koltun]:
  · 能量 = unary + pairwise
  · pairwise: 双边高斯（空间近 + 颜色相似 → 边界处切断平滑）
  ·  mean-field 推理，约 10 次迭代

效果:
  · **物体边界** 显著变锐利
  · FCN-8s 65.9 + CRF 67.2；DeepLab-LargeFOV 62.25 **+ CRF 进一步提升**

局限:
  · CRF 与 CNN **分阶段**，端到端困难（后续 v3 证明可不要）
```

### 2.3 v1 流水线

```
Input
  → VGG encoder（空洞化改造）
  → K 类 score map（OS=8 等）
  → **双线性上采样** 到原图（可选）
  → Dense CRF
  → 最终分割
```

### 2.4 v1 精度（PASCAL VOC 2012 val）

| 模型 | mIOU |
|------|------|
| FCN-8s | 65.9 |
| **DeepLab-LargeFOV** | **62.25**（纯 CNN，与 FCN 同年对比表） |
| DeepLab-LargeFOV-CRF | **+CRF 更高** |

- v1 纯 CNN 略低于 FCN-8s，**+CRF** 后竞争力强，奠定「CNN + CRF」范式

---

## 3. DeepLab v2 — ASPP 与 ResNet 骨干

### 3.1 基本信息

| 项目 | 内容 |
|------|------|
| 论文 | 同上标题扩展；**LSUN / TPAMI 2016–2018** 常称 DeepLabv2 |
| 核心新增 | **Atrous Spatial Pyramid Pooling (ASPP)** |
| 骨干 | **ResNet-101**（替换 VGG） |

### 3.2 ASPP 结构（系列核心模块）

```
在 **同一层** 特征图（如 OS=16 的 1/16 图）上并行多分支:

  Branch 1: 1×1 conv
  Branch 2: 3×3 atrous, rate = 6
  Branch 3: 3×3 atrous, rate = 12
  Branch 4: 3×3 atrous, rate = 18
  （rates 随 OS 调整: OS=8 时常用 12,24,36）

  → 各分支输出 K 通道
  → **逐像素 concat**（沿 channel）
  → 1×1 conv 压回 K 类
  → 可选 BN + ReLU（v3 规范）

直觉:
  · 小 rate → 小感受野 → 细节/小物体
  · 大 rate → 大感受野 → 大物体/上下文
  · 1×1 → 局部外观
  → **多尺度融合在一层完成**，不必像 FPN 建塔
```

**与 PSPNet 金字塔池化的区别**：

| | ASPP | PSP（PSPNet） |
|--|------|----------------|
| 操作 | **并行空洞卷积** | 全局+多档 **自适应池化** 再上采样 |
| 尺度 | 连续 RF 变化 | 固定 bin 几何尺度 |
| 出处 | DeepLabv2 | [PSPNet 2017](./PSPNet.md) |

### 3.3 ResNet + Multi-Grid

```
ResNet block4 / block5 中:
  · 将 **stride=2 下采样** 改为 **atrous conv**（保持 OS）
  · **Multi-grid**: 块内各 3×3 层用不同 rate（如 {1,2,4} × base_rate）

目的: 在最深特征上继续加大 RF，而不把图压得更小
```

### 3.4 v2 训练与推理

```
训练:  随机缩放、裁剪；SGD；OS=16 常见
推理:  multi-scale（0.5, 0.75, 1.0, 1.25, 1.5）+ 水平翻转
       → 平均 softmax 概率
       → **Dense CRF**（v2 仍常用）

PASCAL VOC 2012:
  DeepLabv2-ASPP-ResNet-101: **79.7%** mIOU（大幅超过 v1）
```

---

## 4. DeepLab v3 — 重思考空洞与 ASPP

### 4.1 基本信息

| 项目 | 内容 |
|------|------|
| 论文 | Rethinking Atrous Convolution for Semantic Image Segmentation |
| 发表 | arXiv 2017.06 |
| 主张 | **去掉 CRF**；优化 ASPP 与 OS；级联空洞模块可选 |

### 4.2 三个问题与对策

#### （1）多网格（Multi-Grid）+ OS

```
在 ResNet block4 用 rates {1,2,4}（× multiplier）
block5 用更大 base rate

对比实验:
  OS=32 → 差
  OS=8  → 最好，但最慢
  OS=16 → 性价比最佳（训练默认）
```

#### （2）改进 ASPP

**v2 问题**：rate 极大时 3×3 atrous 退化为 **1×1**（只采到 padding 零），丢失多尺度意义。

**v3 ASPP（标准 5 分支）**：

```
1. 1×1 conv
2. 3×3 atrous rate=6
3. 3×3 atrous rate=12
4. 3×3 atrous rate=18
5. **Image-level features**:
     对顶层特征 **Global Average Pooling** → 1×1 conv → 向量
     → 双线性上采样到 H×W
     → 提供「整图语境」（类似 PSP 的全局分支）

concat → 1×1 conv → K 类
```

**Batch Norm 注意**：

```
ASPP 内 BN 在 **大 batch** 训练时稳定
论文讨论: eval 时 BN 统计与 crop 尺寸相关，需固定 eval 分辨率或使用 sync BN 等（工程细节）

最佳模型 **不用 CRF** 即超过 v2+CRF
```

#### （3）级联空洞模块（Cascade，可选）

```
不只 ASPP 并行，还可 **串行**:
  Block: 3×3 atrous → BN → ReLU，rate 逐层增大
  → 类似「先看清局部再扩大上下文」

与 ASPP 并联实验: ASPP 略优或相当，**ASPP 成默认**
```

### 4.3 v3 精度（VOC 2012 val, 无 CRF）

| 配置 | mIOU |
|------|------|
| DeepLabv3-ASPP-ResNet-101 OS=16 | 78.5% |
| + JFT-300M 预训练 | **85.7%** |
| OS=8 | 略优于 16 |

- **里程碑**：纯 CNN ASPP **无需 CRF** 达 SOTA，CRF 退出主线

---

## 5. DeepLab v3+ — 编码器-解码器 + 可分离空洞卷积

### 5.1 基本信息

| 项目 | 内容 |
|------|------|
| 论文 | Encoder-Decoder with Atrous Separable Convolution |
| 发表 | ECCV 2018 |
| 动机 | v3 高层语义强，但 **物体边界** 仍不如 FCN-8s / U-Net 细；加 **轻量 decoder** |

### 5.2 整体结构

```
┌─────────────────────────────────────────────────────────────┐
│  Encoder: DeepLabv3 骨干（ResNet / Xception / MobileNet）      │
│    · 修改 OS=16 或 4（MobileNet）                            │
│    · 顶层 ASPP 输出 rich 语义特征 F（1/16 或 1/4）             │
└─────────────────────────────────────────────────────────────┘
    │                                    │
    │ 浅层 low-level 特征 L              │ F
    │ （如 ResNet conv2 后，stride=4）    │
    ▼                                    ▼
┌──────────────────┐          ┌──────────────────────────────┐
│ 1×1 conv 压通道   │          │ ASPP on F → 4× 上采样到 1/4   │
│ L → 48 ch（典型） │          └──────────────────────────────┘
└────────┬─────────┘                      │
         │         **concat** [4×F_up ; L']
         ▼
    3×3 conv ×2（256→K 或逐步）
         ▼
    4× 双线性上采样 → **全分辨率** K 类 logits
```

**Decoder 设计原则**：

```
· 只融合 **一层** 浅特征（stride 4），避免 U-Net 式多层 skip 过重
· 浅层通道压到 **48**（论文 ablation: 16/32/48/64 中 48 最佳）
· 顶层语义仍靠 **ASPP + atrous encoder**
· 边界细节靠 **低层纹理 + 少量 conv**
```

### 5.3 深度可分离空洞卷积（Atrous Separable Conv）

```
标准 conv:  spatial 3×3 + channel mix 一步

Depthwise separable:
  1) Depthwise 3×3（每通道独立，可带 atrous rate）
  2) Pointwise 1×1（通道混合）

再对 depthwise 施加空洞 → **速度↑、参数↓**，精度略降可接受

Xception-65 / MobileNet 等轻量骨干上必备
```

### 5.4 Xception 与对齐（Alignment）

```
v3+ 推荐 **Aligned Xception**:
  · 修正原始 Xception 实现与论文不一致处
  · 更深层使用 separable + atrous

JFT 预训练 + MS + flip:
  DeepLabv3+-Xception-OS=16: **89.0%** VOC2012（test set 报告）
```

### 5.5 v3+ 消融要点

| 变体 | 结论 |
|------|------|
| 无 decoder（=v3） | 边界类 IoU 低 |
| + decoder | **+several mIOU**，尤其沿边界类 |
| 浅层通道 48 | 优于过大/过小 |
| separable conv | 快 20%+，mIOU 略降或持平（视骨干） |

---

## 6. 训练与推理（系列通用）

### 6.1 损失

```
逐像素 **Softmax Cross-Entropy**（忽略 255 border label）
可选 bootstrapping hard pixels（v2/v3 部分实验）

**无** U-Net 式距离加权（自然图像任务不同）
```

### 6.2 数据增强

```
随机缩放（0.5~2.0）
随机裁剪（如 513×513）
左右翻转
颜色抖动
```

### 6.3 推理技巧

```
1. **Multi-scale**: 多尺度输入，对 softmax 概率平均
2. **Left-right flip**: 原图+翻转图平均
3. **OS=8** 单尺度优于 OS=16（若算力允许）
4. v1/v2: + **CRF**；v3/v3+ 通常 **不用**

工程:
  TensorFlow 官方 [tensorflow/models/research/deeplab](https://github.com/tensorflow/models/tree/master/research/deeplab)
```

### 6.4 超参快照（ResNet-101, VOC）

| 超参 | 典型值 |
|------|--------|
| 优化器 | SGD momentum 0.9 |
| LR | 0.007（poly decay: (1-iter/max)^0.9） |
| Batch | 16 crop 513×513 |
| OS 训练 | 16；fine-tune OS=8 |
| 迭代 | 30k~80k |

---

## 7. 与 FCN / U-Net / SegNet 对比

| 维度 | FCN-8s | U-Net | SegNet | **DeepLabv3+** |
|------|--------|-------|--------|----------------|
| 分辨率恢复 | skip+deconv | concat skip | unpool | **双线性+浅层 decoder** |
| 多尺度语义 | 多层 skip | 仅 U 形层级 | 无 | **ASPP 并行** |
| 边界 | 较好 | 最好（医学） | 中等 | decoder 补强 |
| 后处理 CRF | 可选 | 少用 | 少用 | **废弃** |
| 自然图像 VOC | 65.9 | 非主战场 | 59.1 | **89.0** |
| 实时性 | 中 | 中 | 较好 | MobileNetv3+ 可近实时 |

```
DeepLab 强项:
  · 固定 backbone 上改 OS + ASPP，**模块感强**，易迁移到检测/其他任务
  · 大尺度预训练（JFT）+ MS 推理拉高上限

弱项:
  · 原生 v1–v3 **无** 多层 skip，细节一度弱于 FCN-8s（v3+ 缓解）
  · 训练 pipeline 重（多尺度、CRF 时代更繁）
```

---

## 8. 系列演进逻辑（精读主线）

```
v1:  证明 **atrous** 可在高分辨率特征上分割 + **CRF** 修边界
      ↓
v2:  **ASPP** 把多尺度并行化；ResNet 加深
      ↓
v3:  ASPP + **全局池化分支**；级联空洞可选；**CRF 退出**
      ↓
v3+: **Encoder-Decoder** + separable atrous；对齐 Xception；冲 89% mIOU
```

**与检测的交叉**：

```
空洞卷积 / ASPP 广泛用于 **语义分割头**、全景分割 stuff 分支
Mask R-CNN 实例 mask 走 per-RoI FCN，语义全图常接 DeepLab 式 head
```

---

## 9. 精读备忘：易混淆点

### 9.1 Atrous = Dilated（同义）

```
文献与框架中混用:
  TensorFlow: atrous_conv2d
  PyTorch: Conv2d(dilation=r)
```

### 9.2 Output Stride 与 Atrous Rate 联动

```
改 OS 必须 **同步改** ASPP 的 rates（6,12,18 是针对 OS=16 标定的）
OS=8 时常用 (12,24,36)，否则 RF 不匹配
```

### 9.3 LargeFOV ≠ ASPP

```
LargeFOV (v1):  **串行** 大 rate 空洞卷积
ASPP (v2+):     **并行** 多 rate + 1×1 +（v3）全局分支
```

### 9.4 v3 最佳模型不用 CRF ≠ CRF 无用

```
v1/v2 时代 CNN 更粗，CRF 增益大
v3 ASPP 已编码多尺度与全局，CRF 边际收益 < 计算成本
```

### 9.5 DeepLabv3+ 的 decoder 不是 U-Net

```
只 concat **一层** stride-4 特征，通道 48
U-Net 是 **每层** skip 对称融合
```

### 9.6 「全连接 CRF」与「ASPP 全局分支」

```
CRF:  推理后处理，双边平滑，不可微（旧）
ASPP image-level:  网络内 **可学习** 的全图先验，前向一次完成
```

### 9.7 MobileNet / 实时版

```
v3+ 用 OS=16 + separable + 裁剪 decoder
  → 车载、移动端分割常用变体（速度-精度旋钮）
```

---

## 10. 后续与相关 work

| 方向 | 代表 |
|------|------|
| 更强上下文 | [PSPNet](./PSPNet.md)、[HRNet](./HRNet.md)、[OCRNet](./OCRNet.md)、[SegFormer](./SegFormer.md) |
| 实例/全景 | Panoptic DeepLab、[Mask2Former](./Mask2Former.md) |
| 医学 | 常直接用 nnU-Net；空洞思想仍见于 3D 分割 |
| 统一卷积 | DCNv2、大核卷积替代部分 ASPP 功能 |

---

## 11. 语义分割脉络（DeepLab 位置）

```
FCN (2015)           skip + deconv                    → [FCN.md](./FCN.md)
U-Net (2015)         concat skip                      → [UNet.md](./UNet.md)
SegNet (2015/2017)   index unpool                     → [SegNet.md](./SegNet.md)
DeepLab v1 (2014/15) atrous + CRF                     ┐
DeepLab v2 (2016)    ASPP + ResNet                   ├─ 本文
DeepLab v3 (2017)    ASPP 改进、无 CRF               │
DeepLab v3+ (2018)   +decoder + separable            ┘
PSPNet (2017)        金字塔池化多尺度          → [PSPNet.md](./PSPNet.md)
HRNet (2019)         高分辨率并行 + 融合         → [HRNet.md](./HRNet.md)
OCRNet (2020)        类级物体语境                → [OCRNet.md](./OCRNet.md)
SegFormer (2021)     MiT + 轻量 MLP 解码器       → [SegFormer.md](./SegFormer.md)
Mask2Former (2022)   mask 分类 + 掩码注意力      → [Mask2Former.md](./Mask2Former.md)
SAM (2023→)          提示式基础模型              → [SAM.md](./SAM.md)
OneFormer (2023)     单模型三任务联合训          → [OneFormer.md](./OneFormer.md)
```

---

## 12. 参考资料

- v1：[Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/abs/1412.7062)
- v2：[DeepLabv2 / LSUN 2015](https://arxiv.org/abs/1606.00915)
- v3：[Rethinking Atrous Convolution](https://arxiv.org/abs/1706.05501)
- v3+：[Encoder-Decoder with Atrous Separable Convolution](https://arxiv.org/abs/1802.02611)
- CRF：[Efficient Inference in Fully Connected CRFs](https://arxiv.org/abs/1210.5644)
- 官方：[TensorFlow DeepLab](https://github.com/tensorflow/models/tree/master/research/deeplab)
- 对比：[FCN.md](./FCN.md)、[UNet.md](./UNet.md)

---

*文档版本：初稿 | 覆盖 DeepLab v1 / v2 / v3 / v3+*
