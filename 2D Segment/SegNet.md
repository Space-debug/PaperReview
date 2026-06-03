# SegNet

> 本文档用于整理 SegNet（A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation）论文精读笔记。  
> 重点：**编码器-解码器对称结构**、**Max-Pooling 索引记忆 + Max-Unpooling 上采样**、**无 skip 的特征传递**，以及与 [FCN.md](./FCN.md) / U-Net 在**上采样与参数量**上的差异。

> 相关：[FCN.md](./FCN.md)（转置卷积 + skip）· [UNet.md](./UNet.md)（concat skip）· [DeepLab.md](./DeepLab.md)（ASPP / 空洞卷积）· [../2D Detection/RCNN/MaskRCNN.md](../2D Detection/RCNN/MaskRCNN.md)

---

## 1. 概览

### 1.1 基本信息

| 项目 | 内容 |
|------|------|
| 论文 | SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation |
| 作者/机构 | Vijay Badrinarayanan, Alex Kendall, Roberto Cipolla（剑桥大学） |
| 发表 | BMVC 2015；扩展版 **IEEE TPAMI 2017** |
| 任务 | **语义分割**；强调 **实时/嵌入式** 场景下的精度-效率权衡 |
| 代码 | [alexgkendall/caffe-segnet](https://github.com/alexgkendall/caffe-segnet)（Caffe + GPU unpooling 算子） |

### 1.2 核心思想（一句话）

**用 VGG-16 卷积层作编码器逐级下采样并保存每次 Max-Pool 的「赢家位置索引」，解码器用 Max-Unpooling 按索引把特征放回高分辨率网格，再接轻量卷积精炼，全程不用可学习转置卷积、也不做 U-Net 式 skip 拼接，以极少额外存储换端到端密集分割。**

| 对比 | FCN-8s | **SegNet** | [U-Net](./UNet.md) |
|------|--------|------------|-------|
| 上采样 | **可学习** 转置卷积 + 双线性 | **固定** Max-Unpool（索引驱动） | 上采样 + **concat** skip |
| 浅层信息 | skip **相加** pool3/4 | **仅** 池化索引，无特征图传递 | skip **拼接** 编码特征 |
| 解码器厚度 | 上采样层 + 1×1 投影 | 每级 **1 层** conv（比编码器薄） | 对称双 conv |
| 参数量 | 较大（含 deconv） | **明显更小** | 中等 |

### 1.3 整体流水线

```
Input: H×W×3
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Encoder（VGG-16 前 13 层，无 FC）                            │
│  每个 block: Conv×2 → BN → ReLU → MaxPool                    │
│  MaxPool 时: 输出下采样特征 + **保存 pool indices**            │
└──────────────────────────────────────────────────────────────┘
    │  最深层特征 (H/32 × W/32)
    │  indices_1 … indices_5  （每级池化的位置图）
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Decoder（与编码器对称，通道逐级减少）                         │
│  每个 block: Max-Unpool(indices) → Conv×1 → BN → ReLU        │
│  空间尺寸逐级 ×2，通道与编码器对应级对齐                       │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Softmax 分类层: 1×1 conv → K 类（与输入同分辨率 H×W）        │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
训练: 加权逐像素 Cross-Entropy（median frequency balancing）
推理: argmax → H×W 标签图
```

### 1.4 精度与效率参考

#### CamVid（11 类街景，论文主 benchmark）

| 模型 | Class avg IOU | Global avg acc | 参数量（约） |
|------|---------------|----------------|--------------|
| FCN-8s | — | — | 大 |
| DeepLab-LFOV | — | — | 大 |
| **SegNet** | **60.1%** | **86.4%** | **~29M**（encoder 共享 VGG） |
| Bayes-SegNet | 63.1% | 87.9% | + Dropout 推理 |

#### PASCAL VOC 2012（val，mean IU）

| 模型 | mean IU |
|------|---------|
| FCN-8s | **65.9** |
| [DeepLab-LargeFOV](./DeepLab.md) | 62.25 |
| **SegNet** | **59.1**（PAMI 表；调参后约 60+） |

- **街景**（CamVid、SUN RGB-D）上 SegNet 与 FCN 接近，边界常更利落
- **VOC** 上略低于 FCN-8s，主因 **无 skip**，小物体与类内细节稍弱
- 卖点：**参数少、推理内存友好、上采样不引入 checkerboard**

---

## 2. 动机：编码器-解码器与「如何上采样」

### 2.1 分割需要对称结构

```
编码器:  H×W → H/32×W/32   提取语义、增大感受野
解码器:  H/32 → H×W        恢复空间分辨率并输出 per-pixel 类

2015 前后主流路线:
  · FCN:     单塔 + 跳层到浅层（不是严格对称）
  · DeconvNet: 对称但用 **多通道 unpool + 可学习 deconv**，参数爆炸
  · U-Net:   对称 + **传递完整特征图**（占显存）
  · SegNet:  对称 + **只传递 pool 索引**（省显存）
```

### 2.2 Max-Pool 丢了什么？SegNet 怎么补

```
Max-Pool 2×2:
  每个窗口只保留 **最大值** 传到下一层
  其余 3 个位置信息丢弃 → 上采样时若「盲目插值」边界糊

SegNet 做法:
  下采样时记录每个窗口内 **最大值的位置**（indices）
  上采样时把特征 **填回该位置**，其余位置为 0
  → 边界处的强响应回到原图对应像素邻域
  → 再经 decoder conv 扩散上下文
```

### 2.3 与 FCN 的根本分歧

| 维度 | FCN | SegNet |
|------|-----|--------|
| 上采样参数 | **有**（转置卷积可训练） | **无**（unpool 由索引确定） |
| 浅层细节来源 | pool3/4 **特征相加** | **仅** 索引定位，无浅层特征图 |
| 设计哲学 | 分类网改造 + 多尺度融合 | **专用** encoder-decoder，推理轻 |

见 [FCN.md](./FCN.md) §4–§5。

---

## 3. 网络结构详解

### 3.1 编码器（Encoder）

与 **VGG-16 前 13 个卷积层** 对齐（去掉全连接），通常 **5 个 block**：

| Block | 卷积 | 输出尺寸（相对输入） | 池化 |
|-------|------|----------------------|------|
| 1 | 64×3×3 ×2 | H/2 | MaxPool + **存 idx₁** |
| 2 | 128×3×3 ×2 | H/4 | MaxPool + **存 idx₂** |
| 3 | 256×3×3 ×3 | H/8 | MaxPool + **存 idx₃** |
| 4 | 512×3×3 ×3 | H/16 | MaxPool + **存 idx₄** |
| 5 | 512×3×3 ×3 | **H/32** | MaxPool + **存 idx₅** |

```
每个 block 内:
  Conv → Batch Normalization → ReLU（×2 或 ×3）
  Max-Pooling 2×2, stride 2

与原始 VGG 差异:
  · 加入 **BN**（PAMI 版训练更稳）
  · 无 FC；pool 层输出 **indices** 供解码器使用
```

**预训练**：ImageNet 上训练好的 VGG-16 卷积权重初始化 encoder（迁移学习）。

### 3.2 解码器（Decoder）— 与 FCN/DeconvNet 的关键区别

```
SegNet decoder（轻量）:
  每个 block 仅 **1 个** 3×3 conv（非 VGG 的 2 个）
  通道数与 encoder 对称递减: 512 → 512 → 256 → 128 → 64

流程（自深到浅）:
  输入: 最深层特征 F₅ (H/32)
    → Max-Unpool 用 idx₅ → (H/16)
    → Conv 512 → BN → ReLU
    → Max-Unpool 用 idx₄ → (H/8)
    → Conv 256 → …
    … 直至 (H/2)
    → 最后一级 unpool → (H×W)
    → 1×1 conv → K 类 logits
```

**无 skip connection**：

```
U-Net:  decoder 输入 = upsample(深层) **concat** encoder 同层特征
SegNet: decoder **只看到** 深层语义 + 索引还原的稀疏 placement
        浅层 **纹理特征图不传递**，只靠 unpool 定位
```

### 3.3 Max-Unpooling 机制（精读核心）

#### 编码阶段：保存 indices

```
输入特征图 X:  H×W×C
MaxPool 2×2, stride 2 → Y: (H/2)×(W/2)×C

对每个通道、每个 pool 窗口 (2×2):
  记录 argmax 位置 m ∈ {0,1,2,3}（扁平索引）
  存为 indices 图: (H/2)×(W/2)，每个 cell 一个整数

存储代价:
  每级约 H×W×2 bits 量级（实现常用 uint8 存 0~3）
  远小于保存完整 encoder 特征图（U-Net skip）
```

#### 解码阶段：按 indices 回填

```
输入:  上采样前特征 Z: (H/2)×(W/2)×C'
indices: 与编码时 **同一 forward** 存的 idx（推理时必须配对）

Max-Unpool:
  输出张量 O: H×W×C'，先 **全零**
  对每个 coarse 位置 (i,j) 和通道 c:
    将 Z[i,j,c] 写到 O 中对应 2×2 窗口内 **indices[i,j] 指定的那一格**
    其余 3 格保持 0

再经过 Conv → 非零位置向邻域扩散信息
```

**直觉图（单通道 1D 简化）**：

```
编码 pool 前:  [1, 5, 3, 2]  → max=5 在 index 1
编码输出:      [5]
保存 index:    1

解码 unpool:   [0, 5, 0, 0]   ← 5 回到原位置
解码 conv:     邻域卷积平滑 → 恢复连续边界
```

#### 与 Zeiler DeconvNet 的 unpool 对比

| | DeconvNet | SegNet |
|--|-----------|--------|
| unpool 后 | 多通道 + **可学习** deconv | **单 conv** 精炼 |
| 参数量 | 很大 | **刻意压缩** |
| 目标 | 重建 / 可视化 | **实时分割** |

### 3.4 输出层

```
最后一层 decoder 输出 H×W×64（典型）
  → 1×1 convolution → H×W×K（K=类别数）
  → Softmax（训练用 CE；实现常 log-softmax + NLL）

**原生全分辨率**: 不像 FCN-8s 还需额外 ×8 双线性到像素；
  decoder 最后一级 unpool 已恢复到 H×W
```

---

## 4. SegNet-Bayesian（不确定性扩展）

PAMI 论文给出 **Bayes-SegNet**：同一架构 + **Dropout** 在推理时保持开启。

```
训练:   标准 SegNet + encoder/decoder 中 Dropout（如 p=0.5）
推理:   同一输入前向 **T 次**（Monte Carlo Dropout）
        得到 T 张 class probability maps
        平均 → 预测分割
        方差 → **像素级不确定性**（认知不确定性）

用途:
  · 自动驾驶：不确定区域交给人或融合传感器
  · CamVid 上 class IOU 提升约 **3%**（60.1 → 63.1）
代价:   推理时间 ×T
```

---

## 5. 损失函数与训练

### 5.1 逐像素加权交叉熵

```
ℓ = - Σ_{i,j} w_{y_ij} · log p_{y_ij}(i,j)

p: Softmax 后类别概率
w_c: 类 c 的权重（缓解背景主导）
```

### 5.2 Median Frequency Balancing（类权重）

```
对训练集每类 c 统计像素频率 f_c
权重:

  w_c = median({f_1, …, f_K}) / f_c

稀有类 f_c 小 → w_c 大 → loss 中占比上升

与 FCN 的 oversampling 类似，SegNet 用 **显式权重** 实现
```

### 5.3 训练超参（论文典型）

| 超参 | 值 |
|------|-----|
| 优化器 | SGD |
| 初始 LR | 0.001（CamVid）；VOC 可调 |
| Momentum | 0.9 |
| Weight decay | 0.0005 |
| LR 策略 | 固定或 step decay（实现依赖） |
| 初始化 | **VGG-16 ImageNet** encoder；decoder **随机** |
| 输入尺寸 | CamVid 360×480；可整图或 crop |
| 数据增强 | 随机翻转、缩放、色彩抖动（数据集相关） |
| 标签 | 忽略 void / 未标注类（数据集定义） |

```
微调策略:
  · Encoder 用预训练权重，lr 可略小
  · Decoder 从头学，需更多 iter 才能对齐边界
  · BN 在 batch 较小时需注意统计稳定性
```

### 5.4 推理

```
单尺度整图前向（与 FCN 相同，无滑窗）
  → Softmax → argmax

可选 Bayes-SegNet: T 次前向平均

后处理:
  · 论文主结果 **不用 CRF**（与 FCN+CRF 对比时纯网络）
  · 街景任务边界已较清晰，得益于 unpool 定位
```

---

## 6. 与 FCN / U-Net / DeepLab 对比

### 6.1 上采样路线三分法

```
① 可学习转置卷积（FCN）
   优点: 可拟合复杂上采样
   缺点: 参数多；大 stride 易 checkerboard

② 固定 unpool + 索引（SegNet）
   优点: 无 deconv 参数；边界对齐有物理意义
   缺点: 无 skip 时细节弱于 FCN-8s / U-Net

③ 双线性/最近邻 + 卷积（常见现代 head）
   优点: 稳定
   缺点: 上采样本身不可学习
```

### 6.2 显存与参数（论文论点）

```
U-Net skip:
  需缓存 encoder 各层完整特征图 → 显存 ∝ 多层 H×W×C

SegNet indices:
  每层只存 (H/2^l)×(W/2^l) 的 index map → 远小于特征图

参数量:
  SegNet decoder 每层 1 conv vs VGG 2 conv
  → 总参数显著小于 DeconvNet / 部分 FCN 变体
  → 适合 **嵌入式、车载**（与 Cambridge 自动驾驶背景一致）
```

### 6.3 任务表现差异

| 场景 | SegNet 表现 | 原因 |
|------|-------------|------|
| **CamVid / 驾驶** | 强（SOTA 级） | 类少、纹理重复；unpool 利边界 |
| **PASCAL VOC** | 略逊 FCN-8s | 类多、小物体；缺 skip 不利 |
| **SUN RGB-D** | 竞争力 | 室内几何边界 + 深度任务扩展 |

---

## 7. 消融与设计选择（论文结论）

### 7.1 上采样方式

| 配置 | 效果 |
|------|------|
| Max-Unpool + indices | **默认**，边界与参数均衡 |
| 双线性上采样替代 | 定位变糊，class IOU 降 |
| 转置卷积替代 | 参数增，未必更优 |

### 7.2 Skip 与否

```
论文明确 **不** 使用 encoder 特征 skip:
  · 控制模型大小
  · 证明仅 indices 已能恢复大量空间结构

代价: VOC 小物体、细长类（bicycle pole）弱于 FCN-8s
```

### 7.3 Encoder 深度

```
VGG-16 13 层是标准配置
更浅 encoder → 语义不足
更深 → 收益递减，实时性变差
```

### 7.4 Batch Normalization

```
PAMI 版强调 BN 加入后训练收敛与精度提升
与当时 Caffe 生态一致
```

---

## 8. 精读备忘：易混淆点

### 8.1 indices 必须「编码-解码配对」

```
同一张图、同一次 forward 的 indices 才能用于 unpool
  · 训练: 正常反传，indices 由当前 batch 图生成
  · 推理: 单次前向链内编码器产出 indices → 解码器消费

不能混用另一张图的 indices（尺寸与内容均不对应）
```

### 8.2 Max-Unpool 输出稀疏，靠后续 Conv 填洞

```
unpool 后大部分格点为 0
  → 必须接 3×3 conv 扩散
  → 不是「unpool 一步就得到最终分割」
```

### 8.3 SegNet ≠「没有上采样参数」的「没有参数」

```
无 **转置卷积** 参数，但 decoder 仍有 **大量 3×3 conv**
  轻量是相对 DeconvNet / 厚 decoder，不是 Zero-param decoder
```

### 8.4 与 FCN 的「FCN」字样

```
SegNet 论文对比对象 FCN 指 Long et al. **整图分割 FCN**
  与 Mask R-CNN 里 **per-RoI mask FCN 头** 不同
见 [FCN.md](./FCN.md) §10.5
```

### 8.5 Bayes-SegNet 不是独立架构

```
结构同 SegNet；差异仅在 **Dropout + 多次推理融合**
```

### 8.6 CamVid 与 VOC 指标不可直接比绝对值

```
类别数、分辨率、评估脚本不同
  读论文表格时注意 **同一 benchmark 行内** 比较
```

---

## 9. 语义分割脉络（SegNet 位置）

```
FCN (2015)          全卷积 + 学习 deconv + skip     → [FCN.md](./FCN.md)
U-Net (2015)        concat skip + 对称编解码      → [UNet.md](./UNet.md)
SegNet (2015/2017)  **索引 unpool**、无 skip、轻量   ← 本文
DeconvNet (2014)    厚 decoder + 学习 deconv
DeepLab 系列         空洞卷积 + ASPP              → [DeepLab.md](./DeepLab.md)
ENet (2016)         进一步压缩 SegNet 思路面向实时
PSPNet               金字塔池化                → [PSPNet.md](./PSPNet.md)
HRNet                高分辨率并行融合            → [HRNet.md](./HRNet.md)
SegFormer            轻量 Transformer 分割      → [SegFormer.md](./SegFormer.md)
Mask2Former          统一 mask 分类              → [Mask2Former.md](./Mask2Former.md)
SAM                  提示式基础模型              → [SAM.md](./SAM.md)
OneFormer            单模型三任务                → [OneFormer.md](./OneFormer.md)
```

**历史地位**：

```
1. 将 **pooling indices** 作为编解码桥梁，影响后续轻量分割与嵌入式部署讨论；
2. 与 FCN 并列证明「**上采样不必可学习**」也能端到端分割；
3. Bayes-SegNet 推动 **分割不确定性估计** 在自动驾驶场景的落地。
```

---

## 10. 参考资料

- 原论文：[SegNet (BMVC 2015)](https://arxiv.org/abs/1511.00561) · [PAMI 2017 扩展](https://ieeexplore.ieee.org/document/7803544)
- 实现：[alexgkendall/caffe-segnet](https://github.com/alexgkendall/caffe-segnet)
- 对比：[FCN.md](./FCN.md)、[UNet.md](./UNet.md)、[DeconvNet](https://arxiv.org/abs/1311.2901)
- 类权重：[ENet](https://arxiv.org/abs/1606.02147) 等沿用 median frequency balancing
- 后续实时：[ENet](https://arxiv.org/abs/1606.02147)、[ERFNet](https://arxiv.org/abs/1611.07647)

---

*文档版本：初稿 | 对应论文 BMVC 2015 / IEEE TPAMI 2017 SegNet*
