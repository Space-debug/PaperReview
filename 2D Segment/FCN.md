# FCN（Fully Convolutional Networks）

> 本文档用于整理 FCN（Fully Convolutional Networks for Semantic Segmentation）论文精读笔记。  
> 重点：**分类网络全卷积化**、**可学习上采样（反卷积）**、**跳跃连接融合浅层细节（FCN-32s / 16s / 8s）**，以及相对 **滑动窗口 / 超像素** 的**端到端密集预测**范式。

> 相关：[OCRNet.md](./OCRNet.md) · [OneFormer.md](./OneFormer.md) · [SAM.md](./SAM.md)（提示式基础模型）· [Mask2Former.md](./Mask2Former.md)（mask 分类）· [SegFormer.md](./SegFormer.md)（MiT / All-MLP）· [HRNet.md](./HRNet.md)（高分辨率并行）· [DeepLab.md](./DeepLab.md)（空洞卷积 / ASPP 系列）· [PSPNet.md](./PSPNet.md)（金字塔池化 PPM）· [UNet.md](./UNet.md)（concat skip）· [SegNet.md](./SegNet.md)（索引反池化上采样）· [../2D Detection/RCNN/MaskRCNN.md](../2D Detection/RCNN/MaskRCNN.md)（per-RoI FCN mask 头）· [../2D Detection/FCOS.md](../2D Detection/FCOS.md)（密集预测与分割同构）· [../backbone/ViT.md](../backbone/ViT.md)（下游密集预测）

---

## 1. 概览

### 1.1 基本信息

| 项目 | 内容 |
|------|------|
| 论文 | Fully Convolutional Networks for Semantic Segmentation |
| 作者/机构 | Jonathan Long, Evan Shelhamer, Trevor Darrell（UC Berkeley） |
| 发表 | CVPR 2015（arXiv 2014.11） |
| 任务 | **语义分割**（Semantic Segmentation）：为每个像素预测类别 |
| 代码 | [shelhamer/fcn.berkeleyvision.org](https://github.com/shelhamer/fcn.berkeleyvision.org)（Caffe） |

### 1.2 核心思想（一句话）

**把 ImageNet 预训练的分类 CNN 中的全连接层改成 1×1 卷积，得到任意尺寸输入下的低分辨率类别图，再用可学习的转置卷积上采样，并通过与浅层特征图的跳跃连接融合，一次性输出与输入对齐的逐像素分类结果。**

| 对比 | 传统做法 | FCN |
|------|----------|-----|
| 推理方式 | **滑动窗口** 裁 patch → 逐块分类 → 拼接 | **整图一次前向** |
| 网络末端 | FC 固定输入尺寸 | **全卷积**，尺寸随输入变 |
| 分辨率 | patch 内高，全局靠拼 | 深层 **1/32** + **学习上采样** + **skip** |
| 训练 | 块级标签 / 块级 loss | **逐像素 loss**（可加权） |

### 1.3 整体流水线

```
Input: H×W×3（任意尺寸，如 500×500）
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Backbone: VGG-16（conv1~conv5 / pool1~pool5）               │
│  空间分辨率逐级 /2 → 最终 score 图 stride = 32                │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  全卷积头: fc6/fc7 → 1×1 conv 4096；fc8 → 1×1 conv K 类      │
│  输出: score 张量 K × (H/32) × (W/32)                       │
└──────────────────────────────────────────────────────────────┘
    │
    ├─ FCN-32s ──→ 32× 转置卷积上采样 ─────────────────────────→ 输出
    │
    ├─ FCN-16s ──→ 与 pool4 特征融合 → 2× 上采样 → 再 2× ─────→ 输出
    │
    └─ FCN-8s  ──→ 与 pool3 特征融合 → 2× → 2× → 2× ──────────→ 输出
    │
    ▼
训练: 逐像素 Softmax + Cross-Entropy（忽略 label=255 边界）
推理: argmax 或 softmax → 与输入同尺寸的类别图
```

### 1.4 PASCAL VOC 精度参考（val，IoU mean）

| 模型 | 数据 | mean IU | pixel acc |
|------|------|---------|-----------|
| SDS [2014] | VOC2011 | 52.6 | — |
| DeepLab-LargeFOV [2014] | VOC2012 | 62.25 | — → [DeepLab.md](./DeepLab.md) §2 |
| **FCN-32s** | VOC2012 | 59.4 | — |
| **FCN-16s** | VOC2012 | 62.4 | — |
| **FCN-8s** | VOC2012 | **65.9** | 90.3% |
| FCN-8s + CRF | VOC2012 | 67.2 | — |

- **FCN-8s** 相对 FCN-32s **+6.5 mean IU**，说明 skip 与更细上采样至关重要
- 论文同时报告 NYUDv2、SIFT Flow 上当时 SOTA 级结果

---

## 2. 动机：从分类到密集预测

### 2.1 语义分割要什么

```
输入:  RGB 图像 H×W
输出:  每个像素 (i,j) 一个类别标签 y_ij ∈ {1..K}（或 background）

难点:
  · 既要 **语义**（这是什么）又要 **位置**（在哪）
  · 物体尺度、边界、小结构需要 **空间细节**
```

### 2.2 2015 年前主流：块分类 + 拼贴

```
1. 用 CNN 对 **固定大小 patch**（如 224×224）做分类
2. 滑动窗口扫全图，每个中心像素/超像素得到一个预测
3. 块与块独立，**上下文有限**；重叠区需平均或投票

代表: R-CNN 系区域特征、patch-based CNN [Farabet et al.]

代价:
  · 推理慢（重复计算重叠区域）
  · 块边界不连续
  · 训练标签与推理窗口对齐麻烦
```

### 2.3 FCN 的洞察：分类网络本来就是「卷积的」

```
AlexNet / VGG / GoogLeNet:
  卷积层 → 特征图 (h, w, c)
  最后 FC 层: 把 h×w 压成 1×1

关键观察:
  · FC 可等价写成 **1×1 卷积**（kernel = 全图 spatial size）
  · 若输入不是 224×224 而是更大图像，卷积部分输出 **空间网格**
  · 在该网格上每个位置已有 **感受野对应原图一块区域** 的类别分数

→ 不必裁 patch；**一次前向 = 整张图的粗粒度分割图**
```

### 2.4 仍待解决的问题（FCN 要补的洞）

| 问题 | 来源 | FCN 对策 |
|------|------|----------|
| 输出太粗糙 | pool5 后 stride=32 | **转置卷积** 学习上采样 |
| 边界糊、小物体差 | 深层语义强、浅层细节丢 | **skip** 融合 pool4/pool3 |
| 固定输入尺寸 | 旧 FC 训练习惯 | **全卷积** + 随机 crop 训练 |
| 类别不平衡 | 背景像素远多于前景 | **逐像素 loss + 空间重加权** |

---

## 3. 全卷积化：把分类网改成 FCN 骨干

### 3.1 VGG-16 分类结构回顾

```
Input 224×224
  conv1_1, conv1_2, pool1     → 1/2
  conv2_*, pool2              → 1/4
  conv3_*, pool3              → 1/8
  conv4_*, pool4              → 1/16
  conv5_*, pool5              → 1/32
  fc6: 7×7 conv 4096（或 flatten + FC）
  fc7: 4096
  fc8: 1000-way ImageNet
```

### 3.2 替换规则（论文 §3.2）

| 原层 | FCN 等价 | 输出形状（输入 H×W） |
|------|----------|----------------------|
| fc6（7×7×512→4096） | **conv6**: kernel=7, pad=3, 4096 通道 | H/32 × W/32 × 4096 |
| fc7 | **conv7**: 1×1, 4096 | 同上 |
| fc8（1000 类） | **score**: 1×1, **K 类**（VOC K=21） | H/32 × W/32 × K |

```
实质:
  · 原 fc6 的 7×7 权重 → 变成 7×7 卷积核，在 1/32 特征图上滑动
  · 原 fc7/fc8 的全连接 → 1×1 卷积，**共享权重于所有空间位置**

→ 输出称为 **score map**（未归一化的 per-class 响应）
```

### 3.3 为何叫「全卷积」

```
性质 1: 输入尺寸可变（训练/测试可用不同 H,W）
性质 2: 输出 score 图尺寸 ∝ 输入尺寸（比例固定为 1/32）
性质 3: 平移等价 — 同一物体在图中平移，对应 score 图上的峰也平移

与「全连接层压成向量」对比:
  FC 破坏空间结构；FCN 保留 (h,w) 网格上的决策
```

### 3.4 其他骨干（论文实验）

| 骨干 | 最深 stride | 备注 |
|------|-------------|------|
| **VGG-16** | 32 | 主结果 |
| AlexNet | 32 | 更快更糙 |
| GoogLeNet | 32 | 多分支，实现需处理 |

---

## 4. 上采样：从 1/32 score 到像素级

### 4.1 三种上采样思路（论文 §3.3）

```
① 双线性插值（固定核）
   · 不学习，仅按邻域加权
   · 用作 **转置卷积初始化**（见下）

② 反池化（Unpooling）
   · 记录 pool 时最大值位置，上采样填回
   · Zeiler & Fergus 可视化常用；FCN 主路径不用

③ 转置卷积 / 反卷积（Stride > 1 的卷积）
   · **可学习** 上采样核
   · FCN 采用：把粗糙 score 放大到更细网格
```

### 4.2 转置卷积（Deconvolution）在 FCN 中的角色

```
输入:  score 图 (K, h, w)，如 h=H/32
操作:  kernel k×k，stride s 的 **转置卷积**
输出:  (K, s·h, s·w)  更密的 score

FCN-32s:
  一层 stride=32 的转置卷积 → 直接从 H/32 拉到 H×W

FCN-16s / 8s:
  多个 stride=2 的转置卷积，中间与 skip 特征 **相加** 后再继续上采样
```

**与「普通卷积 + 插值」的区别**：

```
双线性 resize + conv:
  上采样核固定，只学后续融合

转置卷积:
  上采样 **核本身可训练**，能学习「类间边界如何展开」
  论文图 3：学到的 32× 上采样核类似 **平滑 + 中心增强** 的双线性模式
```

### 4.3 双线性初始化（关键实现细节）

```
对 stride = s 的上采样层，权重初始化为 **双线性核**：

  w[i,j] = (1 - |i - c|/c) × (1 - |j - c|/c)   （c 为中心索引）

效果:
  · 训练起始行为 ≈ 双线性插值，稳定
  · 微调阶段核会偏离，以拟合数据驱动的边界锐化

避免:
  随机初始化大 stride 反卷积 → 训练初期 score 图噪声大、难收敛
```

---

## 5. 跳跃连接：FCN-32s / 16s / 8s

### 5.1 问题：仅深层上采样不够

```
pool5 后特征:
  ✓ 语义强（「这是马」）
  ✗ 空间粗（1/32），边界、细腿、小物体位置不准

pool4 / pool3:
  ✓ 更高分辨率（1/16, 1/8），边缘、纹理定位好
  ✗ 语义弱、感受野小

→ **skip connection**: 把浅层 **局部** 与深层 **语义** 相加融合
```

### 5.2 融合方式（element-wise sum）

```
步骤模板（以 FCN-16s 为例）:

1. 从 pool5 得到 score_s32: K × H/32 × W/32
2. 2× 转置卷积 → score_s16: K × H/16 × W/16
3. 从 pool4 经 1×1 conv 把通道压到 K，得到 score_pool4
4. **score_s16 + score_pool4**（同尺寸相加）
5. 再 2× 转置卷积 ×2 → 到 H/8，再 ×2 → H/4？ 

注意: 论文 FCN-16s 最终到 H/2 还是 H 取决于实现；标准描述是:
  FCN-16s: 融合 pool4 后上采样到 **1/16 相对输入再插值到全图** 或连续 2× twice

标准 Berkeley 实现语义:
  · FCN-32s: 输出 stride **32**（再双线性到全图）
  · FCN-16s: 输出 stride **16**
  · FCN-8s:  输出 stride **8**（最细，再双线性到像素）
```

**FCN-8s 完整路径（最常用）**：

```
pool5 → 1×1 score (stride 32)
  → 转置卷积 stride=2 → stride 16
  → + 1×1 conv(pool4) 投影到 K 维后 **逐元素相加**
  → 转置卷积 stride=2 → stride 8
  → + 1×1 conv(pool3) 投影后相加
  → 得到原生 score 图: K × (H/8) × (W/8)

推理到全像素:
  → **双线性插值 ×8** 拉到 H×W（论文评估用此 stride-8 输出）

命名 "8s": 网络输出的 score **相对输入 stride = 8**，
           不是网络只有 8 层
```

### 5.3 三档模型对比

| 变体 | 融合层 | 输出 stride（相对输入） | mean IU (VOC2012) |
|------|--------|-------------------------|-------------------|
| **FCN-32s** | 无 skip | 32（一次 32× up） | 59.4 |
| **FCN-16s** | pool4 | 16 | 62.4 |
| **FCN-8s** | pool4 + pool3 | 8 | **65.9** |

```
消融结论（论文 Table 3）:
  · 32s → 16s: +3.0 IU（pool4 边界改善明显）
  · 16s → 8s: +3.4 IU（pool3 更细，小结构、轮廓更好）
  · 仅加深网络不加 skip: 提升有限
```

### 5.4 Skip 的 1×1 conv 作用

```
pool4 通道数 512（VGG），pool3 256
score 通道数 K（如 21）

融合前:
  1×1 conv 将 pool4/pool3 特征 **投影到 K 维 score 空间**
  再与上采样后的 score **逐通道相加**

→ 浅层贡献的是 **与类别对齐的局部证据**，而非 raw 特征直接拼
```

---

## 6. 损失函数与训练策略

### 6.1 逐像素损失

```
对每个有效像素 (i,j):

  ℓ_ij = -log p_{y_ij}(i,j)

p_c 来自 score 经 **Softmax**（沿类别维）

总损失:
  L = (1/|Ω|) Σ_{(i,j)∈Ω} ℓ_ij

Ω: 有效像素集合（常 **忽略 label=255** 的 VOC 边界/未标注区）
```

**与检测 loss 对比**：

```
RetinaNet:  稀疏 anchor 上 Focal Loss
FCN:        **稠密** 每像素 CE，背景类占绝大多数
```

### 6.2 空间 subsampling（可选加速）

```
问题: 全图 H×W 像素都算 loss，反传慢

论文 trick **max-deepest-pixel**:
  · 在 score 图（1/32）每个 cell 内，只选 **loss 最大** 的那一个像素反传
  · 类似 hard example mining，保持边界/难像素梯度

效果: 训练加速 ~3×，精度略降（论文可选）
```

### 6.3 类别不平衡：oversampling

```
VOC 中背景像素远多于 horse、person 等

做法:
  · 统计训练集每类像素频率
  · 对稀有类像素 **过采样**（重复计入 loss 或提高权重）

与检测 Focal Loss（2017）对比:
  FCN 用 **采样/权重** 启发式；Focal 用连续调制 (1-p_t)^γ
```

### 6.4 训练超参（VOC，论文典型）

| 超参 | 值 |
|------|-----|
| 优化器 | SGD，momentum 0.9 |
| LR | 1e-10（fc8/score）/ 1e-8（其余）等 **分层 lr**（微调） |
| 初始化 | ImageNet 分类预训练 → 全卷积化 |
| 输入 | **随机 crop**（如 500×500）从大图裁切 |
| 迭代 | 数万 iter（Caffe 时代） |
| 数据增强 | 翻转、缩放、色彩抖动 |

```
分层学习率原因:
  · 预训练 conv 层已收敛 → 小 lr
  · 新 score 层、转置卷积、skip 投影 → 相对大 lr
```

### 6.5 整图推理

```
测试时可输入 **任意尺寸**（受 GPU 显存限制）:

  Forward → score 全分辨率（经上采样）→ argmax

无需滑动窗口；**一次前向覆盖全图**

大图上可用:
  · 多尺度测试（scale jitter）
  · 重叠 tile 仅当显存不够（非论文核心）
```

---

## 7. 与 Mask R-CNN / 检测的衔接

### 7.1 语义分割 vs 实例分割

```
FCN:           每像素 **一个类**（不区分两只相邻的羊）
Mask R-CNN:    每像素 **(类, 实例 id)**，先检测再 per-RoI mask

Mask R-CNN 的 mask 分支:
  RoIAlign 14×14 → 小 FCN（4× 反卷积）→ 28×28 mask
  → 思想直接继承 FCN 的「全卷积密集 mask」
```

见 [MaskRCNN.md](../2D Detection/RCNN/MaskRCNN.md) §4 mask 头。

### 7.2 与 FCOS 的「密集预测」同构

```
FCOS:   特征图每点 → 是否物体 + ltrb
FCN:    特征图每点 → 是否某类（经上采样对齐像素）

二者都强调 **卷积特征图上的密集决策**，无需 RoI 裁剪
```

见 [FCOS.md](../2D Detection/FCOS.md) §2.2。

---

## 8. 后处理与其它工作

### 8.1 Dense CRF（可选，非 FCN 本体）

```
FCN 输出 coarse softmax → 作为 unary potential
+ Krahenbühl & Koltun 全连接 CRF（双边项：颜色+空间）

VOC: FCN-8s 65.9 → **67.2** mean IU

注意:
  · CRF **不可微** 于 FCN 训练（后处理）
  · DeepLab 系列后来把 CRF 思想融入空洞卷积/Margin loss
```

### 8.2 与同期 / 后续方法对比

| 方法 | 核心 | 相对 FCN |
|------|------|----------|
| **Patch CNN** | 滑窗块分类 | FCN 快且全局一致 |
| **DeepLab-LargeFOV** | 空洞卷积扩大感受野 | 同期 VOC 62.25 → [DeepLab.md](./DeepLab.md) |
| **U-Net (2015)** | 编码器-解码器 + **concat** skip | 医学图像 → [UNet.md](./UNet.md) |
| **SegNet** | 反池化上采样 | 无学习 deconv → [SegNet.md](./SegNet.md) |
| **PSPNet** | 金字塔池化 PPM | → [PSPNet.md](./PSPNet.md) |
| **DeepLabv3+** | ASPP + decoder | → [DeepLab.md](./DeepLab.md) |

---

## 9. 消融与关键结论

### 9.1 Skip 与 stride

| 配置 | mean IU 趋势 |
|------|--------------|
| 仅 32× 上采样 | 最低 |
| + pool4 | 明显提升边界 |
| + pool3（FCN-8s） | 最佳 |

### 9.2 上采样核

```
固定双线性 vs 学习转置卷积:
  学习核略优；双线性初始化训练最稳
```

### 9.3 全卷积 vs 滑窗

```
滑窗重叠平均:
  · 慢
  · 块边界 artifact

FCN 整图:
  · 快
  · 上下文为全图感受野链（虽末端 stride 大）
```

---

## 10. 精读备忘：易混淆点

### 10.1 「全卷积」≠ 输出分辨率等于输入

```
FCN 输出可原生 stride 8/16/32
  到 **每个像素** 往往还需最后一层双线性或反卷积

「全卷积」指的是 **网络中无 FC 压扁空间**，不是指输出无下采样
```

### 10.2 FCN-8s 的「8」指 stride，不是 8 层

```
8s = 最终 score 图相对原图下采样 **8 倍**
     不是网络深度 = 8
```

### 10.3 Skip 是 **相加**，不是 U-Net 的 **拼接**

```
FCN:     score_up + conv(pool_shallow)   通道维需对齐到 K
U-Net:   concat(encoder, decoder)       通道加倍再 1×1 压  → [UNet.md](./UNet.md)

相加: 参数少，要求两路语义已在 score 空间
拼接: 保留更多浅层通道信息（后续分割常用）
```

### 10.4 反卷积 ≠ 普通卷积的逆

```
转置卷积是 **可学习上采样** 的实现方式
  可能有 checkerboard artifact（后续文献讨论）
FCN 用大 stride + 双线性 init 缓解
```

### 10.5 语义分割 FCN vs Mask R-CNN 里的「FCN 头」

```
本文 FCN:     **整图** 语义，一类像素一个标签
Mask 分支:    **每个 RoI** 内 28×28 二值 mask，K 类独立通道

同名「FCN」: 都指 **卷积堆叠 + 上采样** 出密集 mask，任务不同
```

### 10.6 255 标签

```
PASCAL VOC 边界像素常标 255
训练 loss **mask 掉** 这些位置，不参与梯度
```

---

## 11. 语义分割脉络（FCN 位置）

```
传统:              滑窗 / 超像素 + 手工特征
FCN (2015)         分类网全卷积化 + skip + 学习 up     ← 本文
SegNet (2015/2017) 索引 unpool、无 skip、轻量           → [SegNet.md](./SegNet.md)
U-Net (2015)       编码-解码 + concat skip（医学）  → [UNet.md](./UNet.md)
DeepLab v1-v3+     空洞卷积 + ASPP + CRF/decoder  → [DeepLab.md](./DeepLab.md)
PSPNet (2017)      金字塔池化 PPM            → [PSPNet.md](./PSPNet.md)
HRNet (2019)       高分辨率并行 + 融合         → [HRNet.md](./HRNet.md)
OCRNet (2020)      物体语境 + HRNet            → [OCRNet.md](./OCRNet.md)
SegFormer (2021)   高效 Transformer 分割     → [SegFormer.md](./SegFormer.md)
Mask2Former (2022) mask 分类（分任务训）        → [Mask2Former.md](./Mask2Former.md)
OneFormer (2023)   Task Token 一次训三任务      → [OneFormer.md](./OneFormer.md)
SAM (2023→)        提示式分割基础模型          → [SAM.md](./SAM.md)
Mask R-CNN (2017)  实例：检测 + per-RoI FCN mask      → [MaskRCNN.md](../2D Detection/RCNN/MaskRCNN.md)
SegFormer              MiT + 轻量 MLP 头        → [SegFormer.md](./SegFormer.md)
Mask2Former            统一 mask 分类              → [Mask2Former.md](./Mask2Former.md)（[DETR.md](../2D Detection/DETR/DETR.md) §10）
```

**历史地位**：

```
1. 首次证明 ImageNet CNN **端到端** 可训语义分割；
2. **skip + 上采样** 成为 encoder-decoder 标准模板；
3. 开启「分割 = 分类 backbone + 密集 head」十年主线。
```

---

## 12. 参考资料

- 原论文：[Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)（CVPR 2015）
- 实现：[fcn.berkeleyvision.org](https://github.com/shelhamer/fcn.berkeleyvision.org)
- 上采样/可视化：[Zeiler & Fergus, Deconvolutional Networks](https://arxiv.org/abs/1311.2901)
- 后处理 CRF：[Krahenbühl & Koltun, Efficient Inference in Fully Connected CRFs](https://arxiv.org/abs/1210.5644)
- 实例分割延伸：[MaskRCNN.md](../2D Detection/RCNN/MaskRCNN.md)
- 后续：[DeepLab.md](./DeepLab.md)、[UNet.md](./UNet.md)

---

*文档版本：初稿 | 对应论文 CVPR 2015 FCN*
