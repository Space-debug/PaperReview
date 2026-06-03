# U-Net

> 本文档用于整理 U-Net（Convolutional Networks for Biomedical Image Segmentation）论文精读笔记。  
> 重点：**对称编解码 + 通道拼接（concat）skip**、**Overlap-tile 大图推理**、**弹性形变增强与边界加权损失**，以及相对 [FCN.md](./FCN.md) / [SegNet.md](./SegNet.md) 在**小样本医学分割**上的设计选择。

> 相关：[FCN.md](./FCN.md)（skip 相加）· [SegNet.md](./SegNet.md)（仅 indices、无特征 skip）· [DeepLab.md](./DeepLab.md)（ASPP / v3+ decoder）· [HRNet.md](./HRNet.md)（全程高分辨率流）· [SegFormer.md](./SegFormer.md)（Transformer 分割）· [Mask2Former.md](./Mask2Former.md)（mask 分类统一三任务）· [SAM.md](./SAM.md)（提示式基础模型）

---

## 1. 概览

### 1.1 基本信息

| 项目 | 内容 |
|------|------|
| 论文 | U-Net: Convolutional Networks for Biomedical Image Segmentation |
| 作者/机构 | Olaf Ronneberger, Philipp Fischer, Thomas Brox（弗赖堡大学） |
| 发表 | **MICCAI 2015**（arXiv 2015.05） |
| 任务 | **生物医学图像分割**（显微细胞、线粒体等）；二分类/多类 **密集 mask** |
| 代码 | 原版 Caffe；社区广泛使用 [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)、[nnU-Net](https://github.com/MIC-DKFZ/nnUNet) |

### 1.2 核心思想（一句话）

**用对称的「收缩路径」编码上下文、用「扩张路径」上采样恢复分辨率，并在每一级用 concat 把编码器的高分辨率特征直接接到解码器，使网络在极少标注（如 30 张图）下仍能精确定位边界；配合弹性形变数据增强与分离粘连细胞的加权损失，赢得 ISBI 细胞追踪挑战赛。**

| 对比 | FCN-8s | SegNet | **U-Net** |
|------|--------|--------|-----------|
| Skip 形式 | 1×1 投影后 **相加** | **无** 特征 skip | **concat** 编码特征 |
| 编解码对称性 | 分类网改造，不完全对称 | 对称但 decoder 更薄 | **严格对称** 双 conv block |
| 上采样 | 转置卷积 / 双线性 | Max-Unpool + indices | **Up-conv 2×2** 或上采样 |
| 典型场景 | 自然图像 VOC | 街景、车载 | **医学小数据集** |

### 1.3 整体流水线（经典 U 形）

```
Input: 572×572×1（灰度显微图，论文示例尺寸）
    │
    ▼  ─── 收缩路径（Contracting / Encoder）────────────────────
    │      [Conv3×3-ReLU]×2 → 64  ──────────────────────────┐ skip
    │      MaxPool 2×2 ↓ 286×286                               │
    │      [Conv]×2 → 128  ───────────────────────────────┐   │
    │      MaxPool ↓ 140×140                               │   │
    │      … 256 → 512 …                                   │   │
    │      MaxPool ↓                                        │   │
    │      [Conv]×2 → 1024  （最底层 / bottleneck）          │   │
    ▼  ─── 扩张路径（Expansive / Decoder）────────────────────   │
           Up-conv 2×2 ↑  （或 upsample + conv）               │
           **Concat** 与左侧同层 skip（**crop** 对齐尺寸）  ←──┘
           [Conv3×3-ReLU]×2 → 512 … 逐级 ↑
           … 直至 388×388 有效预测区（overlap-tile 时）
    │
    ▼
1×1 Conv → C 类（常见 C=2：背景 + 细胞）
Softmax + **加权** Cross-Entropy
```

### 1.4 ISBI 挑战赛结果（论文 Table 2）

| 数据集 | 指标 | 第二名 | **U-Net** |
|--------|------|--------|-----------|
| PhC-U373（DIC） | IOU | 83% | **92%** |
| DIC-HeLa | IOU | 46% | **77.5%** |
| EM 分割（线粒体） | Warping error / Rand error | 竞赛最优 | **冠军** |

- 训练集规模：**30 张 512×512 EM 图**（仅半幅有标注）即可超过 prior
- 说明：**架构 + 增强 + 加权 loss** 对小样本极其关键

---

## 2. 动机：医学分割的特殊约束

### 2.1 与自然图像（VOC）的差异

```
医学显微:
  · 标注极少（专家勾 mask 成本高）
  · 同类物体 **紧挨、粘连**（两个细胞共享边界）
  · 需要 **像素级** 准确边界（面积、形态学后续分析）
  · 图像常 **远大于 GPU 显存**（整图一次放不下）

自然图像 FCN 路线:
  · 大数据集（数千张）
  · 类间空隙大，粘连问题弱
  · 更关注语义类而非单像素形态
```

### 2.2 2015 年编解码方案的缺口

| 方法 | 问题（对医学） |
|------|----------------|
| 滑窗 CNN | 慢；边界块效应 |
| FCN | skip **相加** 且通道压到 K，**丢失** 浅层丰富特征 |
| SegNet | 无特征 skip，仅靠 indices，**细结构** 弱 |
| **U-Net** | **concat** 保留 encoder 全部通道 → 边界、小目标 |

### 2.3 U-Net 的三件套

```
1. **Concat skip**     → 分辨率 + 低层特征直达 decoder
2. **强数据增强**     → 弹性形变，等价于扩增训练集
3. **加权 loss**       → 边界与粘连区域梯度放大
```

---

## 3. 网络结构详解

### 3.1 收缩路径（Encoder / Contracting Path）

每一 **level**（共 4 次下采样 + 1 个 bottleneck）：

```
输入 feature map
  → Conv 3×3, ReLU  （无 padding 时 spatial 略缩）
  → Conv 3×3, ReLU
  → 输出 feature 存为 **skip**（供右侧 concat）
  → Max pooling 2×2, stride 2   （尺寸约 /2）
```

**通道倍增规律**（原版 EM 分割）：

| Level | 卷积后通道 | 空间尺寸（572 输入，示意） |
|-------|------------|----------------------------|
| 0 | 64 | 572 → pool → 286 |
| 1 | 128 | 286 → 140 |
| 2 | 256 | 140 → 68 |
| 3 | 512 | 68 → 32 |
| 4 (bottom) | **1024** | 32（不再 pool） |

```
设计意图:
  · 每层 **两个** 3×3 conv = 足够非线性，参数量适中
  · Pool 只放在 conv block **之后** → skip 是 **高分辨率、语义较浅** 的特征
```

### 3.2 扩张路径（Decoder / Expansive Path）

每一 level（自底向上）：

```
Step A: 上采样
  · 2×2 **Up-convolution**（转置卷积，stride 2）
  · 或 Upsample + 3×3 conv（现代实现常见）

Step B: 与 skip **拼接**
  · Concatenate along **channel** 维
  · 若上采样后尺寸与 skip 差 2~4 像素（valid conv 导致）:
    → 对 skip **中心 crop** 到与 decoder 相同 H×W（论文 Fig.2 灰色裁剪区）

Step C: 精炼
  · Conv 3×3, ReLU
  · Conv 3×3, ReLU
  · 输出通道数为上一 level 的一半（1024→512→…→64）
```

**Concat 后的通道数**：

```
例: decoder 上采样输出 512 ch，skip 512 ch
    concat → 1024 ch
    再经两个 3×3 conv 压回 512 ch

相对 FCN「相加」:
  · concat **不丢** encoder 通道信息
  · 参数量更大，但医学小图可接受
```

### 3.3 输出头

```
最顶层 decoder 输出 64×H×W
  → 1×1 convolution → C×H×W（C=2 或更多类）
  → Softmax（每像素类分布）
```

### 3.4 结构示意图（逻辑）

```
        收缩路径                    扩张路径
    ┌─── conv ───┐              ┌─── conv ───┐
    │            │─── skip ────→│   concat ←─┤ up
    └── pool ────┘              └── up ──────┘
         │                            ↑
         └──────── bottleneck ────────┘
```

---

## 4. Skip Connection：U-Net 的灵魂

### 4.1 Concat vs FCN 相加 vs SegNet 无 skip

| 方式 | 操作 | 优点 | 缺点 |
|------|------|------|------|
| **FCN** | score_up + conv(pool_i) | 参数少 | 浅层被压到 K 维，信息瓶颈 |
| **SegNet** | 仅 pool indices | 显存省 | 无浅层特征图 |
| **U-Net** | [dec_up ; enc_skip] | **细节最全** | 显存、参数量大 |

### 4.2 为何医学分割特别需要 concat

```
粘连细胞边界:
  · 深层特征: 「这里是细胞区域」（语义）
  · 浅层特征: 「边缘梯度、纹理」（定位）
  · 相加: 两类信息必须先投影到同一语义空间，易损细节
  · 拼接: decoder conv **自己学习** 如何融合语义+定位
```

### 4.3 Crop 对齐（实现必做）

```
原版用 **valid** 卷积（无 padding）:
  每级 conv 后 feature map 比输入略小
  上采样后与左侧 skip **尺寸不一致**

解决:
  从 skip 特征图 **对称裁剪** 中心区域，与 decoder 输出 H,W 一致再 concat

现代实现:
  常用 **same padding** 的 3×3 conv → 尺寸对齐更简单
  但读论文图时必须理解 **crop** 的存在
```

---

## 5. Overlap-tile 策略：大图无缝预测

### 5.1 问题

```
显微全图可达 数千×数千
  无法一次送入网络（显存）
  若互不重叠裁 tile → tile 边界预测不可靠
```

### 5.2 做法

```
1. 将大图划分为 **重叠** 的 tile（输入块）
2. 每个 tile 预测时:
     · 输入块比输出 **大**（四周加 **镜像 padding**）
     · 网络只取 **中心区域** 的预测作为有效结果
3. 相邻 tile 的 **中心区** 拼成整图 → 边界处也有充分上下文

论文示例（388 有效输出）:
  · 需要约 572×572 的输入 tile
  · 外围像素仅提供上下文，不参与 loss
```

### 5.3 示意图

```
┌─────────────────────────────┐
│  mirror pad / 上下文区       │
│   ┌─────────────────┐       │
│   │  388×388 有效    │       │  ← 写入全图对应位置
│   │  预测区          │       │
│   └─────────────────┘       │
└─────────────────────────────┘
        572×572 实际输入
```

### 5.4 与滑窗对比

```
滑窗无重叠:  快但接缝明显
Overlap-tile:  慢一些，**无缝**、边界质量高
  → 医学全图推理的事实标准（nnU-Net 等继承）
```

---

## 6. 加权损失：分离粘连实例

### 6.1 普通 CE 的不足

```
两个相邻同类细胞:
  · 像素级标签都是「细胞」
  · 中间边界像素对 loss 贡献与内部相同
  · 网络倾向 **糊成一片**，无法学出清晰分界
```

### 6.2 权重图 w(x)

论文为每个像素 x 定义权重：

```
w(x) = w_c(x) + w_0 · exp( - (d1(x)² + d2(x)²) / (2σ²) )

w_c(x):  类别平衡权重（缓解背景过多）
d1(x):   x 到 **最近** 任一边界（任意实例）的距离
d2(x):   x 到 **第二近** 边界的距离（通常来自 **另一细胞**）
σ:       约 5 像素（与图像分辨率相关）
w_0:     10（论文）
```

**直觉**：

```
· 远离所有边界（细胞内部）:  d1,d2 大 → exp 项 ≈ 0 → w ≈ w_c
· 靠近 **两实例之间** 的缝隙:  d1,d2 都小 → exp 项大 → **w 很大**
  → 梯度强迫网络学清 **实例间边界**

仅靠近单一实例外轮廓:  d2 大 → 权重适中
```

### 6.3 损失形式

```
L = - Σ_x w(x) · log p_{y(x)}(x)

p: Softmax 后真实类的概率
y(x): GT 标签

与 SegNet median frequency 类似，U-Net 把 **形态学边界** 显式写进 w(x)
```

### 6.4 预计算

```
d1, d2 由 GT mask 做距离变换 **离线** 算好
训练时查表，不增加前向开销
```

---

## 7. 数据增强：弹性形变

### 7.1 为何对 U-Net 至关重要

```
仅有 30 张标注图:
  不用增强 → 严重过拟合
  刚性增强（翻转、旋转）→ 不够

弹性形变:
  · 对图像与 mask **同步** 施加平滑随机位移场
  · 模拟细胞 **挤压、变形、接触** 形态
  · 等价于无限扩增训练分布
```

### 7.2 实现要点（论文描述）

```
1. 生成低分辨率随机位移场 Δx, Δy（高斯平滑）
2. 用双三次插值 warp 图像；用 **最近邻** warp 标签（保持离散类）
3. 保证 **微分同胚** 式形变 → mask 拓扑不变（不撕破、不粘连假影）

效果: 在 30 张图上训练仍泛化到全新显微图
```

### 7.3 其它增强

```
· 旋转、缩放、灰度扰动、弹性组合
· 论文强调 **弹性** 是胜负手，不是可有可无
```

---

## 8. 训练与推理

### 8.1 训练设置（论文 EM 任务）

| 超参 | 值 |
|------|-----|
| 优化 | SGD / Adam 类（原版 Caffe momentum） |
| 损失 | 加权 Softmax CE |
| 输入 | 572×572 tile；中心 388×388 监督 |
| 标注 | 二值或实例 mask；背景 + 前景 |
| 初始化 | **随机**（无 ImageNet 预训练，2015 医学常见） |
| 关键 | 弹性形变 + 权重图 |

### 8.2 推理

```
大图 → Overlap-tile 滑窗
  → 每 tile 中心预测拼接
  → 可选简单形态学后处理（论文主结果无 CRF）

实时性:
  原版非为实时设计；后续 U-Net-lite、nnU-Net 优化
```

### 8.3 从二分类到多类

```
结构不变: 最后 1×1 conv 输出 C 通道
加权 w(x) 可按类扩展 w_c(x)
实例分割常配合 **watershed** 对距离图再分实例（后续工作）
```

---

## 9. 与 FCN / SegNet 对比及后续变体

### 9.1 三篇 2015 论文定位

| | FCN | SegNet | U-Net |
|--|-----|--------|-------|
| 首发场景 | PASCAL VOC | CamVid 驾驶 | **ISBI 细胞** |
| Skip | 相加 | 无 | **concat** |
| 小数据 | 依赖 ImageNet 预训练 | 依赖 VGG 预训练 | **从头训 + 强增强** |
| 粘连边界 | 一般 | 一般 | **加权 loss 专门处理** |

### 9.2 常见变体（读 U-Net 之后）

| 变体 | 改动 |
|------|------|
| **U-Net++** | 嵌套密集 skip，多尺度融合 |
| **Attention U-Net** | skip 上加 attention gate |
| **3D U-Net** | 体数据 MICCAI 2016 |
| **nnU-Net** | 自适应深度、patch、预处理，医学 benchmark 默认强基线 |
| **Res-UNet / UNet++** | ResNet block 替换 double conv |

### 9.3 在自然图像上的使用

```
VOC/COCO 上常改用 ResNet-UNet、[DeepLab](./DeepLab.md)、[PSPNet](./PSPNet.md)
但 **encoder-decoder + concat skip** 已成为分割 head 默认模板
  （含 SAM、Mask2Former 的 mask decoder 思想溯源）
```

---

## 10. 消融与论文结论

| 去掉/弱化 | 影响 |
|-----------|------|
| Skip concat | 边界 IOU 显著下降 |
| 弹性形变 | 小训练集几乎无法收敛到 SOTA |
| 加权 w(x) | 粘连细胞 **合并** 预测，HeLa IOU 大跌 |
| Overlap-tile | 大图接缝伪影 |

```
论文核心论点:
  网络结构 + 训练策略 **一体设计**
  不是「只换 U 形」就能复现 92% IOU
```

---

## 11. 精读备忘：易混淆点

### 11.1 U-Net 的「U」= 对称 **路径**，不是字母本身有特殊算子

```
本质是 encoder-decoder + skip
  许多示意图画成 U 形，但实现上是 **两级 list of modules**
```

### 11.2 Concat 后通道翻倍，conv 负责融合

```
不是 concat 完就直接输出
  必须再接 **两个 3×3 conv** 降维/混合
```

### 11.3 388 vs 572 是 **tile 策略**，不是网络只能吃 572

```
网络可适配任意 H,W（需为 2^n 对齐多次 pool）
  论文数字来自 ISBI 裁剪与 valid conv 折中
```

### 11.4 加权 loss ≠ 类别权重 alone

```
w_c:  背景/前景平衡
exp项: **实例间** 边界强化（依赖 d1,d2 距离变换）
  二者缺一不可理解 HeLa 任务
```

### 11.5 与 Mask R-CNN 实例分割

```
U-Net:     通常 **语义** mask（同类粘连仍是一个类通道）
实例分离:  靠加权边界 + 后处理 watershed，或改多通道实例标签

Mask R-CNN: 先检测再 per-RoI mask，实例显式分开
```

### 11.6 现代实现常用 same padding

```
读 2015 原图务必知道 **crop**
  自己写 PyTorch 时常用 padding=1 的 3×3 简化对齐
```

---

## 12. 语义分割脉络（U-Net 位置）

```
FCN (2015)          全卷积 + 学习 up + skip 相加    → [FCN.md](./FCN.md)
U-Net (2015)        **concat skip** + 医学增强/loss  ← 本文
SegNet (2015/2017)  索引 unpool、无 skip            → [SegNet.md](./SegNet.md)
3D U-Net (2016)     体素分割
nnU-Net (2018+)     自配置 U-Net 族，医学 SOTA 基线
DeepLab             空洞 / ASPP               → [DeepLab.md](./DeepLab.md)
PSPNet              金字塔池化 PPM            → [PSPNet.md](./PSPNet.md)
HRNet               高分辨率并行融合            → [HRNet.md](./HRNet.md)
SegFormer           MiT + All-MLP               → [SegFormer.md](./SegFormer.md)
Mask2Former         mask 分类 + 掩码注意力       → [Mask2Former.md](./Mask2Former.md)
SAM                 提示式分割基础模型           → [SAM.md](./SAM.md)
OneFormer           Task Token 三任务一权重      → [OneFormer.md](./OneFormer.md)
```

**历史地位**：

```
1. **Concat skip** 成为分割 decoder 事实标准（优于 FCN 相加的细粒度任务）；
2. 证明 **数据增强 + loss 设计** 与架构同等重要（小样本典范）；
3. 催生医学影像十年主线（细胞、器官、CT/MRI → nnU-Net 生态）。
```

---

## 13. 参考资料

- 原论文：[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)（MICCAI 2015）
- 实现参考：[milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
- 对比：[FCN.md](./FCN.md)、[SegNet.md](./SegNet.md)
- 扩展：[3D U-Net](https://arxiv.org/abs/1606.06650)、[U-Net++](https://arxiv.org/abs/1807.10165)、[nnU-Net](https://arxiv.org/abs/1904.08142)
- 距离变换加权：论文引用 [Ciresan et al. 2012] 显微镜分割经验

---

*文档版本：初稿 | 对应论文 MICCAI 2015 U-Net*
