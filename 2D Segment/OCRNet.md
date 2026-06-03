# OCRNet（Object-Contextual Representations）

> 本文档用于整理 OCRNet（Object-Contextual Representations for Semantic Segmentation）论文精读笔记。  
> 重点：**按类软物体区域 → 区域表征 → 像素–区域关系加权聚合** 三步上下文建模，以及与 [PSPNet.md](./PSPNet.md) / [DeepLab.md](./DeepLab.md)（多尺度）、DANet/OCNet（像素–像素关系）的差异；工程上常与 [HRNet.md](./HRNet.md) **HRNetV2-W48** 组合为 **HRNet+OCR**。

> 相关：[HRNet.md](./HRNet.md) · [DeepLab.md](./DeepLab.md) · [PSPNet.md](./PSPNet.md)

---

## 1. 概览

### 1.1 基本信息

| 项目 | 内容 |
|------|------|
| 论文 | Object-Contextual Representations for Semantic Segmentation |
| 作者/机构 | Yuhui Yuan, Xilin Chen, Jingdong Wang（中科院 / MSRA） |
| 发表 | **ECCV 2020**（arXiv 1909.11065） |
| 扩展 | *Segmentation Transformer*（用 Transformer 语言重述 OCR，等价 encoder–decoder） |
| 代码 | [HRNet-OCR](https://github.com/HRNet/HRNet-Semantic-Segmentation)、[openseg](https://git.io/openseg) |

### 1.2 核心思想（一句话）

**像素类别 = 其所属「物体」（thing + stuff）的类别；先用骨干特征预测 K 张按类的软分割图（粗物体区域），对每类做空间加权聚合得到 K 个物体区域向量，再算每个像素与 K 个区域的相似度，把区域向量加权合成物体语境特征 OCR，与原像素特征融合后做最终分割。**

| 对比 | ASPP / PPM | DANet / Self-Attention | **OCRNet** |
|------|------------|------------------------|------------|
| 上下文按什么分 | **空间尺度**（远近） | **像素–像素** 相似 | **物体类别区域** |
| 同类像素 | 与异类混在同一尺度池 | 按特征相似聚 | **显式按类区域聚合** |
| 区域监督 | 无 | 大多无 | **GT 分割监督粗分割** |
| 参数量（模块） | 较大 | 很大（DANet） | **~10.5M，较轻** |

### 1.3 整体流水线

```
Input Image
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Backbone: Dilated ResNet-101 (OS=8) 或 HRNetV2-W48 (OS=4)   │
│  像素特征 X:  H'×W'×C                                         │
└──────────────────────────────────────────────────────────────┘
    │
    ├─ ResNet-101:  Stage3 特征 → 粗分割头 → 软物体区域 {M_k}
    │               Stage4 特征 → 3×3 conv → 送入 OCR
    │
    └─ HRNet-W48:   最终多分支融合特征 → 粗分割 + OCR（单路）
    │
    ▼
┌─ Step 1: 软物体区域 Soft Object Regions ─────────────────────┐
│  1×1 conv → K 类 softmax 图 M_k（每类一张「属于该类」热力图）    │
│  训练:  L_coarse = CE(粗预测, GT)                             │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌─ Step 2: 物体区域表征 Object Region Repr. f_k ────────────────┐
│  f_k = Σ_i  m̃_{ki} · x_i   （m̃ 为 M_k 上 spatial softmax）   │
│  K 个向量，每类一个「物体级」摘要                              │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌─ Step 3: 物体语境 OCR + 增强特征 ────────────────────────────┐
│  w_{ik} = softmax_k( φ(x_i)^T ψ(f_k) )                       │
│  y_i = ρ( Σ_k w_{ik} · δ(f_k) )     ← 物体语境向量            │
│  z_i = g( concat(x_i, y_i) )        ← 融合后送最终分类头      │
│  L_final = CE(最终预测, GT)                                   │
└──────────────────────────────────────────────────────────────┘
```

### 1.4 精度参考

#### Cityscapes（mIoU）

| 配置 | test（论文 Table 5） | 备注 |
|------|----------------------|------|
| HRNetV2-W48 baseline | 81.6 | 无 OCR |
| **HRNetV2-W48 + OCR** | **82.4**（无 coarse 额外数据）/ **83.0**（+ coarse） | 单模型 |
| + Mapillary 预训练 | **84.2** test | |
| **HRNet + OCR + SegFix** | **84.5** test | ECCV 2020 提交 **榜首** |
| 社区 val 强配置 | **~84.9** val | 与多尺度/额外数据等有关 |

#### 其它基准（HRNet-W48 + OCR，val/test）

| 数据集 | mIoU（约） |
|--------|------------|
| ADE20K | **45.66** |
| LIP | **56.65** |
| PASCAL-Context | **56.2** |
| COCO-Stuff | **40.5** |

#### 同 backbone 增益（ResNet-101 OS=8，Cityscapes test）

```
Baseline → +OCR:  约 **+0.8%**（相对 ASPP）~ **+1.5%**（相对 PPM）
GT-OCR（用 GT 区域）上界远高于 OCR → 证明 **物体区域** 假设极强
```

- OCR 模块 **FLOPs/显存/延时** 优于 PPM、ASPP、DANet（论文 Table 4，~45ms vs 97–121ms）

---

## 2. 动机：像素标签本质是「物体类别」

### 2.1 多尺度语境的盲点

```
ASPP / PPM（见 DeepLab、PSPNet）:
  按 **空间距离** 采样上下文
  ·  dilation 12 的像素可能同时落在 **车** 和 **路** 上
  · 无法区分「同类物体像素」与「异类但相邻」像素

图 2 对比（论文）:
  ASPP:  多尺度邻域 ■ 散布在物体+背景
  OCR:   期望上下文集中在 **同色同类区域**（整辆车）
```

### 2.2 像素–像素关系的盲点

```
DANet / OCNet / Self-Attention:
  y_i = Σ_s  w(x_i, x_s) · x_s
  · 上下文是 **所有像素** 或全局
  · 区域结构 **无监督** 形成，不保证对齐「语义物体」

ACFNet / Double Attention:
  有区域聚合，但区域常 **不对应物体类**，或关系只从 x_i 预测
```

### 2.3 OCR 的命题

```
若已知像素属于哪一类 **物体区域**，
  用该 **整类区域的全局表征** 增强像素特征 → 分割大幅变好（GT-OCR 实验）

OCRNet:  用网络 **学出** 软区域 + 区域向量 + 像素–区域注意力
```

---

## 3. 公式与三步详解

### 3.1 软物体区域（Step 1）

```
从骨干中间特征用 1×1 conv 预测:
  M_k ∈ [0,1]^{H×W}   k = 1…K（K 为语义类数）

训练:
  与 GT 分割做 **像素级交叉熵** → 粗分割可监督

推理:
  M_k 为 **软** 权重图（非硬 mask）
  thing 类可多峰（多实例同类合并到 **同一类通道**）
  → 语义分割设定下，**同类实例共享一个 M_k**
```

**与实例分割区别**：

```
OCR 的「物体区域」= **语义类区域**（所有车像素进同一 M_car）
  不区分第 1 辆车与第 2 辆车
  因而模块用于 **语义分割**，非 per-instance
```

### 3.2 物体区域表征（Step 2）

```
f_k = Σ_{i∈I}  m̃_{ki} · x_i

m̃_{ki}:  M_k 在位置 i 经 **spatial softmax** 归一
  → 强调该类激活最强的空间支持

f_k:  第 k 类的 **全局物体向量**（整图一个 k 一个）
```

### 3.3 物体语境 OCR（Step 3）

```
关系（像素 i 对区域 k）:
  w_{ik} = exp( κ(x_i, f_k) ) / Σ_j exp( κ(x_i, f_j) )
  κ(x,f) = φ(x)^T ψ(f)     （1×1 conv→BN→ReLU，256 维）

物体语境:
  y_i = ρ( Σ_k w_{ik} · δ(f_k) )   （δ, ρ 同为 1×1→BN→ReLU，512 维）

融合:
  z_i = g( [x_i ; y_i] )   → 最终 1×1 分类 → K 类 logits
```

**与 Self-Attention 对照**：

```
Self-Attn:  key/value 来自 **其它像素** x_s
OCR:       key/value 来自 **K 个区域向量** f_k（仅 K 个，K≈150）
  → 计算量 O(HW·K) 远小于 O((HW)²)
  → 论文称更高效
```

### 3.4 Transformer 等价视角（Segmentation Transformer）

```
重述为 Encoder–Decoder:
  · **Decoder cross-attn**:  类别 query = 粗分割 1×1 权重 → 输出 f_k
  · **Encoder cross-attn**:  query = 像素 x_i，KV = 区域表征

→ 与 Mask2Former / DETR 的 query–memory 同族，但 **K 固定为类数** 且 **有粗分割监督**
```

---

## 4. 与 HRNet 的组合（工程标配）

### 4.1 为何常写 HRNet+OCR

```
HRNet:  全程高分辨率特征（见 HRNet.md）
OCR:    在 **高分辨率特征** 上做类级区域聚合

Cityscapes SOTA 配置:
  Backbone: HRNetV2-W48, OS=4
  Head:     HRNet 多分支 concat → **OCR 模块** → cls
  可选:     Mapillary 预训练、SegFix 边界 refine、多尺度测试
```

### 4.2 ResNet-101 双分支（论文原始）

```
Stage 3 特征 → 仅用于 **预测粗分割** M_k
Stage 4 特征 → 3×3 conv (512 ch) → OCR 的像素特征 X

HRNet 上实验发现:
  直接加 PPM/ASPP 到 HRNet **反而掉点**
  **OCR 一致涨点** → 高分辨率骨干与「类区域语境」更合拍
```

### 4.3 SegFix（提交套件，非 OCR 本体）

```
SegFix (ECCV 2020 同期):
  模型无关的 **边界细化** 后处理
HRNet + OCR + SegFix → Cityscapes test **84.5%** 榜首

读 OCRNet 时 SegFix 视为 **可选后处理**，非 OCR 模块一部分
```

---

## 5. 损失函数与训练

### 5.1 总损失

```
L = L_coarse + L_final

二者均为 **逐像素 Cross-Entropy**（忽略 void label）

无额外对比损失 / 无匈牙利匹配（与 Mask2Former 不同）
```

### 5.2 训练设置（典型）

| 超参 | Cityscapes / 通用 |
|------|-------------------|
| 优化器 | SGD momentum 0.9 |
| LR | poly decay，base 0.01 量级 |
| Crop | 512×1024（Cityscapes） |
| 迭代 | 90k~110k |
| Backbone | ImageNet 预训练 HRNet-W48 或 ResNet-101 |

### 5.3 推理

```
单尺度或 **多尺度 + flip**（刷榜配置）
粗分割分支 **可丢弃**，仅 forward OCR+cls（实现依赖 repo）
```

---

## 6. 与 PSPNet / DeepLab / HRNet / Mask2Former 对比

| 维度 | PSPNet PPM | DeepLab ASPP | HRNet only | Mask2Former | **OCRNet** |
|------|------------|--------------|------------|-------------|------------|
| 上下文类型 | 几何多尺度 | 空洞多尺度 | 高分辨率卷积 | query mask | **类级区域** |
| 依赖粗预测 | 否 | 否 | 否 | 否 | **是**（训练监督） |
| 任务 | 语义 | 语义 | 语义/姿态 | 语义无/实例/全景 | **语义**（可扩 panoptic） |
| 与 HRNet | 可接 | 可接 | 基线 | 可换 backbone | **官方最强组合** |

```
Panoptic-FPN + OCR（论文 §5）:
  COCO val PQ **44.2%** → 证明 OCR 是 **可插拔语境模块**
```

---

## 7. 消融与论文结论

| 实验 | 结论 |
|------|------|
| GT-OCR（GT 区域） | 远超 OCR → 区域质量上限高 |
| OCR vs PPM/ASPP（同 ResNet-101） | **+0.7~1.5%** mIoU |
| OCR vs DANet / Self-Attn | 精度更高且 **显存/FLOPs/时间更低** |
| HRNet + PPM/ASPP | **掉点**；HRNet + OCR **+0.8~1.4%** |
| 像素–区域关系用 φ(x)^T ψ(f) | 优于只用 x_i 预测关系 |

---

## 8. 精读备忘：易混淆点

### 8.1 OCR ≠ Optical Character Recognition

```
本仓库 OCR = **Object-Contextual Representations**
  与文字识别无关；见 HRNet.md §10.5
```

### 8.2 「Object」含 stuff

```
天空、路面等 **stuff** 也各对应一张 M_k
  与 panoptic 的 thing/stuff 划分一致，但 OCR 头仍是 **语义类通道**
```

### 8.3 不是实例分割

```
同类两辆车 **共用一个** f_car
  要 instance 需另接实例 head（如 Panoptic-FPN）
```

### 8.4 与 OCNet 关系

```
OCNet（Yuan 等, arXiv 1809）为 **自注意力语境** 前身
OCRNet 改为 **显式 K 类区域 + 监督粗分割**
  作者同组，读文献时勿混为一谈
```

### 8.5 粗分割差时 OCR 受损

```
推理时 M_k 来自 **预测** 粗分割
  粗分割错 → f_k 偏 → w_ik 乱
  训练用 GT 监督粗分支缓解
```

### 8.6 HRNet 文档中的 OCR 节

```
骨干细节见 [HRNet.md](./HRNet.md) §5（简述）
  本文是 **OCR 模块专篇**
```

---

## 9. 后续影响

| 方向 | 说明 |
|------|------|
| **HRNet + OCR** | Cityscapes 多年强基线；后接 HMSA 等达 85.4% |
| Transformer 分割 | OCR 被表述为 **Seg Transformer** 的 cross-attn |
| K-Net / Mask2Former | 更强「区域/实例」建模范式，OCR 仍作 **轻量语境** 参考 |
| SAM / 基础模型 | 提示分割取代部分「手工语境模块」场景 |

---

## 10. 语义分割脉络（OCRNet 位置）

```
多尺度语境:  DeepLab ASPP / PSPNet PPM
高分辨率:    HRNet (2019)                    → HRNet.md
类级区域语境: OCRNet (2020)                   ← 本文
集合预测:    MaskFormer → Mask2Former         → Mask2Former.md
提示式:      SAM 系列                         → SAM.md
```

---

## 11. 参考资料

- 原论文：[Object-Contextual Representations](https://arxiv.org/abs/1909.11065)（ECCV 2020）
- Transformer 重述：[Segmentation Transformer](https://arxiv.org/abs/1909.11065)（同 arXiv 扩展）
- 代码：[HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation)
- 边界 refine：[SegFix](https://arxiv.org/abs/2004.13167)
- 前置：[HRNet.md](./HRNet.md)、[PSPNet.md](./PSPNet.md)、[DeepLab.md](./DeepLab.md)
- 对比：DANet、OCNet、ACFNet

---

*文档版本：初稿 | 对应论文 ECCV 2020 OCRNet / HRNet+OCR*
