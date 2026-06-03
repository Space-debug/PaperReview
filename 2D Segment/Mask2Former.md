# Mask2Former（及 MaskFormer）

> 本文档用于整理 **MaskFormer**（Per-Pixel Classification is Not All You Need）与 **Mask2Former**（Masked-attention Mask Transformer for Universal Image Segmentation）精读笔记。  
> 重点：**Mask 分类范式**（固定 query 预测 mask + 类）、**掩码注意力（Masked Attention）**、**像素解码器多尺度融合**，以及 **语义 / 实例 / 全景** 三任务统一推理。

> 相关：[OneFormer.md](./OneFormer.md)（一次训练三任务）· [SAM.md](./SAM.md)（提示式基础模型）· [SegFormer.md](./SegFormer.md) · [../2D Detection/DETR/DETR.md](../2D Detection/DETR/DETR.md) §10（全景扩展）· [../2D Detection/RCNN/MaskRCNN.md](../2D Detection/RCNN/MaskRCNN.md)

---

## 0. 两篇论文关系（先读此节）

| 项目 | MaskFormer | Mask2Former |
|------|------------|-------------|
| 发表 | **CVPR 2021** | **CVPR 2022** |
| 核心范式 | **Mask 分类** 统一分割 | 在 MaskFormer 上 **改注意力 + 像素解码器** |
| 全景 PQ（COCO val, Swin-L） | 51.1 | **57.8** |
| 实例 AP | — | **50.1** |
| 语义 mIoU（ADE20K） | — | **57.7** |

```
MaskFormer:  证明「每像素分类」可换成「N 个 mask + 类」集合预测
Mask2Former:  用 **Masked Attention** 让 query 只看相关像素 → 更快更准
              成为 **通用分割**（Universal Segmentation）基线
```

---

## 1. 概览（Mask2Former）

### 1.1 基本信息

| 项目 | 内容 |
|------|------|
| 论文 | Masked-attention Mask Transformer for Universal Image Segmentation |
| 作者/机构 | Bowen Cheng, Ishan Misra, Alexander G. Kirillov, …（Meta FAIR） |
| 发表 | **CVPR 2022** |
| 任务 | **语义**、**实例**、**全景** 分割 **同一架构** |
| 代码 | [facebookresearch/Mask2Former](https://github.com/facebookresearch/Mask2Former)、Detectron2 |

### 1.2 核心思想（一句话）

**用 backbone 多尺度特征经像素解码器得到高分辨率 per-pixel embedding，再用固定数量 N 个可学习 query 在 Transformer 解码器中通过「掩码约束的交叉注意力」逐层 refine，每个 query 输出一个类别 logit 与一张二值 mask（点积生成），训练时用匈牙利匹配对 GT mask 集合做监督，推理时按任务把 N 个 mask 合并成语义图、实例图或全景图。**

| 对比 | 逐像素分类（FCN/SegFormer） | Mask R-CNN | DETR 全景 | **Mask2Former** |
|------|---------------------------|------------|-----------|-----------------|
| 输出 | H×W×K softmax | 每 RoI 一个 mask | box + mask + stuff 头 | **N 个 mask + 类** |
| 实例 | 需 watershed 等 | RoI Align | query 点积 mask | **query 点积 mask** |
| NMS | 语义无 | **需要** | 检测无；全景融合启发式 | **通常不需要**（实例） |
| 任务切换 | 换 head / 换数据 | 仅实例 | 多分支 | **同一套权重改推理** |

### 1.3 整体流水线

```
Input: H×W×3
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Backbone: ResNet-50 / Swin-L 等 → 多尺度 {C2,C3,C4,C5}      │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Pixel Decoder（MS-DeformAttn Encoder 或 FPN 变体）           │
│  输出: 高分辨率 mask features F_mask  (B, C_m, H/4, W/4)      │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Transformer Decoder × L 层（如 9 层，3 个 scale 各 3 层）    │
│  输入: N 个 learnable queries（默认 N=100）                   │
│  每层: Self-Attn → **Masked Cross-Attn** → FFN              │
│        → 预测 class_i, mask_i = σ(query_i · F_mask)         │
│        → 下一层 cross-attn 仅在 mask_i>0 像素上 attend       │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
训练: 匈牙利匹配 + CE(cls) + Dice(mask) + Focal(mask)
推理:
  · 语义: 按类聚合 mask
  · 实例: 过滤 ∅ 类 + 低分 query，mask 竞争赋像素
  · 全景: thing 实例 + stuff 区域 合并规则
```

### 1.4 精度参考

#### COCO（val，Swin-L backbone）

| 任务 | 指标 | MaskFormer | **Mask2Former** |
|------|------|------------|-----------------|
| 全景 | PQ | 51.1 | **57.8** |
| 实例 | AP | — | **50.1** |
| 检测式全景对比 | — | Panoptic FPN ~47 | 超越 |

#### ADE20K（语义，val mIoU）

| 模型 | mIoU |
|------|------|
| SegFormer-B5 | 51.0 |
| Swin-L + UPerNet | 53.9 |
| **Mask2Former** | **57.7** |

#### Cityscapes（全景 val PQ）

| 模型 | PQ |
|------|-----|
| **Mask2Former（Swin-L）** | **62.1** |

- **一套模型** 在全景、实例、语义上同时达到当时 **SOTA 级**

---

## 2. 动机：从「每像素分类」到「Mask 分类」

### 2.1 逐像素分类的隐含假设

```
FCN / DeepLab / SegFormer:
  每个像素独立（或靠卷积邻域）预测类别 K

问题:
  ① **实例** 需额外机制区分「同类两个物体」
  ② **全景** 要 things + stuff 两套逻辑（见 DETR §10 stuff 头）
  ③ 高分辨率下 **K 类 softmax** 与 **全局语境** 耦合方式不直观
```

见 [FCN.md](./FCN.md)、[SegFormer.md](./SegFormer.md)。

### 2.2 Mask 分类范式（MaskFormer 提出）

```
固定 N 个 slots（queries），每个 slot 预测:
  · 类别 c ∈ {1…K} 或 **∅（无对象）**
  · 一张 **二值 mask** M ∈ [0,1]^{H×W}

整图语义 = 这 N 个 (c, M) 的 **组合 / 竞争** 结果
  而非 H×W 个独立分类器

类比 DETR:
  DETR:  N 个 (class, box)
  MaskFormer:  N 个 (class, mask)   ← 去掉 box，直接 mask
```

### 2.3 Mask2Former 要修什么

```
MaskFormer 仍存在的问题:
  · Cross-attention 对 **全图所有像素** 计算 → 背景干扰大
  · 像素解码器相对简单 → 边界/小物体 mask 粗
  · 训练慢

Mask2Former:
  ① **Masked Attention**: 只在 **当前预测 mask 内** attend
  ② **多尺度可变形注意力** 像素解码器 → 更细 mask feature
  ③ 去掉 MaskFormer 中 **多余组件**（如 transformer encoder 冗余）
```

---

## 3. Mask 分类范式详解

### 3.1 输出表示

```
预测集合:  {(c_i, M_i)}_{i=1}^N

c_i:     K+1 维分类（含 no-object ∅）
M_i:     与 F_mask 做点积:  M_i = σ( Linear(q_i) · F_mask(x,y) )

σ: sigmoid；训练可用 BCE / Dice / Focal
```

**与 Mask R-CNN 对比**：

```
Mask R-CNN:  先 box → crop RoI → 小 mask CNN
Mask2Former:  **全图共享** F_mask，query 向量决定 mask 形状
              无 RoI、无 align
```

见 [MaskRCNN.md](../2D Detection/RCNN/MaskRCNN.md)。

### 3.2 匈牙利匹配（训练）

```
GT:  变长集合 {(c*, M*)}（实例为 K 个 mask；语义可为每类一张 mask）

代价矩阵 C[i,j] = λ_cls·CE(c_i, c*_j) + λ_mask·(Dice + Focal)(M_i, M*_j)

匈牙利算法 → 一对一匹配
未匹配 query → 监督为 **∅ 类**

与 DETR 相同 **集合预测** 哲学，见 [DETR.md](../2D Detection/DETR/DETR.md)
```

### 3.3 为何实例可不用 NMS

```
每个 query 学不同 **mask 形状**
匹配训练鼓励 **一个 query 对一个实例**
推理时:
  按 score 排序，像素归属 **最高分 mask**（或 mask 竞争）
  重叠由 mask 概率与阈值处理，**非 box IoU-NMS**

注意: 极度拥挤时仍可能需启发式；COCO 上官方报告 **无需 NMS** 即 SOTA
```

---

## 4. 网络结构详解（Mask2Former）

### 4.1 Backbone

```
常用:
  · ResNet-50（轻量实验）
  · **Swin-L**（精度 SOTA 配置）

输出多尺度特征:
  res2 (1/4), res3 (1/8), res4 (1/16), res5 (1/32)
```

### 4.2 Pixel Decoder（像素解码器）

```
目标:  生成 **高分辨率、每像素 embedding** F_mask

实现（论文主配置）:
  **Multi-Scale Deformable Attention Encoder**（源自 Deformable DETR）
  · 将 {C3,C4,C5} 或更多尺度展平为 token
  · 可变形 attention 融合多尺度
  · 上采样到 **1/4 输入分辨率**，通道 C_m（如 256）

对比 MaskFormer:
  MaskFormer 像素解码器较简 → Mask2Former **显著加强**

作用:
  所有 query 的 mask 点积 **共享** 这一张高分辨率特征图
  → 边界比 DETR 步长 32 的 coarse mask 更细
```

### 4.3 Transformer Decoder 与 Masked Attention

**Query 初始化**：

```
N 个可学习 query embedding（+ 可选 query position）
默认 N=100（COCO）；ADE20K 语义可用类似数量
```

**每层 block**（重复 L 次）：

```
1) Self-Attention
     queries 之间交互，避免多个 query 预测同一物体

2) **Masked Cross-Attention**（Mask2Former 核心）
     Q: queries
     K,V: 来自 F_mask 展平的像素 token

     掩码来源:  **上一层** 预测的 mask_i
     M̂_i = (σ(mask_logits_i) > 0.5)   或软掩码变体

     Attention 权重在 **M̂_i=0 的像素上设为 -∞**（不参与）
     → query 只「看见」自己当前认为属于实例/区域的像素

3) FFN

4) 预测头
     class: Linear → K+1
     mask:  Linear(q) · F_mask → H/4×W/4 → 上采样到 H×W
```

**直觉**：

```
普通 Cross-Attn:  query 被全图背景稀释 → 难聚焦物体
Masked Cross-Attn:  逐层 **缩小注意范围** → 类似迭代 refine ROI
  与 RoI Align 不同:  **软、可学习、无矩形框**
```

### 4.4 多尺度 Decoder 设计（可选实现细节）

```
论文将 decoder 分为对 **1/8, 1/16, 1/32** 特征的分段处理:
  在不同分辨率特征上交替 refine
  → 小物体用大分辨率特征上的 mask 点积

实现见 Detectron2 `MaskFormerTransformerDecoder`
```

---

## 5. 三任务推理：同一权重，不同合并

### 5.1 语义分割（Semantic）

```
对每个非 ∅ query (c_i, M_i, s_i=score):

方法（常见）:
  构造 per-class 概率图:
    对每个像素 p，score[p,c] = max_{i: c_i=c} s_i · M_i(p)
  或累加所有 mask 加权

argmax_c → 语义标签图

**无需** K 个独立 softmax 通道互斥于像素；由 mask 竞争产生
```

### 5.2 实例分割（Instance）

```
1. 过滤 c_i = ∅ 或 s_i < 阈值
2. 对每个剩余 query 得到二值 mask（阈值 0.5）
3. 像素级:  assign 到 **最高 s_i·M_i(p)** 的 query（若竞争）
4. 输出 (class, instance_id, mask)

**无 box**；评估用 mask AP（COCO）

相对 Mask R-CNN:  端到端、无 RoI、无 mask NMS（官方设定）
```

### 5.3 全景分割（Panoptic）

```
Things（可数）:
  与实例相同，每 query 一个实例 id

Stuff（不可数）:
  每 **stuff 类** 通常 **一个 query / 一张 mask** 覆盖全区
  （训练标注按类合并）

融合:
  按 PQ 标准:  things 优先（按 score 降序画 mask）
  stuff 填充未被占据像素
  重叠处理:  score 高的覆盖低的

对比 DETR 全景（[DETR.md](../2D Detection/DETR/DETR.md) §10）:
  DETR:  things 用 query mask + **单独 stuff 语义头**
  Mask2Former:  **全部用 mask query** 统一表示（stuff 也是 mask）
```

---

## 6. 损失函数与训练

### 6.1 总损失

```
对匹配上的 query:
  L = λ_ce · CE(c_i, c*) + λ_dice · Dice(M_i, M*) + λ_focal · Focal(M_i, M*)

未匹配 query:
  分类为 ∅（背景类）

Deep Supervision（可选）:
  每个 decoder 中间层也出 mask/cls → 辅助 loss（权重递减）
```

**Dice / Focal**：

```
缓解 mask **前景背景极度不平衡**
  与 DETR mask head、Mask R-CNN mask loss 同族
```

### 6.2 超参（COCO 典型）

| 超参 | 值 |
|------|-----|
| 优化器 | AdamW |
| LR | 1e-4 backbone 更低，head 更高（layer decay） |
| 训练步数 | 90k ~ 160k |
| 输入 | 短边 800~1024 多尺度 |
| N queries | 100 |
| λ | cls=2, mask dice=5, focal=5 等（实现略有出入） |

### 6.3 数据与任务切换

```
同一网络结构:
  · 换训练集标注形式（语义 / 实例 / 全景）
  · 换匹配与 GT mask 构造方式
  · **推理脚本** 不同（semantic.py / instance.py / panoptic.py）

权重可 **在全景上训练，部署到语义**（论文强调通用性）
```

---

## 7. 与 DETR / MaskFormer / CNN 分割对比

### 7.1 演进链

```
FCN/PSP/DeepLab  →  每像素分类
DETR 检测        →  集合预测 + Hungarian
DETR + mask 头   →  query 点积 coarse mask（§10）
MaskFormer       →  去掉 box，纯 mask 分类 + 全景
Mask2Former      →  Masked Attn + 强 pixel decoder
```

### 7.2 总表

| | SegFormer | DETR Panoptic | MaskFormer | **Mask2Former** |
|--|-----------|---------------|------------|-----------------|
| 语义 | 主战场 | stuff 头 | 支持 | **支持** |
| 实例 | 弱 | query mask | 支持 | **SOTA 级** |
| 全景 | 需另建 | 双分支 | 51.1 PQ | **57.8 PQ** |
| 解码 | All-MLP | Encoder 32 stride | Transformer | **Masked Trans.** |
| 速度 | 快 | 慢 | 中 | **较 MaskFormer 快** |

### 7.3 相对 SegFormer 的语义提升来源

```
SegFormer:  每像素独立分类 + 轻量 decoder
Mask2Former:
  · **对象级 query** 显式建模「一块区域一个语义」
  · 高分辨率 mask feature + 多层 refine
  → ADE20K 上 **+6.7 mIoU** 量级（51→57.7）
```

---

## 8. 消融与论文结论

| 去掉 | 影响 |
|------|------|
| Masked Attention | PQ / AP **明显下降**；训练更慢 |
| 多尺度 pixel decoder | 小物体 mask 差 |
| MS-DeformAttn vs 简单 FPN | 大场景 / 多尺度 受益 |
| 深监督 | 收敛与精度略降 |

```
MaskFormer → Mask2Former:
  组件 **更少** 但 **更有效**（去掉冗余 encoder 等）
  → 证明「**注意力掩码**」比堆更多全局 attention 更关键
```

---

## 9. 精读备忘：易混淆点

### 9.1 Mask 分类 ≠ 只预测 mask 不预测类

```
每个 query **同时** 输出 class + mask
  类用于匹配与语义聚合；mask 用于像素归属
```

### 9.2 Masked Attention 的 mask 来自 **预测**，非 GT

```
训练时:  用 **上一层预测 mask**（detach 或 stop-gradient 依实现）
  推理时:  同样自预测迭代
  **不是** 训练时把 GT mask 塞进 attention（那是作弊）
```

### 9.3 N=100 不是「最多 100 类」

```
N = query 槽位数；COCO 80 类可 >100 实例吗？—— 单图实例通常 <100
  极度拥挤场景可能不够（与 DETR 100 queries 同限）
```

### 9.4 语义训练 GT 如何变成 mask 集合

```
ADE20K 语义:  常将 **每个类别** 合并为一张二值 GT mask（每类最多 1 张）
  或按连通域拆多个 mask
  实现决定匈牙利匹配的对象数量
```

### 9.5 与 SAM 的区别

```
SAM:  promptable、零样本分割、无固定类别  → [SAM.md](./SAM.md)
Mask2Former:  **闭集分类** + 固定 K 类，全监督 SOTA 分割器
```

### 9.6 点积 mask 与 DETR 一致

```
技术 lineage:
  DETR pred_masks = einsum(query_emb, encoder_2d)
  Mask2Former     = einsum(query_emb, **pixel_decoder**_out)
  差别在 **F_mask 分辨率与生成方式**
```

---

## 10. 后续工作

| 方向 | 代表 |
|------|------|
| 视频 | Mask2Former-VIS、Tube-Link |
| 三任务单权重 | [OneFormer](./OneFormer.md)（Task Token + 联合训练） |
| 开放词汇 | Open-Vocabulary Mask2Former；提示式见 [SAM.md](./SAM.md) |
| 统一 | OneFormer（任务 token 条件化） |
| 检测 | Mask DINO（检测+分割） |
| 3D | Mask3D |

---

## 11. 语义分割脉络（Mask2Former 位置）

```
CNN 逐像素:  FCN → … → SegFormer           → 各 .md
集合 + mask:  DETR §10 → MaskFormer (2021)
              Mask2Former (2022)            ← 本文
OneFormer      任务条件化统一              → [OneFormer.md](./OneFormer.md)
```

---

## 12. 参考资料

- Mask2Former：[arxiv:2112.01527](https://arxiv.org/abs/2112.01527)（CVPR 2022）
- MaskFormer：[Per-Pixel Classification is Not All You Need](https://arxiv.org/abs/2101.07551)（CVPR 2021）
- 代码：[facebookresearch/Mask2Former](https://github.com/facebookresearch/Mask2Former)
- 前置：[DETR.md](../2D Detection/DETR/DETR.md) §10、[MaskRCNN.md](../2D Detection/RCNN/MaskRCNN.md)
- 对比：[SegFormer.md](./SegFormer.md)、[DeepLab.md](./DeepLab.md)
- 后续：[OneFormer.md](./OneFormer.md)

---

*文档版本：初稿 | MaskFormer CVPR 2021 + Mask2Former CVPR 2022*
