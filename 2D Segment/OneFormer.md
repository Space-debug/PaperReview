# OneFormer

> 本文档用于整理 OneFormer（One Transformer to Rule Universal Image Segmentation）论文精读笔记。  
> 重点：**一次训练、一个权重** 覆盖语义/实例/全景三任务、**Task Token 条件化**、**Query–Text 对比损失**，以及相对 [Mask2Former.md](./Mask2Former.md)「三任务各训一版」的 **真·统一** 差异。

> 相关：[Mask2Former.md](./Mask2Former.md) · [SegFormer.md](./SegFormer.md) · [SAM.md](./SAM.md)（提示式基础模型，非闭集三任务）

---

## 1. 概览

### 1.1 基本信息

| 项目 | 内容 |
|------|------|
| 论文 | OneFormer: One Transformer to Rule Universal Image Segmentation |
| 作者/机构 | Jitesh Jain, Jiachen Li, Mang Tik Chiu, Ali Hassani, …, Humphrey Shi（SHI Labs @ Oregon / UIUC 等） |
| 发表 | **CVPR 2023**（arXiv 2211.06220） |
| 任务 | **语义 + 实例 + 全景** 三任务 **单模型** |
| 代码 | [SHI-Labs/OneFormer](https://github.com/SHI-Labs/OneFormer)、Detectron2 / MMSeg 集成 |

### 1.2 核心思想（一句话）

**在 Mask2Former 式「mask 分类 + 像素解码器 + 多尺度 masked decoder」骨架上，训练时随机采样任务并输入文本 `the task is {semantic|instance|panoptic}` 得到 Task Token，用其初始化 object queries，再与由 GT 生成的文本 query 做双向对比学习；GT 全部从 **全景标注派生**，从而 **只训一次、只存一个模型**，推理时切换 task 字符串即可换任务。**

| 对比 | Mask2Former | **OneFormer** |
|------|-------------|---------------|
| 训练次数 | 每数据集 **×3**（每任务一训） | **×1** 联合训练 |
| 推理权重 | 3 套 checkpoint | **1 套** |
| 任务告知网络 | 无（靠不同 GT 匹配） | **Task Token + 文本对比** |
| 训练标注 | 各任务专用 GT 文件 | **仅 panoptic**，在线派生 |
| ADE20K 语义（Swin-L 级） | 57.7 mIoU（专训） | 联合训 **≥ 专训 Mask2Former** |

### 1.3 整体流水线

```
Input:  Image I  +  Task text  "the task is {semantic|instance|panoptic}"
              │                    │
              │                    ▼
              │            Text Encoder → Task Token Q_task (1×d)
              ▼
┌──────────────────────────────────────────────────────────────┐
│  Backbone (Swin / ConvNeXt / DiNAT) + Pixel Decoder (MSDeformAttn) │
│  多尺度特征 F_{1/4}, F_{1/8}, F_{1/16}, F_{1/32}              │
└──────────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────┐
│  Query 构建（训练）                                           │
│  · Object queries Q:  repeat Q_task → (N-1)×d，经 2-layer    │
│    Transformer 融合 F_{1/4}，再 concat Q_task → N×d          │
│  · Text queries Q_text:  由 GT 生成 T_pad → 6-layer text enc │
│    + learnable Q_ctx；**仅训练用，推理丢弃**                  │
│  · L_{Q↔Q_text}:  双向 InfoNCE 式对比损失                    │
└──────────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────┐
│  Multi-scale Masked Transformer Decoder（同 Mask2Former）     │
│  在 1/8, 1/16, 1/32 上交替 Masked Cross-Attn + Self-Attn    │
│  → class (K+1) + mask = einsum(Q, F_{1/4})                  │
└──────────────────────────────────────────────────────────────┘
              │
              ▼
训练:  L = 0.5·L_contrast + 2·L_cls + 5·L_bce + 5·L_dice
      匈牙利匹配（GT 由 panoptic 按当前 task 派生）
推理:  固定 task 文本 → 无 text mapper → 后处理同 Mask2Former
```

### 1.4 精度参考（论文主表，相对「分任务专训」Mask2Former）

#### ADE20K val（语义 mIoU，多尺度测试）

| 方法 | 训练方式 | mIoU（约） |
|------|----------|------------|
| Mask2Former-Semantic（Swin-L） | **仅语义** 160k | 57.3 |
| Mask2Former（专训语义） | 单任务 | **57.7** |
| **OneFormer**（Swin-L） | **联合** 160k | 57.7 |
| **OneFormer**（DiNAT-L） | 联合 | **58.7** |

#### 资源对比（ADE20K，论文论述）

```
Mask2Former:  语义 + 实例 + 全景 各训 160k → 合计 **480k iter**，**3 个模型**
OneFormer:    联合训 **160k iter**，**1 个模型**
  → 训练时间、存储约 **3×** 节省，且指标不低于专训 Mask2Former
```

#### Cityscapes / COCO

- **Cityscapes**：语义 mIoU、实例 AP、全景 PQ 均 **超过** 同 backbone 下分任务 Mask2Former
- **COCO 全景**：Swin-L 配置 PQ 达 **57.4+**（见论文 Table；ConvNeXt/DiNAT 更高）

---

## 2. 动机：「能换任务 infer」≠「真统一」

### 2.1 半统一 vs 真统一

```
MaskFormer / Mask2Former（图 1b）:
  · **同一架构** 可接三种 head 推理
  · 但 SOTA 要对 **每个任务单独训练** 一套权重
  · ADE20K 上 ≈ 480k iterations、3× 存储

OneFormer（图 1c）:
  · **一次联合训练** → 三任务 SOTA
  · 落实 Panoptic Segmentation 最初「统一语义+实例」愿景
```

见 [Mask2Former.md](./Mask2Former.md) §0–§1。

### 2.2 联合训练失败假设（论文）

```
无任务条件时，同一套 queries 要同时学:
  · 语义:  每类 **一片** 连通 mask（stuff 合并）
  · 实例:  仅 **thing**，每实例一 mask，忽略 stuff
  · 全景:  thing 逐实例 + stuff 每类一片

→ 目标冲突 → 联合训练掉点（论文 Tab.7 复现 Mask2Former joint 掉点）

OneFormer 解法:
  ① 显式 **Task Token** 告诉网络「当前优化哪种目标」
  ② **Query–Text 对比** 对齐「图像 query」与「GT 文本描述」
```

---

## 3. Task-Conditioned Joint Training

### 3.1 每步训练采样

```
对每张图:
  1. 均匀采样 task ∈ {semantic, instance, panoptic}   (p=1/3)
  2. 构造 task 文本:  I_task = "the task is {task}"
  3. 从 **全景 GT** 派生该 task 的 mask 集合与类别标签
  4. 构造文本列表 T_list → padding 为 T_pad（长度 N_text）
```

### 3.2 从 Panoptic 派生三种 GT（关键细节）

| Task | GT mask 集合规则 |
|------|------------------|
| **Semantic** | 每个出现类别 **1 张** 二值 mask（同类合并，含 stuff） |
| **Instance** | 仅 **thing** 类，每个实例 **1 张** mask；**无 stuff** |
| **Panoptic** | stuff：每类 **1 张**；thing：每实例 **1 张** |

```
训练数据准备:
  只需下载 **panoptic 标注**
  不必维护三份独立 label 目录（工程上大幅简化）
```

### 3.3 文本列表与 padding（Fig.3）

```
对 GT 中每个二值 mask:
  模板  "a photo with a {CLS}"

T_pad 长度固定 N_text:
  不足部分用  "a/an {task} photo"  填充
  → 对应 **no-object** query（与 DETR ∅ 类似）

例（semantic, 图中有 cat、sky）:
  ["a photo with a cat", "a photo with a sky", "a semantic photo", …]
```

---

## 4. Task Token 与 Query 表示

### 4.1 Task Token `Q_task`

```
输入:  "the task is panoptic"
  → 文本 tokenizer + 映射 →  **1×d** 向量 Q_task

用途:
  ① 复制 (N-1) 份初始化 object queries Q'
  ② Q' 与展平的 F_{1/4} 过 **2 层 Transformer** 注入图像上下文
  ③ 输出与 Q_task **concat** → 最终 N 个 object queries Q
  ④ 解码器第一路 token 常保留 Q_task（task-dynamic）
```

**与 DETR 全零初始化对比**：

```
任务条件初始化 + concat Q_task
  → 消融中 **联合多任务训练可行** 的关键（论文 §4.3）
```

### 4.2 Text Queries `Q_text`（仅训练）

```
T_pad → 6-layer Text Encoder（类 CLIP 文本塔）
  → N_text 条 embedding
  → concat **N_ctx** 个可学习 context embedding Q_ctx
  → 共 N 个 Q_text

推理:
  **整段 Text Mapper 丢弃** → 参数量与 Mask2Former 推理相当
```

### 4.3 Query–Text 对比损失

```
batch 内 B 对 (q_obj, q_txt):
  L_{Q→Q_text} = -log exp(q_i^obj · q_i^txt / τ) / Σ_j exp(...)
  L_{Q_text→Q} = 对称项
  L_{Q↔Q_text} = 二者之和

τ: 可学习温度

作用:
  · **任务间**: semantic / instance / panoptic 的 query 分布拉开
  · **类间**: 减少 mask 分类 head 的类别混淆
```

---

## 5. 其余结构（继承 Mask2Former）

### 5.1 Pixel Decoder

```
**Multi-Scale Deformable Attention**（Deformable DETR 系）
  融合 backbone 多尺度 → 输出 F_{1/4}…F_{1/32}
  与 Mask2Former 相同选择
```

### 5.2 Transformer Decoder

```
多尺度 **Masked Cross-Attention**（见 Mask2Former §4.3）:
  在 1/8、1/16、1/32 特征上交替更新 Q
  + Self-Attention + FFN，重复 L 层

Mask 预测:
  M_i = σ( einsum(Q_i, F_{1/4}) )
  上采样到原图
```

### 5.3 损失与匹配

```
L_final = 0.5·L_{Q↔Q_text} + 2·L_cls + 5·L_bce + 5·L_dice

匹配:  匈牙利（class + mask cost），与 Mask2Former 相同
∅ 类:   L_cls 权重 0.1
```

### 5.4 推理与后处理

```
用户指定 task 字符串（或嵌入）:
  "the task is semantic"  →  语义图合并规则
  "the task is instance"  →  实例过滤 + mask 竞争
  "the task is panoptic"  →  PQ 融合（thing 优先）

分数阈值（panoptic）:
  ADE20K 0.5 / Cityscapes 0.8 / COCO 0.8（与 Mask2Former 一致）
```

---

## 6. 与 Mask2Former / SegFormer / SAM 对比

| 维度 | Mask2Former | OneFormer | SegFormer | SAM |
|------|-------------|-----------|-----------|-----|
| 范式 | 闭集 mask 分类 | 闭集 + **任务条件** | 逐像素 | 提示式 |
| 三任务权重 | 3 套 | **1 套** | 通常仅语义 | 非三任务 |
| 训练标注 | 三套 GT | **panoptic 即可** | 语义 GT | SA-1B |
| 文本 | 无 | **训练用** task/类描述 | 无 | 开放 prompt |
| 推理开销 | 中 | 中（无 text enc） | 低 | 高 encoder |

```
OneFormer = Mask2Former 骨干 + **任务条件训练 recipe**
  不是全新分割范式，而是 **统一训练与部署** 的解决方案
```

---

## 7. 消融结论（论文 §4.3 要点）

| 去掉 | 影响 |
|------|------|
| Task Token | 联合训练 **大幅掉点**，接近 naive joint |
| Query–Text 对比 | 三任务互扰，类混淆增 |
| Task 条件 query 初始化（改随机） | 多任务学不动 |
| 仅 panoptic 派生 GT（vs 三份 GT 文件） | 性能 **相当**，数据管线更简单 |

```
联合训练 Mask2Former（无 task token）:
  显著低于分任务训练 → 证明 **条件化** 必要
```

---

## 8. 精读备忘：易混淆点

### 8.1 「Universal」= 三任务一个模型，不是开放词汇

```
OneFormer 仍是 **闭集 K 类**（ADE20K 150 等）
  开放词汇分割见 SAM 3 / 开放词汇扩展工作
见 [SAM.md](./SAM.md)
```

### 8.2 Text Mapper 推理时不加载

```
训练:  Q_text + 对比损失
推理:  仅 **Q_task** 条件化 → 与 Mask2Former 速度同级
```

### 8.3 Task 文本是固定模板，不是用户自然语言

```
"the task is semantic"  为 **离散三选一**
  非 CLIP 式自由描述（区别于 LMSeg 等多数据集工作）
```

### 8.4 与 Panoptic FPN「双分支」不同

```
Panoptic FPN:  架构上 **两条分支** 算语义+实例
OneFormer:     **一条 mask 分类流**，任务由 token 切换
```

### 8.5 Queries 数量

```
常用 N=150~250；Mask2Former 论文称 250 queries 在部分设置 **退化**
  OneFormer 采用类似量级，需按数据集调
```

### 8.6 Backbone 升级收益

```
Swin-L → ConvNeXt-L / DiNAT-L
  联合模型仍涨点 → 与 **训练 recipe** 正交
```

---

## 9. 后续工作

| 方向 | 代表 |
|------|------|
| 更强 backbone | 与 Mask2Former 同步演进 |
| 视频 | Mask2Former-VIS、OneFormer 未原生视频 |
| 开放词汇 | Open-Vocabulary Mask2Former、EVA |
| 任务统一+提示 | SAM 3 PCS；与 OneFormer 闭集三任务互补 |
| 检测 | Mask DINO（检测+分割统一） |

---

## 10. 语义分割脉络（OneFormer 位置）

```
MaskFormer (2021)   mask 分类
Mask2Former (2022)  masked attn + 强 pixel decoder；**分任务训**  → Mask2Former.md
OneFormer (2023)    **+ Task Token + 对比 query，一次训练**        ← 本文
SAM / SAM2 / SAM3   提示式基础模型                               → SAM.md
```

---

## 11. 参考资料

- 原论文：[OneFormer](https://arxiv.org/abs/2211.06220)（CVPR 2023）
- 代码：[SHI-Labs/OneFormer](https://github.com/SHI-Labs/OneFormer)
- 前置：[Mask2Former.md](./Mask2Former.md)、[DETR.md](../2D Detection/DETR/DETR.md)
- 对比：[SegFormer.md](./SegFormer.md)、[SAM.md](./SAM.md)
- 数据：ADE20K、Cityscapes、COCO Panoptic

---

*文档版本：初稿 | 对应论文 CVPR 2023 OneFormer*
