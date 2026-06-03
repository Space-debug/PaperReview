# SAM 系列（Segment Anything）

> 本文档用于整理 Meta **SAM → SAM 2 → SAM 3 / SAM 3.1** 系列精读笔记。  
> 重点：**可提示（Promptable）分割基础模型**、**图像编码器 + 提示编码器 + 掩码解码器**、**视频记忆传播**、**开放词汇概念分割（PCS）**，以及与 [Mask2Former.md](./Mask2Former.md)（闭集监督分割）的路线差异。

> 相关：[Mask2Former.md](./Mask2Former.md) · [OneFormer.md](./OneFormer.md)（闭集三任务单权重）· [SegFormer.md](./SegFormer.md) · [../backbone/ViT.md](../backbone/ViT.md)

---

## 0. 系列地图（先读此节）

| 版本 | 发表 | 核心能力 | 提示类型 |
|------|------|----------|----------|
| **SAM** | ICCV **2023** | 图像 **任意对象** 分割 | 点、框、粗 mask |
| **SAM 2** | **2024** | 图像 + **视频** 分割与跟踪 | 点、框、mask（跨帧传播） |
| **SAM 3** | **2025.11** | **概念（Concept）** 级检测/分割/跟踪 | **文本**、exemplar 图、点/框/mask |
| **SAM 3.1** | **2026.03** | SAM 3 加速版（视频多对象） | 同 SAM 3；**Object Multiplex** |

```
演进主线:
  SAM:    「分割一切」— 视觉提示，零样本泛化，SA-1B 训练
  SAM 2:  「在视频里跟住」— 记忆库 + 流式传播
  SAM 3:  「按概念找全图/全视频所有实例」— 开放词汇短语 + exemplar
  SAM 3.1: 视频多对象 **单次前向最多 16 目标**，吞吐↑（H100 上 16→32 FPS 量级）

与 Mask2Former:
  Mask2Former = 固定 K 类、全监督、COCO/ADE 指标
  SAM 系列    = **提示驱动**、开放域、可 **无微调** 迁移
```

---

## 1. SAM（Segment Anything Model, v1）

### 1.1 基本信息

| 项目 | 内容 |
|------|------|
| 论文 | Segment Anything |
| 作者/机构 | Alexander Kirillov, Eric Mintun, …, Ross Girshick（Meta FAIR） |
| 发表 | **ICCV 2023**（arXiv 2023.04） |
| 数据 | **SA-1B**：1100 万张图、**11 亿+** 自动标注 mask |
| 代码 | [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) |

### 1.2 核心思想（一句话）

**用大规模伪标签训练一个与任务无关的「分割基础模型」：重型 ViT 图像编码器只算一次，轻量提示编码器把点/框/mask 转成稀疏 token，掩码解码器用双向 Transformer 在图像嵌入上预测二值 mask，并对歧义输出多张候选 mask + IoU 质量分供选择。**

| 对比 | Mask2Former | **SAM v1** |
|------|-------------|------------|
| 类别 | 闭集 K 类 | **无固定类别表** |
| 训练目标 | COCO 等 GT | **SA-1B 通用对象** |
| 使用方式 | 整图前向 | **提示后** 才出 mask |
| 视频 | 需扩展 | v1 **仅图像** |

### 1.3 三组件架构

```
Input Image
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Image Encoder（ViT-H / L / B，MAE 预训练）                    │
│  输入 1024×1024 归一化 → 图像嵌入 E_img (64×64×256 示意)      │
│  **每张图只算一次**，可缓存                                   │
└──────────────────────────────────────────────────────────────┘
    │
    │  E_img（frozen 或微调）
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Prompt Encoder                                              │
│  · 前景点 / 背景点 → positional encoding + label embedding   │
│  · 框 → 角点编码                                              │
│  · 输入 mask → 下采样 conv 与 E_img 对齐                      │
│  输出: 稀疏 prompt tokens P                                   │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Mask Decoder（轻量，Two-Way Transformer ×2 层）               │
│  · 可学习 **mask tokens**（含 IoU token + 多条 mask token）   │
│  · 图像嵌入 + 位置编码 ↔ prompt 双向 cross-attention          │
│  · 上采样 head → **3 张候选 mask**（消歧义）+ IoU 预测分数    │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
选 IoU 最高 mask，或用户指定；可迭代用预测 mask 作新 prompt refine
```

### 1.4 提示（Prompt）设计

| 提示 | 编码方式 | 用途 |
|------|----------|------|
| **点** | 前景/背景标签 + 随机 Fourier 位置 | 交互式点击 |
| **框** | 左上、右下嵌入 | 框选物体 |
| **Mask** | 低分辨率 mask → conv 对齐特征图 | 迭代 refine |

```
训练时 **模拟交互**:
  随机采样点/框/mask 序列，模仿用户点击过程
  → 模型学会从稀疏提示恢复完整 mask
```

### 1.5 歧义与多 mask 输出

```
一点可对应:  子部分 / 整物 / 多物组合

解码器输出 **3 个 mask 候选** + 每个的 **IoU 预测**
  → 默认选 IoU 分数最高
  → 或返回全部供用户选（「歧义消解」）

损失:
  · 对每个候选算 focal + dice
  · **仅最小 loss 的 mask** 反传分割梯度（winner-take-all）
  · IoU head: MSE(预测 IoU, 真 IoU)
```

### 1.6 训练与 SA-1B 数据引擎

```
数据构建（无人工逐像素）:
  1. 图像 + 粗 mask 提议（AMG 等）
  2. 用 **SAM 自身** 迭代标注，人工仅做质量过滤
  → 规模化到 11 亿 mask

训练:
  · 在线随机 prompt（点/框/mask）
  · 主损失: focal loss + dice loss（与 Mask2Former mask 分支同族）
  · 短 schedule 微调 decoder，encoder 可 frozen
```

### 1.7 零样本能力与局限

```
在 23+ 个分割 benchmark 上 **零样本** 迁移:
  边缘、点提示、框提示等多 setting

强项:
  · **开放对象**、标注工具、数据引擎、机器人抓取等

局限（v1）:
  · **无文本提示**（「分割黄色的校车」做不到）
  · **无视频** 一致 id 跟踪
  · 细微结构、透明物体、极度拥挤仍难
  · ViT-H 重，需 **图像嵌入缓存** 才能交互流畅
```

---

## 2. SAM 2（图像 + 视频）

### 2.1 基本信息

| 项目 | 内容 |
|------|------|
| 论文 | SAM 2: Segment Anything in Images and Videos |
| 作者/机构 | Meta FAIR（Ronghang Hu, Nikhila Ravi, …） |
| 发表 | **2024**（arXiv 2408.00714） |
| 代码 | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) |

### 2.2 核心思想（一句话）

**在 SAM 三分量基础上加入「记忆」：视频每一帧仍用图像编码器，但掩码解码器可 attend 到 **过去帧的记忆特征**（maskmem）与 **对象指针（object pointer）**，使用户在任意帧点击一次即可 **全序列跟踪分割**；图像视作单帧视频，统一框架。**

### 2.3 相对 SAM v1 的增量

```
┌─────────────────────────────────────────────────────────────┐
│  Image Encoder（Hiera / 改进 backbone，更快）                  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  SAM Mask Decoder（继承 prompt + 双向 Transformer）            │
│  + **Memory Attention** 层                                   │
│      当前帧特征 attend 到:                                    │
│        · Memory Bank: 过去帧 mask 编码特征（多帧 slot）        │
│        · Object Pointer: 来自 decoder 的对象级向量             │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Memory Encoder                                              │
│  将当前预测 mask + 图像特征 **写入** 记忆，供后续帧读取         │
└─────────────────────────────────────────────────────────────┘
```

### 2.4 视频交互逻辑

```
用户流程:
  1. 在任意帧 t 用 **点/框** 指定对象
  2. SAM 2 分割该帧，并 **编码进 memory**
  3. 向前/向后传播: 后续帧 decoder 读 memory → 预测 mask
  4. 可在新帧 **追加修正点**（纠正漂移）

训练数据:
  SA-V 等大规模视频 mask 数据（论文构建视频分割数据集）

相对光流/传统 VOS:
  · 与 SAM 提示接口 **一致**
  · 支持 **多对象**（每对象独立 memory 槽，依配置）
```

### 2.5 SAM 2.1（checkpoint 迭代）

```
非全新架构，主要为 **更强权重** 与训练 recipe:
  · 小物体、遮挡、边界改进
  · 官方 Hugging Face `sam2.1` 系列
文档阅读时 **SAM 2 = 架构族**，2.1 = 推荐推理权重
```

---

## 3. SAM 3（Segment Anything with Concepts）

### 3.1 基本信息

| 项目 | 内容 |
|------|------|
| 论文 | SAM 3: Segment Anything with Concepts |
| 发表 | **2025.11**（arXiv 2511.16719） |
| 任务 | **PCS（Promptable Concept Segmentation）** |
| 代码 | [facebookresearch/sam3](https://github.com/facebookresearch/sam3) |
| 数据/评测 | **SA-Co** benchmark；训练 **4M 概念** 标签（含 hard negatives） |

### 3.2 核心思想（一句话）

**在 SAM 2 的点/框/mask 能力之上，增加 **开放词汇文本短语**（如 “yellow school bus”）与 **图像 exemplar** 作为「概念提示」，由共享骨干的 **图像检测器** 在单帧找出该概念 **所有实例**，再由 **记忆式视频跟踪器** 在时序上保持 id；用 **Presence Head** 解耦「有没有」与「在哪」。**

### 3.3 PCS 任务定义

```
输入提示（概念）:
  · 短 **名词短语** text
  · 或 **exemplar** 图像（目标外观）
  · 或 二者组合
  · 仍可叠加 SAM 2 式 **点/框/mask** 精修

输出:
  · 图中/视频中 **所有匹配该概念的实例** mask
  · 每个实例 **唯一 id**（视频跟踪）

与 SAM v1/v2 差异:
  v1/v2:  一次提示 → **一个** 对象（或 3 歧义候选）
  SAM 3:  概念提示 → ** exhaustively 所有同类实例**
```

### 3.4 架构要点

```
统一模型，两大分支 **共享 backbone**:

① Image-level Detector（图像）
   · 文本 / exemplar 编码为 concept embedding
   · 与 **感知编码器（Perception Encoder）** 特征融合
   · 预测:  boxes / masks + **presence**（该概念是否存在）

② Memory-based Video Tracker（视频）
   · 继承 SAM 2 记忆机制
   · 对检测到的多实例 **跨帧关联**

**Presence Head**（论文强调）:
  识别（concept 在不在）与 定位（实例在哪）解耦
  → 降低 false positive，提升开放词汇检测精度

性能（论文）:
  · 图像/视频 PCS 相对既有系统约 **2×** 提升
  · 保留 SAM 2 交互分割能力
  · SA-Co: 270K 唯一概念，约为旧 benchmark 50×；人机性能约 75–80%
```

### 3.5 数据引擎 SA-Co

```
4M 唯一概念标签，跨图像与视频
  含 **hard negatives**（形似但非目标）
训练 scalable pipeline → 支撑开放词汇泛化

评测 SA-Co / SA-Co-VID:
  社区基准 **Promptable Concept Segmentation**
```

---

## 4. SAM 3.1（Object Multiplex）

### 4.1 基本信息

| 项目 | 内容 |
|------|------|
| 发布 | **2026.03**（Meta blog + `RELEASE_SAM3p1.md`） |
| 性质 | SAM 3 的 **drop-in 加速** 权重与推理策略，非全新任务定义 |
| 关键术语 | **Object Multiplex**（对象复用/共享记忆） |

### 4.2 核心改进

```
问题:
  SAM 3 视频上对 **多对象** 逐对象传播 → 前向次数多，GPU 吞吐受限

Object Multiplex:
  · **共享记忆（shared-memory）** 方式 **联合** 跟踪多对象
  · 单次前向最多 **16 个对象**（官方表述）
  · 中等对象数场景: H100 上吞吐约 **16 → 32 FPS** 量级
  · 精度相对 SAM 3 **不牺牲**（官方 claim）

使用:
  Hugging Face `facebook/sam3.1` checkpoints
  与 SAM 3 提示接口兼容，便于生产 **实时** 视频编辑（Edits 等）
```

---

## 5. 训练、推理与工程实践（系列通用）

### 5.1 推理模式对比

| 模式 | 典型流程 |
|------|----------|
| **图像单次** | `set_image` → `predict(point/box)` |
| **图像自动** | AMG（Automatic Mask Generator）网格点扫全图 |
| **视频** | 首帧提示 → `propagate_in_video` |
| **SAM 3 概念** | `text="cat"` 或 exemplar → 返回 **多实例** masks + ids |

### 5.2 嵌入缓存（性能关键）

```
Image Encoder 最重:
  交互应用应先 `set_image()` 缓存 E_img
  后续多次 prompt 只跑 **Prompt Encoder + Mask Decoder**

视频:
  逐帧 encoder + memory 更新；SAM 3.1 降低多对象重复开销
```

### 5.3 损失函数（SAM v1 代表）

```
L_mask:  min over 3 candidates of (λ_f·Focal + λ_d·Dice)
L_iou:   MSE between predicted IoU and true mask IoU

SAM 2/3:  在 mask loss 上增加 **视频时序一致性**、**检测匹配**、**概念对比** 等（实现见官方 repo）
```

### 5.4 常见衍生模型（非 Meta 主线，了解即可）

| 模型 | 作用 |
|------|------|
| **MobileSAM / FastSAM** | 蒸馏或替代 backbone，**边缘部署** |
| **SAM-HQ** | 提升边界质量 head |
| **Grounded SAM** | 接 Grounding DINO 文本框 + SAM mask |
| **SAM 3D**（Meta 另发） | 3D 重建/人体 mesh，**非** 2D SAM 主线 |

---

## 6. 与仓库内其它分割文档对比

| 维度 | FCN / DeepLab / SegFormer | Mask2Former | **SAM 系列** |
|------|---------------------------|-------------|--------------|
| 范式 | 每像素分类 | 闭集 mask 分类 | **提示 → mask** |
| 监督 | 全像素 GT | 实例/全景 GT | **SA-1B / SA-Co 伪标签** |
| 开放词汇 | 固定 K 类 | 固定 K 类 | **SAM 3 文本/概念** |
| 视频 | 逐帧独立 | 需扩展 | **SAM 2+ 原生** |
| 交互 | 无 | 无 | **核心能力** |
| 指标 | mIoU / PQ | COCO SOTA | 零样本 transfer、PCS、人机对比 |

```
Mask2Former 点积 mask 与 SAM 解码器 **结构灵感相近**（query/token ↔ 图像特征）
  但 SAM **不预测固定 K 类**，且 encoder 为 **提示条件** 计算
见 [Mask2Former.md](./Mask2Former.md) §9.5
```

---

## 7. 精读备忘：易混淆点

### 7.1 SAM ≠ 语义分割模型

```
不输出 「每像素必属 150 类之一」
  输出 **由提示定义的 foreground mask**
语义分割需:  对每类给提示 / 接 CLIP+文本（SAM 3）/ 外接分类器
```

### 7.2 三候选 mask 仅 v1/v2 交互歧义

```
SAM 3 **概念模式** 返回 **多实例**，不是 3 个歧义候选
  勿与 v1 的 3-mask 机制混谈
```

### 7.3 SAM 2 图像 = 单帧视频

```
同一套权重与 API；图像可 `frames=1`
```

### 7.4 SAM 3 的 text 是 **短语概念**，非完整句子推理

```
有效:  "yellow school bus", "traffic sign"
不是:  复杂问答或长描述推理（与 VLM 分割不同）
```

### 7.5 ViT-H vs MobileSAM

```
论文 SOTA 用 **ViT-H**；产品部署用蒸馏版
  精度—速度差一个数量级
```

### 7.6 SAM 3.1 ≠ SAM 3D

```
SAM 3.1:  2D 视频 **更快** 跟踪
SAM 3D:   三维重建/人体，另仓库 [sam-3d-body](https://github.com/facebookresearch/sam-3d-body) 等
```

---

## 8. 系列演进逻辑

```
分割范式三条线（本仓库）:

A. 监督密集预测:  FCN → DeepLab → PSPNet → HRNet → SegFormer
B. 闭集集合预测:  DETR mask → MaskFormer → Mask2Former → OneFormer
C. 基础模型提示:  SAM → SAM 2 → SAM 3 → SAM 3.1

SAM 系列把分割变成 **可提示基础设施**
  → 数据标注、视频编辑、机器人、Agent 工具链
```

---

## 9. 语义分割 / 基础模型脉络

```
2015–22:  CNN / Transformer 监督分割     → 2D Segment 各 .md
2021–22:  Mask 分类统一任务              → Mask2Former.md
2023:     SAM 图像基础模型               ┐
2024:     SAM 2 视频                      ├─ 本文
2025:     SAM 3 概念 + 文本                │
2026:     SAM 3.1 实时多对象              ┘
```

---

## 10. 参考资料

- SAM v1：[Segment Anything](https://arxiv.org/abs/2304.02643)（ICCV 2023）· [segment-anything](https://github.com/facebookresearch/segment-anything)
- SAM 2：[SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) · [sam2](https://github.com/facebookresearch/sam2)
- SAM 3：[SAM 3: Segment Anything with Concepts](https://arxiv.org/abs/2511.16719) · [sam3](https://github.com/facebookresearch/sam3)
- SAM 3.1：[Meta blog](https://ai.meta.com/blog/sam-3-1/) · `RELEASE_SAM3p1.md`
- 对比：[Mask2Former.md](./Mask2Former.md)、[ViT.md](../backbone/ViT.md)
- 生态：MobileSAM、SAM-HQ、Grounded-SAM

---

*文档版本：初稿 | 覆盖 SAM / SAM 2 / SAM 3 / SAM 3.1（截至 2026）*
