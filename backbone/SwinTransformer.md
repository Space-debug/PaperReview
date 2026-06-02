# Swin Transformer

> 本文档用于整理 Swin Transformer（Swin-T / Swin-S / Swin-B）论文精读笔记，并与同仓库 [ViT.md](./ViT.md) 做系统对比。  
> 重点：**Shifted Window Attention**、**层次化 Patch Merging**、**相对位置偏置**，以及**作为检测/分割 backbone 的多尺度特征输出**。  
> **想快速建立整体印象**：先读 **§1.5 ~ §1.9** 与 **§14 ViT 对比总表**，再按需深入后续章节。

---

## 1. 概览

### 1.1 基本信息

| 项目 | 内容 |
|------|------|
| 论文 | Swin Transformer: Hierarchical Vision Transformer using Shifted Windows |
| 作者/机构 | Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baoqiang Yao（Microsoft Research Asia） |
| 发表 | ICCV 2021（Best Paper） |
| 任务 | 通用视觉 backbone（分类、检测、分割统一架构） |
| 代码 | [microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer) |

### 1.2 核心思想（一句话）

**在 ViT 的全局 attention 之外，引入「窗口内局部 attention + 移位窗口跨块连接」降低复杂度，并通过 Patch Merging 构建 {H/4, H/8, H/16, H/32} 层次特征——像 CNN 一样有多尺度 pyramid，又像 Transformer 一样用 attention 建模。**

Swin 相对 ViT 要解决的两个痛点：

1. **ViT 是单尺度、全局 O(N²) attention** → 检测/分割需要 FPN 式多尺度，大图算不动；
2. **ViT 缺局部归纳偏置，小数据训不动** → Swin 的 window 限制 + 层次结构重新引入「局部→全局」偏置。

### 1.3 整体流水线

```
Input Image (batch, 3, H, W)     例如 H=W=224
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1: Patch Partition + Linear Embed（Stem）          │
│  P=4 切块 → H/4 × W/4 个 token，维度 C（如 96）           │
│  224 图 → 56×56 = 3136 tokens（仍在 Stage1 内用 window）  │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Swin Stage 1（depth×2，dim=C）                           │
│  交替: W-MSA（窗口内 attention）↔ SW-MSA（移位窗口）       │
│  分辨率保持 H/4 × W/4                                     │
└─────────────────────────────────────────────────────────┘
    │
    ▼ Patch Merge（2×2 邻域 concat + Linear，分辨率 ÷2，通道 ×2）
┌─────────────────────────────────────────────────────────┐
│  Swin Stage 2（dim=2C）   分辨率 H/8 × W/8                │
└─────────────────────────────────────────────────────────┘
    │
    ▼ Patch Merge
┌─────────────────────────────────────────────────────────┐
│  Swin Stage 3（dim=4C）   分辨率 H/16 × W/16              │
└─────────────────────────────────────────────────────────┘
    │
    ▼ Patch Merge
┌─────────────────────────────────────────────────────────┐
│  Swin Stage 4（dim=8C）   分辨率 H/32 × W/32              │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  分类: LayerNorm + Global Avg Pool + Linear → K 类       │
│  检测/分割: 取 Stage 2/3/4 输出 → FPN / UPerNet / Mask R-CNN │
└─────────────────────────────────────────────────────────┘
```

### 1.4 ImageNet-1k 精度参考（论文 Table 1）

| 模型 | 参数量 | FLOPs | Top-1 | 说明 |
|------|--------|-------|-------|------|
| Swin-T | 28M | 4.5G | **81.3** | Tiny，C=96 |
| Swin-S | 50M | 8.7G | **83.0** | Small |
| Swin-B | 88M | 15.4G | **83.5** | Base，C=128 |
| ViT-B/16 | 86M | 17.6G | 77.9* | *JFT 预训练+384 微调；仅 1k 训 ~74.5 |
| ResNet-50 | 25M | 4.1G | 76.5 | 同期 CNN 参照 |

> **读表要点**：Swin 在 **ImageNet-1k 从头训练** 即可超过同量级 ViT（无 JFT），且 FLOPs 相近时精度更高——这是「层次 + 窗口」归纳偏置的直接收益。

### 1.5 快速理解：三个支柱

```
                    ┌─────────────────────────────────┐
                    │  Swin = 窗口 Attention + 金字塔  │
                    └─────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
   ① Window Attention          ② Shifted Window              ③ Patch Merging
   「局部算、省 O(N²)」          「错位开窗、跨窗口通信」        「逐 stage 降采样」
          │                           │                           │
   7×7 窗口内 MSA               下一层平移 M/2               2×2 merge → 通道翻倍
   复杂度 ∝ N·M²                 打破窗口边界                  输出 4 级特征图
   M=7 固定                      偶数层 SW-MSA                 对齐 CNN-FPN
          │                           │                           │
          └───────────────────────────┴───────────────────────────┘
                                      │
                         结果：可当检测/分割 backbone，无需改 ViT 式 reshape  trick
```

| 支柱 | 一句话 | 作用 |
|------|--------|------|
| **Window Attention** | 只在 M×M 小窗口内做 self-attention | 计算从 O(N²) 降到 O(N·M²)，N 很大时可行 |
| **Shifted Window** | 奇数层常规窗、偶数层窗格平移 M/2 | 相邻层间接实现跨窗口信息传递 |
| **Patch Merging** | 每 stage 末 2×2 合并 patch | 层次化多尺度，对接 Mask R-CNN / UPerNet |

### 1.6 一张图的 Walkthrough（Swin-T，224 输入）

```
Step 0  输入 x: (3, 224, 224)

Step 1  Patch Embed (P=4)
        → (56, 56, 96)  即 3136 个 token，每 token 96 维

Step 2  Stage1（2 个 Swin Block，56×56）
        Block0: W-MSA   — 每个 7×7 窗口内独立 attention
        Block1: SW-MSA  — 窗口向右下平移 3 格，相邻窗互相「看见」

Step 3  Patch Merge → (28, 28, 192)

Step 4  Stage2 → 输出 P3 级特征 (B, 28, 28, 192)  ← 检测用

Step 5  Patch Merge → (14, 14, 384)
        Stage3 → P4  (B, 14, 14, 384)

Step 6  Patch Merge → (7, 7, 768)
        Stage4 → P5  (B, 7, 7, 768)

Step 7  分类头
        Norm → GAP(7×7) → Linear → 1000 类
```

### 1.7 常见困惑 FAQ

| 问题 | 简短回答 |
|------|----------|
| **Swin 还有 [CLS] token 吗？** | **没有**。分类用 **Global Average Pooling** 对最终 stage 所有 token 平均。 |
| **Window 大小多少？** | 默认 **M=7**；224 图 Stage1 为 56×56 grid，每窗 7×7 token = 49 个/窗。 |
| **移位多少？** | **⌊M/2⌋=3**；SW-MSA 层把窗口划分原点从 (0,0) 移到 (3,3)。 |
| **边界 patch 不足 7×7 怎么办？** | ** cyclic shift + attention mask**：padding 到窗大小，mask 掉非法位置。 |
| **位置信息怎么编码？** | **可学习相对位置偏置** B∈R^{(2M-1)×(2M+1)}，加在 attention logits 上；非 ViT 的绝对 PE。 |
| **为何适合检测？** | 天然输出 **多尺度特征 {P3,P4,P5}**，与 ResNet+FPN 接口一致。 |
| **Swin 比 ViT 一定更好吗？** | 分类/密集预测/中等数据上 often 是；极大模型+超大数据纯 ViT scaling 仍强。各有场景。 |

### 1.8 文档阅读路线

| 你的目标 | 建议阅读 |
|----------|----------|
| **5 分钟懂 Swin** | §1.5 ~ §1.9 + §14 ViT 对比 |
| **搞清 Window / Shift** | §3 W-MSA → §4 SW-MSA → §5 相对位置偏置 |
| **搞清多尺度怎么来** | §2 Patch Merge → §6 四 Stage 配置 |
| **接 Mask R-CNN / UPerNet** | §9 下游接口 |
| **和 ViT 选型** | §14 全文对比表 |
| **读代码** | §13 timm / 官方对照 |

---

## 2. Patch Embedding（Stem）

与 ViT 的 P=16 不同，Swin **Stem 用 P=4**，得到 **更高分辨率** 的第一 stage。

```
输入: (B, 3, H, W)
Patch Partition: 不重叠 4×4
Linear Embed: 4×4×3 = 48 → C（Swin-T 中 C=96）

输出: (B, H/4, W/4, C)  或 flatten 为 (B, (H/4·W/4), C)
```

| 对比项 | ViT-B/16 | Swin-T |
|--------|----------|--------|
| Stem patch | P=16 | P=4 |
| Stage1 分辨率 (224) | 14×14 | **56×56** |
| Stage1 token 数 | 196 | **3136** |
| 为何可行 | 全局 attention，N 小 | **Window**，N 大但每窗仅 49 token |

**设计意图**：检测需要 **高分辨率浅层特征**（小物体）；Swin 从 H/4 就开始堆 Transformer，而不是 ViT 那样一步到 H/16。

---

## 3. Window Multi-Head Self-Attention（W-MSA）

### 3.1 动机：ViT 全局 attention 的瓶颈

```
ViT:  N = 196 (224, P=16)  →  attention matrix 196×196
      若 N = 3136 (Swin Stage1) → 3136² ≈ 980 万元素/层/头 → 不可接受

Swin: 固定窗口 M=7 → 每窗 N_w = 49
      总复杂度 ≈ (N/49) × 49² = N × 49 = O(N·M²)
      M 固定时 ≈ 线性于 patch 数 N
```

### 3.2 窗口划分（W-MSA 层）

```
特征图 h: (H, W, C)   例如 56×56×96

划分不重叠 M×M 窗口（M=7）:
  56/7 = 8  →  每行 8 个窗，共 8×8 = 64 个窗口

每个窗口内:
  49 个 token 做标准 Multi-Head Self-Attention
  窗口之间 **无** attention 连接
```

```
56×56 grid 示意（每格=1 token，粗线=窗口界）:

┌───────┬───────┬ ... ───┐
│ 7×7   │ 7×7   │       │   Window A 内 token 只与 A 内交互
├───────┼───────┤       │
│ 7×7   │ 7×7   │       │
├───────┴───────┴ ... ───┤
│  ...                  │
└───────────────────────┘
```

### 3.3 单窗口内 MSA（与 ViT 相同）

```
窗内 token x ∈ R^{M²×C}

Q,K,V = x W_{Q,K,V}
Attn = softmax(QK^T / √d + B) V

其中 B 为相对位置偏置（§5），非 ViT 的全局绝对 pos_embed
```

---

## 4. Shifted Window MSA（SW-MSA）

### 4.1 问题：只用 W-MSA 会怎样

```
连续两层都是 W-MSA:
  同一对窗口边界两侧的 token，永远不在同一窗内
  → 跨窗口信息需要堆很多层才能间接传到
```

### 4.2 解法：交替 W-MSA / SW-MSA

```
Swin Block 典型 pattern（每个 Stage 内）:

  Block 2k:   W-MSA   （常规窗口）
  Block 2k+1: SW-MSA  （移位窗口）

Shift 量: shift_size = M/2 = 3（M=7）
```

**SW-MSA 步骤**：

```
1. cyclic shift: 特征图向右上 roll (-3, -3)
2. 在新坐标系下做 W-MSA（窗口边界已错开）
3. reverse cyclic shift 还原空间布局
4. 对 roll 产生的「假邻居」用 attention mask 置 -∞
```

```
示意（一维简化，M=4, shift=2）:

原序列:  [A B | C D | E F | G H]   窗1  窗2
roll后:  [E F | G H | A B | C D]   窗1' 窗2'
         → E 能与 G 同窗（原属不同窗）
reverse: 还原顺序
```

### 4.3 Attention Mask

SW-MSA 中，经 cyclic shift 后同一物理窗口内的 token 可能来自原图不同区域，需 **mask 表** 区分：

```
对每个 window，给 9 种相对区域编号（3×3 子块）
不同编号之间 attention score → -100（softmax 后≈0）
同编号内正常 attention
```

这是 Swin 实现里较「工程化」的部分；理解即可，读代码时对照 `window_partition` / `create_mask`。

---

## 5. 相对位置偏置（Relative Position Bias）

ViT 用 **绝对** 可学习 `pos_embed ∈ R^{(N+1)×D}`，加在 token 上。  
Swin 用 **相对** 偏置，加在 **attention logits** 上：

```
对每个 head:
  QK^T 的每一项 (i,j) 加上 B[Δx, Δy]

Δx, Δy = token i 与 j 在窗口内的相对坐标差
B 从可学习表 B_table ∈ R^{(2M-1)(2M+1)} 索引得到
（M=7 时表大小 13×13=169，按相对距离共享参数）
```

| 对比 | ViT | Swin |
|------|-----|------|
| 类型 | 绝对 PE，加在 input embedding | 相对 PE，加在 QK^T |
| 跨分辨率 | 需 **插值** pos_embed | 窗口大小固定，**相对关系不变**，迁移更自然 |
| 参数量 | O(N·D) | O(M²) 每 head，与图像大小无关 |

---

## 6. Patch Merging：构建层次金字塔

每个 Stage 末尾（最后一个 Block 后）：

```
输入: (H, W, C)

1. 按 2×2 邻域分组，concat 通道 → (H/2, W/2, 4C)
2. LayerNorm + Linear(4C → 2C)

输出: (H/2, W/2, 2C)
```

**四 Stage 尺度（224 输入，Swin-T C=96）**：

| Stage | 分辨率 | 通道 | 相对 CNN | 检测命名 |
|-------|--------|------|----------|----------|
| 1 | 56×56 | 96 | — | 通常不直接用于 FPN |
| 2 | 28×28 | 192 | ~C2/C3 | **P3** |
| 3 | 14×14 | 384 | ~C4 | **P4** |
| 4 | 7×7 | 768 | ~C5 | **P5** |

```
与 ResNet-50 对齐（概念）:

ResNet:  C2 1/4   C3 1/8   C4 1/16   C5 1/32
Swin:    —      1/8      1/16      1/32   （Stage1 更细，可选接 neck）
```

这正是 Swin 成为 **COCO 检测/ADE 分割 SOTA backbone** 的原因：无需像 ViT 那样只拿最后一层 reshape。

---

## 7. Swin Transformer Block 完整结构

```
输入 x: (B, H, W, C)

── Block l（W-MSA 或 SW-MSA）──
  x' = x + DropPath(LN(W-MSA/SW-MSA(x)))

── FFN ──
  x'' = x' + DropPath(LN(MLP(x')))

MLP: Linear(C → 4C) → GELU → Linear(4C → C)
```

| 组件 | 说明 |
|------|------|
| **Pre-LN** | 与 ViT 相同，Norm 在子层前 |
| **DropPath** | Stochastic Depth，训练时随机 drop 整条 residual 路径 |
| **无 [CLS]** | 所有 spatial token 对称处理 |

### 7.1 Swin-T 深度配置

```
Stage1: depth=2, C=96,   56×56,  [W-MSA, SW-MSA]
Stage2: depth=2, C=192,  28×28,  [W-MSA, SW-MSA]  → 输出给下游
Stage3: depth=6, C=384,  14×14,  ×6 blocks
Stage4: depth=2, C=768,   7×7,   ×2 blocks

总 depth = 2+2+6+2 = 12（与 ViT-B 层数相同，但分布在不同分辨率）
```

Swin-S/B 增大 C 或 depth（Swin-B: C=128, depth 2+2+18+2）。

---

## 8. 分类头与读出

```
Stage4 输出: (B, H/32, W/32, 8C)
    │
    ▼ LayerNorm
Global Average Pooling（对空间维）
    │
    ▼
Linear(8C → num_classes)
```

| 对比 | ViT | Swin |
|------|-----|------|
| 聚合方式 | **[CLS] token** | **GAP 所有 token** |
| 读出位置 | 任意层均可取，分类用 z[0] | 默认 **最后一 stage** |

---

## 9. 下游任务：检测与分割接口

### 9.1 COCO 检测（Mask R-CNN + FPN 风格）

```
Backbone: Swin-T
Neck:     简单 FPN 或 BiFPN（论文用 Mask R-CNN 框架 + 多尺度特征）

输入 Stage2/3/4 输出:
  P3: 1/8  192-d
  P4: 1/16 384-d
  P5: 1/32 768-d

接 RPN + RoI Head → 与 ResNet 训练流程几乎相同
```

论文结果（COCO test-dev）：Swin-T Mask R-CNN **46.0 AP**，同设定 ResNet-50 **40.9 AP**。

### 9.2 ADE20K 语义分割（UPerNet）

```
Backbone 多尺度特征 → UPerNet PSP + FPN 融合 → per-pixel 分类
Swin-T: 45.8 mIoU vs ResNet-50 42.1
```

### 9.3 与 ViT 接下游的差异

```
ViT 分类 backbone:
  常用 z_L[1:] reshape → (B, D, 14, 14)  **单尺度**
  检测需 ViTDet 等额外设计（中间层 + 简单 FPN）

Swin:
  **原生 3 尺度** (1/8, 1/16, 1/32)
  直接替换 ResNet in Mask R-CNN / Cascade R-CNN
```

---

## 10. 训练策略（ImageNet-1k）

### 10.1 超参概要

```
优化器: AdamW
base lr: 1e-3（Swin-T），按 batch 线性缩放
weight decay: 0.05
batch size: 1024
schedule: 300 epoch，cosine decay
warmup: 20 epoch
增强: RandAugment, Mixup(0.8), CutMix(1.0), Random Erasing
正则: DropPath rate 0.2（Swin-T）
```

### 10.2 与 ViT 训练差异

| 项目 | ViT（1k 从头） | Swin（1k 从头） |
|------|----------------|-----------------|
| 是否需要 JFT | 强烈依赖才 SOTA | **不需要** |
| 典型 batch | 4096（预训练） | 1024 |
| 位置编码 | 绝对 PE，改分辨率要插值 | 相对 bias，窗口固定 |
| 收敛 | 慢，易过拟合 | 相对稳，数据效率更高 |

---

## 11. 复杂度分析

设特征图有 h×w 个 token，窗口 M×M，C 为通道，L 为 stage 内 block 数。

```
W-MSA 单层（单 stage）:
  O(h·w·M²·C + h·w·C²)   窗内 attention + 线性投影

Patch Merge:
  O(h·w·C)

四 stage 总 token 数约 N_total = hw·(1 + 1/4 + 1/16 + 1/64) < 1.33·hw
→ 主要为 Stage1 的 56×56 贡献计算，但 window 使每层 attention 仅 O(49) 每 token
```

**ViT vs Swin（224，量级对比）**：

| | ViT-B/16 | Swin-T |
|---|----------|--------|
| 主要 attention 域 | 全局 N=196 | 窗内 49，共 64 窗/stage1 |
| 多尺度 | 无 | 4 stage |
| FLOPs | ~17.6G | ~4.5G |
| 1k Top-1 | ~74.5（无 JFT） | **81.3** |

---

## 12. Swin Block 信息流动（与 ViT 对照）

```
ViT 一层（全局）:
  每个 token ←── attention ──→ 全部 196 token

Swin 两层（一对 W + SW）:
  Layer0 W-MSA:   token 只与同窗 49 个交互
  Layer1 SW-MSA:  窗界错开 → 原相邻窗的 token 可同窗
  → 两层后感受野扩大，类似 CNN 两层 3×3
  → 堆深 + Patch Merge → 层次化大感受野
```

```
感受野增长（直觉）:

Stage1 若干 W/SW block  →  局部纹理、边缘
Stage2 merge 后         →  部件级
Stage3                  →  物体级
Stage4                  →  场景级语义（+ GAP 分类）
```

---

## 13. 代码对照

### 13.1 官方 PyTorch

| 模块 | 路径/符号 |
|------|-----------|
| Window partition | `swin_transformer.py` → `window_partition` |
| W-MSA / SW-MSA | `WindowAttention` |
| Patch Merging | `PatchMerging` |
| 相对位置偏置 | `relative_position_bias_table` |

### 13.2 timm

```python
import timm
model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
# features only:
model.forward_features(x)  # 多 stage 特征或最终 GAP 前
```

| timm 名 | 含义 |
|---------|------|
| `swin_tiny_patch4_window7_224` | Swin-T, P=4, M=7 |
| `swin_base_patch4_window7_224` | Swin-B |

---

## 14. Swin vs ViT：系统对比（核心章节）

### 14.1 总览对照表

| 维度 | **ViT** | **Swin** |
|------|---------|----------|
| **发表** | ICLR 2021 | ICCV 2021 |
| **Attention 范围** | **全局**（所有 patch） | **窗口内** + 移位连接 |
| **复杂度** | O(N²) | O(N·M²)，M 固定 ≈ O(N) |
| **结构** | **单尺度** flat | **层次金字塔** 4 stage |
| **Stem patch** | 通常 P=16 | P=4 |
| **224 时 Stage1 token** | 14×14=196 | 56×56=3136 |
| **位置编码** | 绝对可学习 PE | **相对位置偏置** |
| **分类 token** | **[CLS]** | **GAP**（无 CLS） |
| **归纳偏置** | 弱 | **较强**（局部窗 + 层次） |
| **ImageNet-1k 从头训** | 弱（~74%） | **强（~81%）** |
| **依赖 JFT/21k** | 强依赖才 SOTA | 不必须 |
| **检测/分割** | 需改造（ViTDet 等） | **原生多尺度**，易接 FPN |
| **分辨率迁移** | pos_embed **插值** | 相对 bias **更稳** |
| **代表后续** | MAE, DeiT, CLIP | SwinV2, SimMIM |

### 14.2 架构哲学差异

```
ViT:
  「图像 = 句子，尽量照搬 NLP Transformer」
  用 scale（大数据）换 inductive bias

Swin:
  「视觉仍需 CNN 的两个优点：局部性 + 金字塔」
  用 window / shift / merge 把 bias 写回 Transformer
```

### 14.3 Attention 模式对比图

```
ViT 全局 Attention（N=9 示意，全连接）:

    t1 t2 t3
    t4 t5 t6   每个 token 与所有 token 相连
    t7 t8 t9

Swin W-MSA（M=3，3×3 窗，无 shift）:

    ┌─────┬─────┐
    │全连接│全连接│   窗内全连接，窗间断开
    ├─────┼─────┤
    │全连接│全连接│
    └─────┴─────┘

Swin SW-MSA（下一层，shift 后）:

    窗界移动 → 原先分隔的 token 落入同一窗
    → 跨窗信息一层就可达
```

### 14.4 何时选 ViT，何时选 Swin

| 场景 | 更推荐 | 原因 |
|------|--------|------|
| 有 JFT/21k/MAE 大预训练 | **ViT** 或 **ViT+MAE** | 大模型 scaling、生态成熟 |
| 仅 ImageNet-1k | **Swin** | 数据效率、默认更强 |
| COCO 检测 / ADE 分割 | **Swin** | 多尺度 backbone 开箱即用 |
| 极大分辨率图像 | **Swin**（窗口）或 **局部 ViT 变体** | ViT 全局 N² 爆炸 |
| 多模态 CLIP 式预训练 | **ViT** | CLIP/SigLIP 默认 ViT |
| 自监督 MAE 预训练再微调 | **ViT**（MAE 原论文） | MAE 掩码 patch 与 ViT 序列天然契合 |
| 与 DETR 同框架实验 | 两者皆可 | DETR 原论文 ResNet；换 Swin 作 backbone 常见 |

### 14.5 与 DeiT、MAE 的关系（训练配方，非 Swin 本体）

| 方法 | 解决 ViT 的什么问题 | 与 Swin 关系 |
|------|---------------------|--------------|
| **DeiT 蒸馏** | 1k 数据训 ViT | Swin **不靠蒸馏**也能 1k SOTA；DeiT 是 ViT 线 |
| **MAE 预训练** | 无标签预训练 ViT | 主要针对 ViT；Swin 有 **SimMIM** 等对标工作 |

详见下一节 §15 概念速查。

---

## 15. 附录：DeiT 蒸馏与 MAE 预训练（ViT 训练扩展）

> 你在 ViT 笔记里看到的两个名词，本质是 **「让 ViT 更好训」** 的两条路线；Swin 通过结构改进了数据效率，但理解它们有助于选型。

### 15.1 DeiT：Data-efficient Image Transformers（ICML 2021）

**问题**：ViT 在 ImageNet-1k **从头监督训练** 不如 ResNet，且需要 JFT 级数据。

**做法**：在标准 ViT 上增加 **知识蒸馏（Knowledge Distillation）**：

```
Teacher: 强 CNN（如 RegNetY-16GF），在 ImageNet-1k 上训好，冻结
Student: ViT（如 DeiT-S/B）

Student 输入同一图像，输出两路:
  ① [CLS] token → 硬标签 CE loss（与 GT 类别）
  ② [DIST] token → 与 teacher 的 soft logits 做 KL 蒸馏 loss

总损失: L = L_CE + λ · L_distill
```

**Distillation token**：与 [CLS] 并列的 **第二个可学习 token**，专门用于从 teacher 学「暗知识」（类间相似度），避免干扰 [CLS] 学分类。

```
序列: [CLS, DIST, patch_1, ..., patch_196] + pos_embed
       ↓      ↓
     CE loss  KL( student_DIST , teacher_soft )
```

**结果**：DeiT-S 在 **仅 ImageNet-1k、无 JFT** 下达到 79.8% Top-1，证明 ViT **可以** 在中等数据上训起来，但需要 **蒸馏 + 强增强** 等配方。

**与 Swin 对比**：Swin **没有** distillation token，靠 **window + 金字塔结构** 直接在 1k 上训到 81%+。

---

### 15.2 MAE：Masked Autoencoder（CVPR 2022）

**问题**：ViT 监督预训练仍要大量 **标注** 数据；能否像 BERT 一样 **自监督** 预训练？

**做法**：**掩码图像建模（MIM）**

```
1. Patch 化（同 ViT，P=16）
2. 随机 mask 掉 ~75% 的 patch（只保留 ~25% 可见）
3. Encoder: 标准 ViT，**只输入可见 patch**（+ 可选 [CLS]）→ 省算力
4. Decoder: 轻量 Transformer，输入「可见 token + mask token 占位符」
5. 目标: 重建 **被 mask patch 的像素**（归一化像素 MSE）

预训练完成后:
  丢弃 Decoder，Encoder 权重初始化 ViT → ImageNet 微调分类
```

```
可见 patch (25%)  →  ViT Encoder  →  latent
                                      ↓
mask token + 位置编码  →  轻量 Decoder  →  重建 75% 被遮 patch 的像素
```

**结果**：MAE 在 ImageNet-1k **无标签预训练** 后微调，ViT-H 可达 **87.8%** Top-1；**标注需求** 远低于 JFT 监督预训练。

**与 ViT / Swin 关系**：

| | ViT 监督 | DeiT | MAE | Swin |
|---|----------|------|-----|------|
| 预训练信号 | 类别标签 | 标签 + teacher soft | **像素重建（无标签）** | 多为 1k 监督 |
| 核心 trick | JFT scale | 蒸馏 token | **高比例 mask + 非对称 enc/dec** | window + merge |
| 代表下游 | CLIP, SAM | timm `deit_*` | 微调分类/检测 | SimMIM（Swin 版 MAE） |

**一句话**：

- **DeiT** = 有 teacher 带着 ViT **监督学习**（省 JFT，不省标签）  
- **MAE** = 先 **遮住大部分 patch 做填空题** 预训练 Encoder，再微调（省标签）  
- **Swin** = 改 **网络结构** 让 Transformer 更像 CNN，1k 监督就能训强

---

## 16. 小结速查卡

```
┌─────────────────────────────────────────────────────────────────┐
│ Swin-T @ 224 一句话                                              │
│ P=4 embed → 4 stage (W-MSA↔SW-MSA + PatchMerge) → GAP → 1k 类   │
├─────────────────────────────────────────────────────────────────┤
│ vs ViT   窗口 attention | 金字塔 | 相对位置 | GAP | 1k 友好      │
│ 检测用法  直接输出 P3/P4/P5 → Mask R-CNN / UPerNet               │
│ 必记交替  奇数层 W-MSA，偶数层 SW-MSA（shift M/2）               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 参考文献

- Liu et al., *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*, ICCV 2021. [arXiv:2103.14030](https://arxiv.org/abs/2103.14030)
- Dosovitskiy et al., *An Image is Worth 16x16 Words (ViT)*, ICLR 2021. → [ViT.md](./ViT.md)
- Touvron et al., *DeiT*, ICML 2021. [arXiv:2012.12877](https://arxiv.org/abs/2012.12877)
- He et al., *Masked Autoencoders Are Scalable Vision Learners (MAE)*, CVPR 2022. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
- Cao et al., *Swin Transformer V2*, CVPR 2022.（扩展：cosine attention、预训练稳定）
