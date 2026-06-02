# Deformable DETR

> 本文档用于整理 Deformable DETR（Deformable Transformers for End-to-End Object Detection）论文精读笔记。  
> 重点：**多尺度可变形注意力（MSDeformAttn）**、**参考点（Reference Point）机制**、**多尺度特征融合**，以及相对 DETR 的 **收敛加速与计算稀疏化** 原理。  
> **想快速建立整体印象**：先读 **§1.5（与 DETR 差异总览）** 与 **§3（MSDeformAttn）**；读实现时看 **§11（CUDA 算子）**。

> 前置阅读：[DETR.md](./DETR.md) —— 理解 Object Query、匈牙利匹配、Encoder-Decoder 检测范式后再读本篇。

---

## 1. 概览

### 1.1 基本信息

| 项目 | 内容 |
|------|------|
| 论文 | Deformable DETR: Deformable Transformers for End-to-End Object Detection |
| 作者/机构 | Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, Jifeng Dai（SenseTime 等） |
| 发表 | ICLR 2021（arXiv 2020.10） |
| 任务 | 端到端目标检测（DETR 的高效可收敛改进版） |
| 代码 | [fundamentalvision/Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR) |

### 1.2 核心思想（一句话）

**保留 DETR 的「集合预测 + 匈牙利匹配」框架，把 Transformer 里 O(hw²) 的全局注意力替换为「以参考点为中心、在多尺度特征上只采样 K 个稀疏点」的可变形注意力；并引入 FPN 式多尺度特征，使 query 能高效看到小物体。**

Deformable DETR 相对 DETR 的三项关键改动：

| 改动 | 解决什么问题 |
|------|--------------|
| **Multi-Scale Deformable Attention** | 注意力不再对全部 hw 个 token 做 dot-product，复杂度从 O(hw²) 降到 O(hw·L·K) |
| **多尺度特征（L=4 层）** | 单尺度 stride-32 看不清小物体 → 同时用 stride 8/16/32/64 特征 |
| **参考点 + 迭代框 refine** | query 一开始「不知看哪」→ 用 reference point 给采样中心，逐层 refine 框 |

### 1.3 整体流水线

```
Input Image (B, 3, H, W)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1: ResNet Backbone + 多尺度投影                   │
│  输出 L=4 层特征: stride 8 / 16 / 32 / 64                │
│  每层 1×1 Conv → d=256，加 pos enc + level_embed         │
│  flatten 后拼接 → 总长 S = Σ_l (H_l × W_l)               │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2: Deformable Encoder（6 层）                     │
│  每层: MSDeformAttn(self) + FFN                          │
│  每个 token 以自身网格点为参考点，在 L 层特征上各采 K=4 点  │
│  输出 memory: (B, S, 256)                                │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 3: Deformable Decoder（6 层）                     │
│  N=300 Object Queries + reference points (cx,cy)         │
│  每层: Self-Attn(q↔q) → MSDeformAttn(q→memory) → FFN     │
│  可选: 逐层迭代 refine bbox，更新 reference points        │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 4: 预测头（与 DETR 相同逻辑）                      │
│  分类 Linear → K+1（含 ∅）；回归 MLP → (cx,cy,w,h)       │
│  匈牙利匹配 + Set Loss                                   │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Output: 300 组 (class, box) → 阈值过滤，无需 NMS
```

### 1.4 COCO 精度参考（论文主结果，R50，50 epoch）

| 配置 | AP | AP_S | AP_M | AP_L | 说明 |
|------|-----|------|------|------|------|
| Deformable DETR（单尺度） | 39.4 | 20.6 | 43.0 | 55.5 | 仅 res5，验证 MSDeformAttn 本身有效 |
| Deformable DETR（多尺度） | **44.5** | **27.1** | 47.6 | 59.6 | 默认配置 |
| + iterative bbox refine | 46.2 | 28.3 | 49.2 | 61.5 | 逐层 refine 框 |
| ++ two-stage | **46.9** | **29.6** | 50.1 | 61.6 | Encoder 产出 top-K query 初始化 |

> 训练 **50 epoch** 即可超过 DETR **500 epoch** 的 AP（42.0），且 AP_S 从 ~20 提升到 ~27+。

### 1.5 与 DETR 差异总览（先读本节）

Deformable DETR **保留** DETR 的集合预测框架（Object Query、匈牙利匹配、∅ 类、无 NMS），**替换** 的是「特征怎么读」和「特征从哪来」。下表按模块对照：

| 模块 | DETR | Deformable DETR | 差异本质 |
|------|------|-----------------|----------|
| **Backbone 输出** | 仅 conv5，单尺度 stride 32 | C3~C5 + 额外 C6，**4 尺度** stride 8/16/32/64 | 小物体在高分辨率层保留细节 |
| **Encoder 输入** | hw≈825 个 token | S≈17700 个 token（四层 concat） | 序列更长，但 attn 稀疏所以可算 |
| **Encoder Attention** | 标准 **全局** self-attn，O(hw²) | **MSDeformAttn**，每 token 采 L×K 点，O(S·L·K) | 从「全图两两交互」→「稀疏跨尺度采样」 |
| **Decoder Cross-Attn** | 全局 cross-attn，Q·K 对 **全部 hw** key | **MSDeformAttn**，只对 **L×K=16** 采样点加权 | 定位从「825 选谁」→「16 个点读 value」 |
| **Decoder Self-Attn** | 标准 MultiheadAttention | **相同**（未改） | query 间仍互相协调去重 |
| **空间先验** | query **无** 坐标，靠 cross-attn 自己找 | 每个 query 有 **reference point (cx,cy)** | 采样有明确中心，收敛快 |
| **框的迭代** | 仅最后一层输出框 | 可选 **逐层 refine ref**，6 层 cascade | 采样中心随层收紧 |
| **Query 数量** | N=100 | N=**300** | 配合 two-stage top-K |
| **训练 epoch** | 300~500 | **50** | 稀疏 attn + ref + 多尺度共同作用 |
| **匹配 / Loss** | 匈牙利 + CE + L1 + GIoU | **相同**（λ 略调） | 问题定义未变 |

**一句话**：DETR 用 **全局 Attention** 做定位；Deformable DETR 用 **参考点 + 稀疏可变形采样 + 多尺度 value**，在 **同一套集合预测训练** 下把「看哪里」变得便宜且可学。

```
DETR 定位:
  query ──cross-attn──► 对 hw 个 grid 算 softmax ──► 加权 memory

Deformable 定位:
  query + ref(cx,cy) ──MSDeformAttn──► 在 4 尺度各采 4 点 ──► 双线性读 value ──► 16 点加权
```

---

## 2. 动机：DETR 的三条瓶颈与 Deformable DETR 的对策

### 2.1 瓶颈 ① 收敛极慢

**原因（机制层面）**：

```
DETR Decoder cross-attention:
  每个 query 对全部 hw≈800+ 个 spatial token 做 softmax 加权
  初始 query 随机 → attention 均匀分散 → 梯度稀、信号弱
  query 需要「从零学会该看哪里」→ specialization 出现很晚（~100 epoch 后）
```

**Deformable DETR 对策**：

- 每个 query 只采样 **L×K = 4×4 = 16** 个位置（每 head 每 level 4 点），attention 权重只在 16 个采样点上 softmax
- 引入 **reference point**（初始由 `Linear(query_embed)` 预测 (cx,cy)）→ 采样 **围绕已知大致位置**，搜索空间大幅缩小
- **迭代 bbox refine**：每层 Decoder 更新 reference point → 下一层在更准的位置附近采样

### 2.2 瓶颈 ② 计算与内存贵

```
DETR Encoder self-attention:  O(hw²)     825² ≈ 680K 每 head
DETR Decoder cross-attention: O(N·hw)    100×825 每 head

Deformable Encoder self-attn: O(S·L·K)   每个 token 只采 L·K 点
Deformable Decoder cross-attn: O(N·L·K)  每个 query 只采 L·K 点

当 S≈5000（多尺度拼接后）仍远小于 S²；Decoder 侧 N·L·K ≈ 300×16 = 4800 << N·S
```

### 2.3 瓶颈 ③ 小物体弱

DETR 仅用 stride-32 的 conv5 → 小物体在特征图上仅占 1~2 格，信息损失大。

**对策**：拼接 **stride 8/16/32/64** 四层特征；MSDeformAttn 的 K 个采样点可 **跨层** 分配——小物体 query 可在高分辨率层（stride 8）上采更多有效点。

---

## 3. 核心模块：Multi-Scale Deformable Attention

这是 Deformable DETR 相对 DETR **改动最大** 的部分；Encoder self-attn 和 Decoder cross-attn **全部** 换成 MSDeformAttn（Decoder 的 query↔query 仍用标准 MultiheadAttention，与 DETR 相同）。

### 3.0 同一 query 找猫：DETR vs Deformable 并行对照

假设 query#7 需要定位「猫」，图像 800×1066，特征已进 Transformer。

#### DETR 的 Cross-Attention 在做什么

```
输入:
  q_7 = query#7 + query_pos          (256-d)
  memory: 825 个 token，每个有 k_i, v_i

计算（每个 head 独立，略）:
  α_i = softmax( q_7 · k_i / √32 )     i = 1..825
  h_7 = Σ_{i=1}^{825}  α_i · v_i

特点:
  ① key 来自 **固定 grid** 上的 825 个位置，不能「插值到格子之间」
  ② softmax 在 **825 维** 上归一化 → 初期近似均匀 → 梯度稀释
  ③ q_7 **没有** 显式 (x,y)，必须靠 dot-product 自己「发现」猫在哪
  ④ 只有 **stride-32** 一层 memory，猫若占 1~2 格，信息极少
```

#### Deformable DETR 的 MSDeformAttn 在做什么

```
输入:
  z_7 = tgt_7 + query_pos            (256-d)
  ref_7 = (cx,cy) ≈ (0.4, 0.35)    由 Linear(query_pos) 初始化或上层 refine
  memory: S≈17700 token，分属 L=4 个尺度

Step A  预测「去哪采」（由 z_7 决定，非固定 grid）
  Δp_{lqk} = Linear_offset(z_7)     8 heads × 4 levels × 4 points × 2
  A_{lqk}  = softmax( Linear_weight(z_7) )   在 L×K=16 个点归一化

Step B  算采样坐标（归一化到各层特征图）
  p_sample = ref_7 + Δp_{lqk} / (W_l, H_l)

Step C  读 value（可 bilinear，**不限于整数格点**）
  v_{lqk} = bilinear_sample( V_l, p_sample )

Step D  加权聚合
  h_7 = Σ_l Σ_k  A_{lqk} · W_v · v_{lqk}

特点:
  ① 只看 **16 个点**，不是 825 个
  ② 采样位置 **连续**，可落在格子之间
  ③ ref_7 提供 **空间锚点**，Δp 只做局部微调
  ④ 可在 **stride-8** 层采猫身边 4 点，小物体信息足
```

#### 核心差异小结

| 维度 | DETR Cross-Attn | MSDeformAttn |
|------|-----------------|--------------|
| 交互对象 | 825 个 **离散 token** | 16 个 **连续坐标** 上的 value |
| 权重来源 | Q·K 相似度 | 独立 Linear 预测 A_{lqk}（与 offset 同源） |
| 空间先验 | 无 | reference point |
| 多尺度 | 无 | 4 层同时采 |
| 计算 | O(N·hw) | O(N·L·K) |
| 梯度聚焦 | 分散到 hw 维 softmax | 集中在 16 维 softmax |

### 3.1 与 DETR 全局 Attention 的公式对比

```
DETR 全局 Attention:
  output(q) = Σ_{i=1}^{hw}  softmax(q·k_i) · v_i
  对所有 spatial 位置 i 求和

MSDeformAttn:
  output(q) = Σ_{l=1}^{L} Σ_{k=1}^{K}  A_{lqk} ·  bilinear(V_l, p_q + Δp_{lqk})
  只在 L 个尺度 × 每尺度 K 个采样点上求和
  Δp 和 A 由 query q 动态预测
```

**类比**：DETR 像「全图搜索」；Deformable DETR 像「拿着 GPS 坐标（reference point），在附近 L 张不同比例地图上各标 K 个钉子，只看钉子处的特征」。

### 3.2 数学定义

给定 query 特征 `z_q`，参考点 `p_q`，多尺度 value 特征 `{V_l}`：

```
DeformAttn(z_q, p_q, {V_l}) = Σ_{l=1}^{L} Σ_{k=1}^{K}  A_{lqk} · W_v · V_l(p_q + Δp_{lqk})

其中:
  L     = 特征层数（默认 4）
  K     = 每层采样点数（默认 4，enc/dec 可独立设 enc_n_points / dec_n_points）
  p_q   = 参考点，归一化坐标 ∈ [0,1]²（或 4 维 box）
  Δp_{lqk} = Linear(z_q) 预测的偏移（2 维）
  A_{lqk}  = softmax over (l,k) 的注意力权重，Σ A = 1
  V_l(·)   = 在第 l 层特征图上双线性插值采样
  W_v      = value_proj（Linear d→d）
```

多头版本：每个 head 独立一组 `{Δp_{lqk}, A_{lqk}}`，维度 `d_model / n_heads = 32`。

### 3.3 参考点与采样位置（代码级逻辑）

MSDeformAttn 的 `forward` 中（`ms_deform_attn.py`）：

**① reference 为 2 维 (cx, cy)**（Encoder 和 Decoder 常用）：

```
offset_normalizer = (W_l, H_l)  各层特征图宽高

sampling_loc = ref_point + sampling_offsets / offset_normalizer

ref_point:       (B, Len_q, L, 2)   归一化到 [0,1]，已乘 valid_ratio
sampling_offsets: (B, Len_q, n_heads, L, K, 2)  由 Linear(query) 预测
```

**② reference 为 4 维 (cx, cy, w, h)**（迭代 refine 后）：

```
sampling_loc = ref_box[:2] + (offsets / K) * ref_box[2:] * 0.5

偏移量相对 box 宽高缩放 → 采样点在预测框内部/附近
```

**③ 双线性采样 + 加权**：

```
对每个 (l, k):  grid_sample(V_l, sampling_loc[l,k])
output = Σ_{l,k}  A_{lqk} · sampled_value_{lqk}
output = output_proj(output)
```

### 3.4 参数量与初始化

```
MSDeformAttn 内部 Linear:
  sampling_offsets:  d_model → n_heads × L × K × 2
  attention_weights: d_model → n_heads × L × K
  value_proj / output_proj:  d_model → d_model

默认 d=256, heads=8, L=4, K=4:
  offsets: 256 → 8×4×4×2 = 1024
  weights: 256 → 128

偏移初始化（重要）:
  weight = 0
  bias 按不同 head 角度 × 不同 k 半径 预设成「星形」分布
  → 训练初期就在 reference 周围有一圈探索点，而非全挤在原点
```

### 3.5 逐点 Walkthrough（Decoder cross-attn 一例）

设 query#7 找「猫」，reference point 经 init 后 `(cx,cy)≈(0.4, 0.35)`：

```
Step 1  预测偏移与权重
        z_7 → Linear → 8 heads × 4 levels × 4 points 的 (Δx,Δy)
        z_7 → Linear → softmax → A_{lqk}

Step 2  计算 4×4=16 个采样坐标（每层 4 点）
        Level 0 (stride 8,  100×133 特征图): 4 点围绕 (0.4, 0.35)
        Level 1 (stride 16, 50×67):          4 点
        Level 2 (stride 32, 25×33):          4 点
        Level 3 (stride 64, 13×17):          4 点

Step 3  双线性插值取 value
        在 16 个坐标处读 V_l → 16 个 d/8 维向量

Step 4  加权求和
        out_7 = Σ A_{lqk} · V_l(p_q + Δp)  →  (256-d)
        再经 output_proj

Step 5  接 cls/bbox head → 「猫」+ 框
```

**小物体为何受益**：猫若只占几个 stride-8 像素，Level 0 上的 4 个采样点可以精确落在猫身；DETR 只能在 25×33 粗网格上均匀搜。

### 3.6 Encoder Self-Attn：DETR 全局 vs Deformable 稀疏

Encoder 侧的差异同样关键——DETR 用 O(hw²) 全局 self-attn；Deformable 也必须改，否则 S≈17700 时 self-attn 不可承受。

```
DETR Encoder（第 l 层，单 token 视角）:
  token_i 输出 = Σ_{j=1}^{hw}  α_{ij} · v_j
  每个 token 与 **全部 825** token 交互
  6 层后：每个位置「看过」全图，但计算 hw²×6

Deformable Encoder:
  token_i 的 reference = 自己在 **所在层** 的网格中心 (x_i, y_i)
  输出 = Σ_{l=1}^{L} Σ_{k=1}^{K}  A_{ilk} · V_l( ref_i + Δp_{ilk} )

  token_i 只从 **L×K=16** 个连续位置读 value
  但 16 个点可以跨 **4 个尺度** → 仍能获得 multi-scale 上下文
  计算：O(S·L·K) per layer，S 虽大但 L·K 常数小
```

**与 DETR 的语义对应**：

| DETR Encoder | Deformable Encoder |
|--------------|-------------------|
| 825 token 全局混合 | S token 稀疏跨尺度混合 |
| 1 张 stride-32 特征图 | 4 张不同 stride 特征图 |
| ref 隐含在 pos enc 里 | ref = 网格中心，显式参与采样 |
| 感受野一步到位（一层 attn 全图） | 感受野靠多层 stack + 偏移扩大 |

---

## 4. 多尺度特征构造

DETR ** deliberately 不用 FPN**，只用 conv5；Deformable DETR **必须** 引入多尺度——否则 MSDeformAttn 的 L 层采样都落在同一 stride 上，小物体问题无法解决。

### 4.0 与 DETR 的特征输入对比

```
DETR:
  ResNet → conv5 (stride 32) → 1×1 Conv → (B, 256, h, w)
  flatten → (hw, B, 256)     hw ≈ 825
  每个 token 对应原图 32×32 像素块

Deformable DETR:
  ResNet → {C3, C4, C5} + conv(C5→C6)
  4 层分别投影 → 各 (B, 256, H_l, W_l)
  + pos_enc + level_embed[l]
  concat flatten → (S, B, 256)   S ≈ 17721

  Level 0: 1 token ≈ 原图 8×8 像素   ← DETR 完全没有这一粒度
  Level 1: 1 token ≈ 16×16
  Level 2: 1 token ≈ 32×32          ← 与 DETR 唯一重叠的尺度
  Level 3: 1 token ≈ 64×64
```

**level_embed 是 DETR 没有的**：同一 (x,y) 在四个尺度会出现四个 token，需可学习向量区分「这是第几层特征」，否则 MSDeformAttn 无法判断该信哪层。

### 4.1 四层特征从哪来

默认 **num_feature_levels = 4**（`r50_deformable_detr.sh`）：

```
Backbone ResNet 输出中间层:
  C3: stride 8,  通道 512
  C4: stride 16, 通道 1024
  C5: stride 32, 通道 2048

额外层:
  C6: 对 C5 做 3×3 conv stride=2 → stride 64

每层经 1×1 Conv + GroupNorm 投影到 d=256
```

单尺度变体（ablation）：只用 C5（stride 32），验证 MSDeformAttn 不含多尺度时仍有效。

### 4.2 拼接与 level embedding

```
对每层 l:
  src_l: (B, 256, H_l, W_l)
  pos_l: 2D 正弦位置编码
  src_l + pos_l + level_embed[l]     ← level_embed 是可学习 (L, 256)

flatten 并 concat:
  src_flatten: (B, S, 256)    S = Σ H_l·W_l

例: 输入 800×1066 padding 后
  Level0: 100×133 = 13300
  Level1:  50×67  =  3350
  Level2:  25×34  =   850
  Level3:  13×17  =   221
  S ≈ 17721（比 DETR 单尺度 hw≈825 大，但 attention 稀疏）
```

辅助张量：

```
spatial_shapes:      (L, 2)    每层的 (H_l, W_l)
level_start_index:   (L,)      每层在 flatten 序列中的起始 index
valid_ratios:        (B, L, 2) 有效区域占 padding 后尺寸的比例，修正参考点
```

### 4.3 valid_ratio 的作用

图像 padding 后，有效像素区域小于 tensor 尺寸。参考点归一化时需按 **有效宽高** 计算：

```
ref_x = (grid_x + 0.5) / (W_l × valid_ratio_w)
ref_y = (grid_y + 0.5) / (H_l × valid_ratio_h)

保证 reference point 落在真实图像区域内，而非 padding 区
```

---

## 5. Encoder：Deformable Self-Attention

### 5.0 结构对比（DETR vs Deformable）

```
┌──────────────────────── DETR Encoder 一层 ────────────────────────┐
│  MultiheadAttention:  Q,K = x+pos,  V = x                           │
│    Attention matrix: (825 × 825)  每个 token 看全部 token             │
│  FFN                                                                │
└─────────────────────────────────────────────────────────────────────┘

┌────────────────────── Deformable Encoder 一层 ──────────────────────┐
│  MSDeformAttn:  query = x+pos                                       │
│    ref_points = 各 token 在自己层的网格中心 (B, S, L, 2)              │
│    每个 token 只采 L×K 个点，Attention 等效于 16 路 soft pooling     │
│  FFN                                                                │
└─────────────────────────────────────────────────────────────────────┘
```

DETR Encoder 一层参数量主要在 Q/K/V/O（4×256²）；Deformable 额外有 `sampling_offsets` / `attention_weights` 两个 Linear，但 **避免了 825² 的 attention 矩阵**。

### 5.1 结构（每层）

```
DeformableTransformerEncoderLayer:
  MSDeformAttn (self)  →  LayerNorm  →  FFN  →  LayerNorm

与 DETR Encoder 区别:
  ✗ 标准 MultiheadAttention（全局 hw×hw）
  ✓ MSDeformAttn（每 token 采 L×K 点）
```

### 5.2 Encoder 的 reference point 从哪来

**每个 spatial token 以自身网格中心为参考点**（无需预测）：

```
get_reference_points(spatial_shapes, valid_ratios):

对第 l 层 H×W 网格:
  ref_x = (0.5, 1.5, ..., W-0.5) / (W × valid_ratio_w)
  ref_y = (0.5, 1.5, ..., H-0.5) / (H × valid_ratio_h)

所有层 concat → (B, S, L, 2)

含义: 第 l 层的 token 在 MSDeformAttn 里以「自己在第 l 层的格子中心」为主参考，
       但可以从 **所有 L 层** 的 K 个偏移点采样（跨尺度 self-attn）
```

### 5.3 Encoder 在学什么

```
CNN 各层特征仍是局部的
Deformable Encoder 6 层堆叠:
  每个位置 token 通过稀疏采样「看」其他尺度、其他位置的上下文
  计算量可控（O(S·L·K) per layer）的前提下建立 multi-scale 全局混合

输出 memory 供 Decoder cross-attn 读取
```

---

## 6. Decoder：Query、Reference Point 与迭代 Refine

Decoder 是 DETR 与 Deformable DETR **差异最集中** 的模块：Self-Attn 相同，Cross-Attn 换 MSDeformAttn，并新增 reference point 生命周期。

### 6.0 Decoder 逐层对照表

| 步骤 | DETR Decoder 一层 | Deformable Decoder 一层 |
|------|-------------------|-------------------------|
| 输入 query | tgt + query_embed | tgt + query_pos（同结构，512→拆 256+256） |
| Self-Attn | 标准 MHA，(100×100) | **相同**，标准 MHA，(300×300) |
| Cross-Attn | Q·K 对 hw=825 keys | MSDeformAttn，L×K=16 采样点 |
| Cross 的「看哪里」 | 由 Q·K  softmax 隐式决定 | ref point + Δp **显式决定** |
| 层间框更新 | **无**（只有最后一层出框） | **有**（with_box_refine：每层更新 ref） |
| 输出到 head | 仅最后一层（+aux） | 每层 aux + 最后一层（默认全开） |

```
DETR Decoder 层间状态:
  query 向量变，但 **没有** 显式坐标状态
  框只在 bbox MLP 最后一层一次性输出

Deformable Decoder 层间状态:
  query 向量变 + reference_points 逐层 refine
  Layer ℓ 输出 → bbox_embed[ℓ] → new_ref → 供 Layer ℓ+1 的 MSDeformAttn 采样
  「先粗定位采样 → 再细定位采样」的 cascade
```

### 6.1 Query 初始化（单阶段，默认）

```
query_embed: nn.Embedding(300, 512)  →  split 两半
  query_pos (256-d):  加在 attention 的 Q/K 上
  tgt content (256-d): 初始 target 向量（类似 DETR）

reference_points = Linear(query_pos).sigmoid()   → (B, 300, 2)
  每个 query 一开始就有一个归一化 (cx, cy) 猜测
  作为 MSDeformAttn 的采样中心
```

相对 DETR：DETR 的 query **没有** 显式 spatial prior；Deformable DETR **强制** 每个 query 带一个参考坐标。

```
DETR:
  query_embed (100, 256)  →  直接进 Decoder
  cross-attn 的「中心」完全由训练后的 Q·K 模式决定

Deformable:
  query_embed (300, 512)  →  split → query_pos + tgt
  ref_0 = sigmoid( Linear(query_pos) )   (300, 2)
  MSDeformAttn 以 ref_0 为圆心做第一次采样

  即使 cross-attn 权重尚未学好，ref_0 也把采样限制在图像某区域
  → 匈牙利匹配给的梯度更「局部」→ 收敛快
```

### 6.2 Decoder 单层顺序

```
DeformableTransformerDecoderLayer（注意顺序与 DETR 不同）:

  ① Self-Attn (标准 MultiheadAttention, query↔query)
  ② MSDeformAttn (cross, query → memory)
  ③ FFN

DETR 顺序是 self → cross → FFN；Deformable 相同，但 cross 换成 MSDeformAttn
```

Cross-attn 调用：

```
MSDeformAttn(
  query = tgt + query_pos,
  reference_points = ref_points × valid_ratios,   # (B, N, L, 2)
  input_flatten = memory,
  ...
)
```

### 6.3 迭代边界框 refine（with_box_refine）

默认强配置开启。**每层 Decoder 输出后更新 reference points**：

```
对 Decoder 第 lid 层输出 output_l:

  Δbox = bbox_embed[lid](output_l)          # 共享或每层独立 MLP
  new_ref = sigmoid( inverse_sigmoid(ref) + Δbox )   # 4 维 (cx,cy,w,h)

  reference_points = new_ref.detach()       # detach 稳定训练
  下一层 MSDeformAttn 在新 ref 附近采样

最后一层 ref 即最终框（或与 pred head 输出一致）
```

**原理**：

```
Layer 1: ref 粗略 → 采样点大致覆盖物体区域 → 粗分类+粗框
Layer 2: ref 更准 → 采样点更集中 → 细化
...
Layer 6: ref 精确 → 16 个采样点高度集中在目标上 → 精框
```

类似 cascade refinement，但在 **Transformer Decoder 内部** 完成，且采样位置随 ref 联动。

#### 与 DETR 框回归的差异

```
DETR:
  6 层 Decoder 只传递 hidden state
  bbox_embed 仅在 **最后一层** hs 上预测 (cx,cy,w,h)
  中间层 aux loss 监督的是「一次性框猜测」，层间 **不** 更新空间位置

Deformable (+ with_box_refine):
  每层 lid:  Δ = bbox_embed[lid](hs[lid])
             ref_{lid+1} = σ( σ^{-1}(ref_lid) + Δ )   detach
  MSDeformAttn 在 **ref_{lid}** 附近采样，而非永远在同一位置

  层 1: ref 在物体大致区域 → 粗特征
  层 6: ref 贴近真框 → 采样点像「落在物体轮廓上」

  bbox head 从「一次猜框」变成「级联修正框 + 级联修正采样中心」
```

DETR 的框完全由 **最后一层 MLP 从语义向量解码**；Deformable 的框还 **反过来控制下一层读哪里**——这是收敛加速的核心机制之一。

### 6.4 两阶段变体（two-stage）

```
Stage A  Encoder memory 上对每个 spatial 位置做 dense 分类（objectness）+ box 预测
         gen_encoder_output_proposals() 生成 anchor-like proposals

Stage B  取 top-K（K=300）高分 proposal 的 (coord, feature) 作为 Decoder query 初始化
         query_embed / ref_points 由 Encoder 输出构造，而非 nn.Embedding

效果: query 起点更准 → 收敛更快、AP 更高（+0.7 左右）
代价: 不再是最「纯」的固定 query DETR；工程上 two-stage 是常见最强配置
```

---

## 7. 预测头、匹配与损失

### 7.0 与 DETR 相同 vs 不同

```
与 DETR 完全相同（集合预测「外壳」）:
  ✓ 匈牙利二分匹配分配 query ↔ GT
  ✓ Set Loss = λ_cls·CE + λ_box·L1 + λ_giou·GIoU
  ✓ ∅ 类 + eos_coef=0.1
  ✓ 推理：score 阈值过滤，无 NMS

与 DETR 不同（训练「内壳」）:
  ✗ query 数 100 → 300
  ✗ 每层 Decoder 输出 **都** 算 loss（Deformable 默认 return_intermediate_dec=True，且与 refine 强绑定）
  ✗ 每层有独立 bbox_embed[lid]（iterative refine 时）
  ✗ 匹配代价 λ 默认 cls=2（DETR 为 1），略更重视分类在匹配阶段的作用
  ✗ 推理阈值 0.3（DETR 常用 0.7）—— 因模型校准不同，非框架本质差异
```

**关键理解**：换 MSDeformAttn **不改变** 损失函数形式；改变的是 **更快产生可匹配的 (class, box) 预测**，使匈牙利匹配在训练早期就能给出有意义的监督。

### 7.1 与 DETR 的继承关系

| 组件 | Deformable DETR |
|------|-----------------|
| 分类头 | Linear(d → K+1)，含 ∅ |
| 回归头 | MLP(d → 4)，sigmoid → cxcywh |
| 匈牙利匹配 | 同 DETR：λ_cls=2, λ_box=5, λ_giou=2（官方默认） |
| Set Loss | CE + L1 + GIoU；∅ 降权 eos_coef=0.1 |
| 辅助损失 | **每层** Decoder 输出都算 loss（return_intermediate_dec=True） |

**query 数**：默认 **N=300**（DETR 为 100）—— 更多槽位有利于 crowded 场景与 two-stage top-K。

### 7.2 辅助损失与 iterative refine 的配合

```
return_intermediate_dec=True 时:
  6 层 Decoder 每层输出 hs_l 都接 class_embed[l] / bbox_embed[l]
  每层独立匈牙利匹配 + loss

好处:
  浅层也被迫学「粗定位」，不只靠最后一层
  与 reference point 逐层 refine 形成 **深监督 + 坐标级联**
```

---

## 8. 训练配置与推理

### 8.1 优化（默认 COCO）

| 项目 | 设置 |
|------|------|
| 优化器 | AdamW |
| backbone LR | 2e-5 |
| 其余 LR | 2e-4 |
| batch size | 32 |
| 训练轮数 | **50 epoch**（lr drop 40 epoch） |
| 梯度裁剪 | max_norm = 0.1 |

### 8.2 推理

```
与 DETR 相同:
  300 预测 → score = max(softmax 80 类) →  threshold 0.3（Deformable 默认低于 DETR 的 0.7）
  无 NMS
```

---

## 9. 核心原理深讲：相对 DETR 改了什么、为什么有效

### 9.0 三项改动的因果链

```
DETR 痛点                Deformable 改动                    直接效果
─────────────────────────────────────────────────────────────────────────
cross-attn 825 键难学  → MSDeformAttn 16 点 + ref        梯度集中、有采样中心
单尺度 stride 32       → FPN 4 尺度 + level_embed         小物体有 high-res value
框与采样位置脱节       → 逐层 refine ref                  层间 cascade，采样跟着框走
Encoder O(hw²)         → Encoder 也改 MSDeformAttn        多尺度 S 大但仍可训练
300 epoch 才收敛       → 以上叠加 + 50 epoch schedule     实用训练成本
```

**未改动的部分**（与 DETR 完全一致）：集合预测定义、匈牙利匹配、∅ 类、无 NMS 推理、Set Loss 形式。  
因此 Deformable DETR 是 **同一范式下的 attention 机制升级**，不是新检测框架。

### 9.1 为什么稀疏采样能加速收敛（相对 DETR）

```
DETR  cross-attn 梯度:
  ∂L/∂q  分散到 hw 个 key 上，每个 key 分到的梯度 ∝ softmax weight
  初期 weight 均匀 → 每个 spatial 位置梯度极小 → 定位信号弱

Deformable:
  梯度只通过 K×L 个采样点回传
  reference point 把采样局限在目标邻域
  softmax 在 16 个（而非 825 个）点上竞争 → 赢家通吃更明显 → 定位梯度更大
```

**相对 DETR 的梯度对比（直觉）**：

```
DETR 第 1 个 epoch，猫所在格 i* 分到的 cross-attn 梯度:
  ∂L/∂α_{i*} ≈ (1/825) × (某个小量)     均匀 softmax 下极其微弱

Deformable 第 1 个 epoch，若 ref 已在猫附近:
  16 个点中 2~3 个落在猫身，A 在这 2~3 点 softmax 竞争
  ∂L/∂A 更集中 → ∂L/∂Δp、∂L/∂ref 更大 → query 更快学会「往猫挪」
```

### 9.2 多尺度 + 可变形：分工（及 DETR 缺什么）

| 机制 | 职责 |
|------|------|
| **多尺度特征** | 提供不同粒度的 value map（小物体看 L0，大物体看 L2/L3） |
| **可变形偏移 Δp** | 在每层特征上 **微调** 读哪里，补偿 grid 离散化 |
| **reference point** | 决定 **在哪块区域** 开始搜 |
| **A_{lqk} weight** | 决定 **更信哪层、哪个采样点** |

DETR 只有 stride-32 一种 value，**不存在「选哪层」**；Deformable 的 A_{lqk} 可学到「小物体 query 把权重放在 Level 0，大物体 query 放在 Level 2/3」——这是 AP_S 提升 +6 AP 的主因之一。

### 9.3 复杂度对照（量级）

设 S=ΣH_l W_l ≈ 1.7×10⁴，L=4，K=4，N=300，d=256，heads=8：

| 模块 | DETR | Deformable DETR |
|------|------|-----------------|
| Encoder attn | O(S²·d) ≈ 10⁹ 级 | O(S·L·K·d) ≈ 10⁶ 级 |
| Decoder cross | O(N·S·d) | O(N·L·K·d) ≈ 10⁵ 级 |
| Decoder self | O(N²·d) | O(N²·d) 相同 |

注意：Deformable 的 S > hw，但 **不会** 做 S²；这是相对 DETR 能加多尺度仍可控的原因。

### 9.4 训练动态：50 epoch vs DETR 300 epoch

```
DETR 典型 AP 曲线（概念）:
  epoch 0~100:   AP 很低，query 尚未 specialization
  epoch 100~300: AP 缓慢爬升
  epoch 300+:    才到 42 AP 附近

Deformable 典型 AP 曲线:
  epoch 0~10:    ref 初始化 + 星形 offset → 已有局部采样，AP 快速上升
  epoch 10~40:   refine + 多尺度权重分化
  epoch 50:      44.5 AP（多尺度默认）

差异来源（按贡献排序，经验上）:
  1. reference point 提供 spatial prior（DETR 无）
  2. MSDeformAttn 稀疏梯度（DETR 825 维 softmax）
  3. 多尺度 value（DETR 单 stride-32）
  4. iterative bbox refine（DETR 仅末层框）
  5. 更多 query + 每层 aux（DETR 100 query，aux 有但无 ref 联动）
```

### 9.5 信息流动总图（标注相对 DETR 的新增路径）

```
                    ┌─────────────────────────────────────┐
                    │  多尺度 CNN 特征（局部语义）          │
                    └─────────────────────────────────────┘
                                      │
                    flatten + pos + level_embed
                                      ▼
                    ┌─────────────────────────────────────┐
                    │  Deformable Encoder                  │
                    │  每 token: 以自身为 ref，稀疏采 L×K 点 │
                    └─────────────────────────────────────┘
                                      │ memory
                    query_embed + init ref (cx,cy)
                                      ▼
                    ┌─────────────────────────────────────┐
                    │  Deformable Decoder ×6               │
                    │  self-attn 协调 → MSDeformAttn 定位  │
                    │  → refine ref/box → 下一层            │
                    └─────────────────────────────────────┘
                                      │
                              cls / bbox head
                                      ▼
                         匈牙利匹配 → Set Loss（与 DETR 相同）
```

**新增路径（相对 DETR）** 用 `*` 标出：

```
* level_embed 区分 4 尺度
* ref_0 = Linear(query_pos)  初始化采样中心
* 每层 MSDeformAttn 读 L×K 点而非 hw 键
* ref_{l+1} = refine(ref_l, bbox_l)  层间 cascade
```

---

## 10. 维度推导（默认 R50 多尺度，800×1066）

| 符号 | 含义 | 典型值 |
|------|------|--------|
| L | 特征层数 | 4 |
| K | 每层采样点 | 4（enc/dec） |
| S | flatten 总 token 数 | ~17700 |
| N | query 数 | 300 |
| d | d_model | 256 |
| heads | 头数 | 8 |

### 10.1 MSDeformAttn 张量

```
query:              (B, Len_q, d)
reference_points:   (B, Len_q, L, 2)  或 (B, Len_q, L, 4)
input_flatten:      (B, S, d)
spatial_shapes:     (L, 2)

sampling_offsets:   (B, Len_q, heads, L, K, 2)
attention_weights:  (B, Len_q, heads, L, K)   softmax over L×K
value:              (B, S, heads, d/heads)

output:             (B, Len_q, d)
```

### 10.2 Decoder 一层

```
tgt:                (B, 300, 256)
ref_points:         (B, 300, 2)  → broadcast (B, 300, L, 2)

Self-Attn:          (300, B, 256) 标准 MHA
MSDeformAttn cross: Len_q=300, 每 query 采 L×K=16 点
FFN:                (B, 300, 256)

bbox_embed 更新 ref: (B, 300, 4)
```

---

## 11. MSDeformAttn CUDA 算子深讲

§3 从 **算法** 讲了 MSDeformAttn 算什么；本节从 **实现** 讲官方如何用 CUDA 把它算快，以及和 PyTorch 参考实现的对应关系。

### 11.1 为什么需要自定义 CUDA

MSDeformAttn 的前向本质是：**对每个 query、每个 head、每个尺度、每个采样点做双线性插值，再按 attention weight 加权求和**。

```
朴素 PyTorch 实现（ms_deform_attn_core_pytorch）:
  for level l in 0..L-1:
    value_l reshape → (N*M, D, H, W)
    grid_sample(value_l, sampling_grid_l)    → (N*M, D, Lq, P)
  stack + multiply attention_weights + sum

问题:
  ① L 次 grid_sample + 多次 kernel launch，Python 循环开销大
  ② 中间 tensor 多，显存带宽浪费
  ③ Encoder/Decoder 每层都调用，MSDeformAttn 是 **绝对热点**
```

DETR 的标准 Attention 只用 `torch.matmul` + `softmax`，无自定义算子。  
Deformable DETR 的瓶颈在 **稀疏采样聚合**，故借鉴 Deformable Conv 的 im2col 思路，写成 **单个融合 CUDA kernel**（`MultiScaleDeformableAttention` 扩展）。

### 11.2 调用链（从 Python 到 GPU）

```
MSDeformAttn.forward()                     models/ops/modules/ms_deform_attn.py
    │  计算 sampling_locations, attention_weights
    │  value = value_proj(input_flatten)
    ▼
MSDeformAttnFunction.apply(...)            models/ops/functions/ms_deform_attn_func.py
    │  forward → MSDA.ms_deform_attn_forward(...)
    │  backward → MSDA.ms_deform_attn_backward(...)
    ▼
ms_deform_attn_cuda_forward / backward     models/ops/src/cuda/ms_deform_attn_cuda.cu
    │  按 im2col_step 切 batch
    ▼
ms_deformable_im2col_cuda                  ms_deform_im2col_cuda.cuh  ← 核心 kernel
ms_deformable_col2im_cuda                  （backward，反传 value / loc / weight）
```

编译产物：在 `models/ops/` 下 `sh make.sh`，生成 Python 可 import 的 `MultiScaleDeformableAttention` 包。

### 11.3 CUDA 入口张量约定

`ms_deform_attn_cuda_forward` 接收的 tensor 布局（与 Python 侧对齐）：

| 参数 | 形状 | 含义 |
|------|------|------|
| `value` | **(B, S, M, C)** | 多尺度 flatten 后的 value；M=num_heads，C=d/M |
| `spatial_shapes` | **(L, 2)** | 每层 (H_l, W_l) |
| `level_start_index` | **(L,)** | 每层在 S 维上的起始 index |
| `sampling_loc` | **(B, Lq, M, L, P, 2)** | 归一化采样坐标 (x, y) ∈ [0,1] |
| `attention_weights` | **(B, Lq, M, L, P)** | 已对 L×P 做 softmax 的权重 |
| `im2col_step` | int，默认 **64** | batch 分块大小 |

输出：

```
output: (B, Lq, M*C)  →  Python 侧 reshape 为 (B, Lq, d_model)
```

**与 §10.1 的对应**：`Len_q` = Lq（Encoder 时 Lq=S，Decoder 时 Lq=N=300）；`heads` = M；`d/heads` = C。

### 11.4 im2col_step：batch 分块

```cpp
const int im2col_step_ = std::min(batch, im2col_step);   // 默认 64
// 要求 batch % im2col_step_ == 0

for (int n = 0; n < batch / im2col_step_; ++n) {
  // 每次处理 batch_n = im2col_step_ 张图
  ms_deformable_im2col_cuda(..., value + n * im2col_step_ * per_value_size, ...);
}
```

**作用**：限制单次 kernel 的工作集，降低峰值显存；大 batch 时分块多次调用。  
训练 batch=32 时通常 `im2col_step=64` 实际等于一次处理整个 batch。

### 11.5 前向 kernel 在算什么

每个 CUDA 线程负责输出 `output[b, q, m, c]` 的一个标量，即 **第 b 张图、第 q 个 query、第 m 个 head、第 c 维 channel** 的聚合结果：

```
output[b,q,m,c] = Σ_{l=0}^{L-1} Σ_{p=0}^{P-1}
                      attn_w[b,q,m,l,p] · BilinearSample(value[b], l, m, c, loc[b,q,m,l,p])
```

**BilinearSample** 由 device 函数 `ms_deform_attn_im2col_bilinear` 实现：

```
输入: 连续坐标 (h, w)  在特征图 l 上的 **像素坐标**（已由归一化 loc 换算）
      floor → 四个整数格点 (h_low,w_low), (h_high,w_low), ...

双线性权重:
  lh = h - h_low,  lw = w - w_low
  w1 = (1-lh)(1-lw),  w2 = (1-lh)lw,  w3 = lh(1-lw),  w4 = lh·lw

val = w1·V[p1] + w2·V[p2] + w3·V[p3] + w4·V[p4]

边界: 超出 [0,H-1]×[0,W-1] 的格点贡献置 0（zero padding）
```

**value 内存布局**（flatten 后按层切片）：

```
对 level l，offset = level_start_index[l]
在 (H_l, W_l) 网格上，位置 (y,x) 的 head m、channel c:

  ptr = base + y * (W * M * C) + x * (M * C) + m * C + c

即 spatial 维和 head/channel 维 **交织存储**，便于 coalesced 读取
```

### 11.6 坐标换算：归一化 loc → 像素 (h, w)

Python 侧 `MSDeformAttn.forward` 中（reference 为 2D 时）：

```
sampling_locations = ref + offsets / (W_l, H_l)     归一化到 [0, 1]

CUDA kernel 内再映射到像素索引（概念上）:
  h_im = loc_y * H_l - 0.5
  w_im = loc_x * W_l - 0.5
```

PyTorch 参考实现则用 `grid_sample` 约定：

```
sampling_grids = 2 * sampling_locations - 1    映射到 [-1, 1]
grid_sample(..., align_corners=False)
```

两套约定等价：**align_corners=False** 时 `[-1,1]` 端点对应像素中心的外延，与 CUDA 中 `-0.5` 偏移一致。读源码时勿混淆「[0,1] 归一化」与「grid_sample 的 [-1,1]」。

### 11.7 PyTorch 参考实现（调试对照）

`ms_deform_attn_func.py` 中 `ms_deform_attn_core_pytorch` **仅用于 debug/test**，注释写明 production 必须用 CUDA 版：

```python
# 逻辑等价于 CUDA kernel 的分解写法:
for lid, (H, W) in enumerate(spatial_shapes):
    value_l = value_list[lid].reshape(N*M, D, H, W)
    sampling_grid_l = (2 * sampling_locations[..., lid, :] - 1)  # (N, M, Lq, P, 2)
    sampled = F.grid_sample(value_l, sampling_grid_l, mode='bilinear', align_corners=False)
    # → (N*M, D, Lq, P)

output = (stack(sampled) * attention_weights).sum(-1)   # 对 L×P 加权
```

**读 CUDA 前建议**：先用 PyTorch 版在小 tensor 上 print shape，理解 `(N*M, D, H, W)` 与 `(N, Lq, M, L, P, 2)` 的 broadcast 关系。

### 11.8 反向传播：三路梯度

`ms_deform_attn_cuda_backward` 返回：

| 梯度 | 形状 | 回传给谁 |
|------|------|----------|
| `grad_value` | (B, S, M, C) | value_proj ← memory，继续反传到 Encoder/Backbone |
| `grad_sampling_loc` | (B, Lq, M, L, P, 2) | sampling_offsets ← query → **ref 与 offset 都可学** |
| `grad_attn_weight` | (B, Lq, M, L, P) | attention_weights ← query |

`col2im` kernel `ms_deform_attn_col2im_bilinear` 做两件事：

```
① 对 value 的四邻域格点 atomicAdd（双线性反传）
   top_grad_value = grad_output * attn_weight

② 对采样坐标 (h, w) 求导 → grad_sampling_loc
   利用四邻域像素值对 h,w 的偏导，链式法则累加

③ grad_attn_weight = grad_output * bilinear_sampled_value
```

**@once_differentiable**：backward 仅实现一次求导，高阶导不支持（检测训练足够）。

**spatial_shapes / level_start_index 无梯度**：层结构是固定的。

### 11.9 与 DETR Attention 的实现对比

| | DETR MultiheadAttention | MSDeformAttn CUDA |
|--|-------------------------|-------------------|
| 核心算子 | cuBLAS matmul + softmax | 自定义 bilinear im2col |
| 是否需要编译 | 否 | **是**（`sh make.sh`） |
| 主要 FLOPs | QK^T: Lq×S×d | Lq×M×L×P×C 次插值 + 乘加 |
| 内存热点 | (Lq, S) attention 矩阵 | 无大矩阵；多次读 value |
| CPU 训练 | 可行 | 极慢，几乎不可用 |

### 11.10 工程注意事项

```
编译:
  cd models/ops && sh make.sh
  python test.py          # 单元测试 forward/backward 与 PyTorch 版对齐

常见报错:
  batch must divide im2col_step  → 调整 batch size 或 im2col_step
  value tensor has to be contiguous  → .contiguous() 再传入

性能建议（源码 warning）:
  d_model / n_heads 应为 2 的幂（默认 256/8=32 ✓）
  CUDA kernel 对 head 维度有特殊内存对齐优化

HuggingFace transformers 实现:
  纯 PyTorch grid_sample 复现，无自定义 CUDA，易部署但较慢
  官方 Deformable-DETR repo 训练仍推荐编译 ops
```

### 11.11 单 query 在 kernel 中的工作量（数量级）

默认 Decoder cross-attn：`Lq=300, M=8, L=4, P=4, C=32`：

```
每个 query 输出 d=256 维:
  每维 c 来自 L×P=16 次双线性采样 × 每次 4 格点读取
  总采样次数 ≈ 300 × 8 × 16 × 32 ≈ 1.2M 次/图/层/MSDeformAttn

6 层 Decoder + 6 层 Encoder 均调用 MSDeformAttn
  → 该算子占训练时间大头，CUDA 融合价值高
```

---

## 12. 常见困惑 FAQ

| 问题 | 回答 |
|------|------|
| **MSDeformAttn 和 Deformable Conv 什么关系？** | 思想同源（学习偏移采样位置）；MSDeformAttn 是 **attention 版**，采样点权重由 A_{lqk} 动态决定，且跨 **多尺度** feature map。 |
| **Encoder 为什么也用 Deformable？** | 多尺度 S 很大，全局 self-attn 仍 O(S²)；Deformable self-attn 使 Encoder 可处理 FPN 拼接后的长序列。 |
| **reference point 和 query_pos 区别？** | `query_pos` 是 attention 的 Q/K 偏置 embedding；`reference_points` 是 **空间坐标**，告诉 MSDeformAttn 去哪采样 value。 |
| **为何 query 从 100 增到 300？** | 两阶段需 top-300 proposal；更多槽位提升 crowded 场景召回；匹配代价略增但可接受。 |
| **迭代 refine 为何 detach ref？** | 防止梯度通过 ref 更新路径不稳定；下一层把 ref 当 **常数中心** 重新采样。 |
| **50 epoch 就够的原因？** | 稀疏 attn + ref 初始化 + 多尺度 + 辅助 loss 共同作用；不是单一 trick。 |
| **CUDA ops 必须编译？** | 是；见 **§11** 全文。PyTorch 参考实现仅 debug，训练必须用 `MultiScaleDeformableAttention` CUDA 扩展。 |

---

## 13. 源码阅读顺序

| 概念 | 文件 |
|------|------|
| MSDeformAttn 模块 | `models/ops/modules/ms_deform_attn.py` |
| Autograd 封装 | `models/ops/functions/ms_deform_attn_func.py` |
| CUDA 入口 | `models/ops/src/cuda/ms_deform_attn_cuda.cu` |
| CUDA kernel | `models/ops/src/cuda/ms_deform_im2col_cuda.cuh` |
| PyTorch 对照 | `ms_deform_attn_core_pytorch`（同 func 文件） |
| Encoder/Decoder | `models/deformable_transformer.py` |
| 多尺度 backbone 拼接 | `models/backbone.py`, `models/deformable_detr.py` |
| 迭代 refine / two-stage | `models/deformable_transformer.py` Decoder.forward |
| 匈牙利 + Loss | 继承 DETR 的 `matcher.py`, `deformable_detr.py` SetCriterion |

**建议顺序**：

1. `ms_deform_attn_core_pytorch` → 用 PyTorch 理解算法（§11.7）  
2. `MSDeformAttn.forward` → sampling_offsets / attention_weights 如何进 CUDA  
3. `MSDeformAttnFunction` → forward/backward 接口  
4. `ms_deform_attn_im2col_bilinear` → 双线性采样 device 函数（§11.5）  
5. `DeformableTransformerEncoder.get_reference_points` → Encoder ref 构造  
6. `DeformableTransformerDecoder.forward` → ref 逐层 refine  

---

## 14. 文档阅读路线

| 目标 | 章节 |
|------|------|
| **与 DETR 差在哪** | **§1.5** → §3.0 → §6.0 → §9.0 |
| 5 分钟懂改进点 | §1.2 ~ §1.3 + §2 |
| 搞懂 MSDeformAttn | §3 全文（尤其 §3.0、§3.6） |
| 搞懂多尺度 | §4（§4.0 与 DETR 对比） |
| 搞懂 Decoder/refine | §6（§6.0、§6.3 与 DETR 框回归差异） |
| 原理与收敛 | §9 全文 |
| **CUDA 算子实现** | **§11** |
| 维度与张量 | §10 + §11.3 |
| 源码对照 | §13 |

---

## 15. 小结

Deformable DETR **没有改 DETR 的集合预测哲学**（Object Query + 匈牙利匹配 + 无 NMS），而是改 **Attention 怎么读特征、特征从哪来、框如何驱动采样**：

| 相对 DETR | 改动 | 效果 |
|-----------|------|------|
| Cross/Enc Attn | 全局 Q·K → **MSDeformAttn（ref + Δp + 16 点）** | 计算降、梯度聚焦、收敛快 |
| 特征 | 单尺度 conv5 → **4 尺度 FPN + level_embed** | AP_S 大幅提升 |
| 空间状态 | 无 ref → **ref 初始化 + 逐层 refine** | 采样中心 cascade，层间联动 |
| Query 数 / epoch | 100 / 300+ → **300 / 50** | 工程可训 |

三者（稀疏 attn + 多尺度 + ref cascade）叠加，成为 DETR 系列 **第一个实用基线**，并延续到 DINO 等后续工作。

---

## 参考文献

1. Zhu X., et al. **Deformable DETR: Deformable Transformers for End-to-End Object Detection.** ICLR 2021.  
2. Carion N., et al. **End-to-End Object Detection with Transformers.** ECCV 2020.  
3. Dai J., et al. **Deformable Convolutional Networks.** ICCV 2017.（可变形采样思想渊源）  
4. Lin T-Y., et al. **Feature Pyramid Networks for Object Detection.** CVPR 2017.（多尺度结构）
