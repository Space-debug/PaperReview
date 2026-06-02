# ViT（Vision Transformer）

> 本文档用于整理 ViT（An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale）论文精读笔记。  
> 重点：**Patch 序列化与嵌入**、**标准 Transformer Encoder**、**位置编码与 [CLS] 读出**，以及 **JFT 监督预训练 / DeiT 蒸馏 / MAE 掩码预训练** 三条训练路线与处理方式。  
> **想快速建立整体印象**：先读 **§1.5 ~ §1.9**，再按需深入后续章节。

---

## 1. 概览

### 1.1 基本信息

| 项目 | 内容 |
|------|------|
| 论文 | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale |
| 作者/机构 | Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby（Google Research, Brain Team, Zurich） |
| 发表 | ICLR 2021 |
| 任务 | 图像分类（将 NLP 中的 Transformer 几乎原样用于视觉，验证「纯 attention、弱归纳偏置 + 大数据」的可行性） |
| 代码 | [google-research/vision_transformer](https://github.com/google-research/vision_transformer) |

### 1.2 核心思想（一句话）

**把图像切成固定大小的 Patch，线性投影成 token 序列， prepend 一个 [CLS] token，用标准 Transformer Encoder 做全局 self-attention，最后用 [CLS] 表示做分类——几乎不用卷积，靠大规模预训练弥补缺失的局部性/平移等归纳偏置。**

ViT 的关键突破不是「又一个 ImageNet SOTA」，而是系统证明了：

1. **CNN 不是视觉识别的必需品**：纯 Transformer 在足够数据 + 足够算力下可以匹配或超越 ResNet；
2. **Patch = Word**：视觉任务可以被表述为「对 patch 序列的建模」，与 BERT/GPT 的 token 序列同构；
3. **Scale matters**：小数据上 ViT 不如 ResNet（缺归纳偏置）；大数据预训练后 ViT 的 scaling 曲线更陡——为后续 Swin、DeiT、MAE 及 DETR 系 backbone 铺路。

### 1.3 整体流水线

```
Input Image (batch, 3, H, W)     例如 H=W=224（预训练）；微调可 384/512
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1: Patch Embedding（可视为「视觉词嵌入层」）        │
│  图像 → 不重叠 P×P patch → flatten → Linear 投影到 D 维   │
│  224×224, P=16 → 14×14=196 个 patch token                │
│  可选：Hybrid 时先用 CNN 得到特征图再切 patch              │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2: 序列构造 + 位置编码                             │
│  prepend [CLS] token → 序列长度 197                      │
│  加可学习 1D 位置编码 pos_embed ∈ R^{(N+1)×D}            │
│  可选 Dropout                                            │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 3: Transformer Encoder（L 层，结构同 NLP）         │
│  每层: LN → MSA → 残差 → LN → MLP → 残差                  │
│  全局 self-attention：每个 patch 与所有 patch 交互         │
│  输出: 197 × D（含 [CLS] 与各 patch 的最终表示）           │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 4: 分类头（预训练 / 微调阶段不同）                  │
│  取 z_0^L（[CLS] 的最终向量）→ LayerNorm → Linear → K 类  │
│  预训练 JFT 可用 head 映射到 18k 类；微调 ImageNet 换 1k 头 │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Output: logits (batch, num_classes)  →  softmax / CE loss
```

### 1.4 ImageNet 精度参考（论文主结果，Top-1 / Top-5）

| 模型 | Patch | 预训练数据 | 微调分辨率 | Top-1 | 说明 |
|------|-------|------------|------------|-------|------|
| ViT-B/16 | 16 | JFT-300M | 384 | **77.9** | 论文最强纯 ViT 配置之一 |
| ViT-L/16 | 16 | JFT-300M | 512 | **85.2** | 更大模型 + 更高分辨率 |
| ViT-H/14 | 14 | JFT-300M | 518 | **88.55** | Huge，patch 更小、序列更长 |
| ViT-B/16 | 16 | ImageNet-21k | 384 | 84.0 | 无 JFT 时 21k 仍很强 |
| ViT-B/16 | 16 | **仅 ImageNet-1k** | 224 | ~74.5 | 小数据直接训，明显弱于 ResNet |
| ResNet152 | — | ImageNet-1k | 224 | ~79.8 | 同设定下 CNN 仍占优 |

> **读表要点**：ViT 在 **ImageNet-1k 从头训练** 时往往打不过同算力 ResNet；**换大数据预训练 + 高分辨率微调** 后优势才显现。这是理解 ViT 作为 backbone 时的第一原则。

### 1.5 快速理解：三个支柱

```
                    ┌─────────────────────────────────┐
                    │   ViT = 图像当句子、Patch 当词    │
                    └─────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
   ① Patch 序列化              ② Transformer Encoder          ③ Scale + 微调
   「怎么变成 token」           「全局关系建模」                「怎么训才强」
          │                           │                           │
   P×P 切块 + 线性投影          L 层 MSA + MLP                 大数据预训练
   196 个 patch + 1 个 CLS      每个 token 看全图               高分辨率微调
   可学习位置编码               弱局部归纳偏置                  位置编码插值
          │                           │                           │
          └───────────────────────────┴───────────────────────────┘
                                      │
                         结果：纯 attention  backbone，下游可接检测/分割头
```

| 支柱 | 一句话 | 在 ViT 中的作用 |
|------|--------|------------------|
| **Patch 序列化** | 用一块块 patch 代替 CNN 的逐像素卷积 | 定义「视觉 token」，决定序列长度与计算量 |
| **Transformer Encoder** | 全局 self-attention 混合所有 patch 信息 | 建模长程依赖；无内置平移/局部性偏置 |
| **Scale + 微调** | 预训练学通用表示，微调适配下游分辨率/类别 | 弥补归纳偏置不足；位置编码需随分辨率调整 |
| **DeiT / MAE** | 无 JFT 时让 ViT 训得动 | 见 **§9**（1k 蒸馏）、**§10**（掩码自监督） |

**心智模型（类比）**：

把 **224×224 图** 看成一篇 **196 词的文章**（每词是 16×16 像素块）；**[CLS]** 是 **摘要句**；Transformer **让每个词看到全文**；分类时 **只读摘要句** 的向量。

### 1.6 一张图的 Walkthrough（ViT-B/16，224 输入）

假设输入一张 **224×224×3** 的 ImageNet 图，类别 **「金毛犬」**（id=207）。

#### 前向（一个 iteration）

```
Step 0  输入
        x: (3, 224, 224)，ImageNet mean/std 归一化

Step 1  Patch Embedding
        Reshape + 重排: (3,224,224) → (196, 768)
        每个 patch: 16×16×3=768 维 raw → Linear(768→768) → 768-d token
        （实现上常用 Conv2d(kernel=P, stride=P) 一次完成切分+投影）

Step 2  加 [CLS] 与位置编码
        x_0 = [CLS_emb; patch_emb_1; ...; patch_emb_196] + pos_embed
        形状: (197, 768)

Step 3  Encoder ×12 层
        每层所有 197 token 做 self-attention
        浅层: 局部纹理、边缘类模式开始混合
        深层: 更全局的语义组合
        输出 z_L: (197, 768)

Step 4  分类头
        y = LayerNorm(z_L[0])           # 只取 [CLS]
        logits = Linear(768 → 1000)
        pred = argmax(logits)

Step 5  损失（训练）
        L = CrossEntropy(logits, label=207)
        反传更新 patch embed、pos embed、12 层 Encoder、分类头
```

#### 与 CNN（ResNet）对比的「感受野」直觉

```
ResNet-50:
  浅层 3×3 卷积 → 天然局部、平移等变
  深层感受野逐步扩大到全图

ViT-B/16:
  第 1 层 MSA 后 → 每个 patch token 已「看过」全部 196 patch
  全局感受野 = 1 层；但有效语义需多层堆叠才稳定
```

### 1.7 常见困惑 FAQ

| 问题 | 简短回答 |
|------|----------|
| **为什么叫 16×16 Words？** | Patch 大小 P=16 时，224 图被切成 14×14=**196** 个 patch；类比 NLP 里 196 个 word token。 |
| **ViT 完全没有卷积吗？** | **Patch Embedding** 可用 `Conv2d(P,P)` 实现（等价于分块+Linear）；**Hybrid** 变体前面还有 CNN。纯 ViT 主体无 depthwise/局部卷积栈。 |
| **为什么用 [CLS] 而不是全局平均池化？** | 沿用 BERT 设计；[CLS] 是可学习「聚合槽位」，通过 attention 从所有 patch 抽取信息。实验上 [CLS] 与 GAP 接近，[CLS] 为默认。 |
| **位置编码为什么用 1D 可学习向量？** | 简单有效；patch 展平为序列后，每个位置一个 embedding。2D 相对/正弦编码也可，论文主配置为 **learnable absolute PE**。 |
| **微调分辨率变大怎么办？** | **位置编码双线性插值**：训练 224→14×14 grid；微调 384→24×24 grid 时对 pos embed 做 2D 插值，[CLS] 位置不变。 |
| **小数据集能训 ViT 吗？** | 直接训很难。可选：**§9 DeiT**（1k + 蒸馏）、**§10 MAE**（无标签预训练 + 微调）、或 **ImageNet-21k/JFT** 监督预训练再微调。 |
| **ViT 当检测 backbone 怎么用？** | 取 **最后一层 patch token** reshape 成特征图（如 14×14×768），接 FPN/检测头；或取中间层多尺度（如 ViTDet）。DETR 系最初用 ResNet，后续大量工作换 ViT/Swin。 |
| **计算瓶颈在哪？** | **Self-Attention 复杂度 O(N²)**，N=patch 数。P=16、224 图 N=196 尚可；P=14、518 图 N≈37×37 代价很高。 |
| **B/16、L/16、H/14 命名含义？** | **B/L/H**=Base/Large/Huge（深度、宽度）；**16/14**=patch 边长 P。P 越小 → token 越多 → 越细、越贵。 |

### 1.8 文档阅读路线

| 你的目标 | 建议阅读 |
|----------|----------|
| **5 分钟懂原理** | §1.5 ~ §1.9 + §1.3 流水线 |
| **搞清网络结构** | §2 Patch Embed → §3 Encoder Block → §4 分类头 → §5 维度推导 |
| **搞清训练怎么跑** | §6 预训练 → §7 微调与位置插值 → §8 数据增强 |
| **1k 数据如何训 ViT** | **§9 DeiT 蒸馏**（有标签 + teacher） |
| **无标签如何预训练 ViT** | **§10 MAE 掩码预训练**（自监督 + 微调） |
| **搞清为何要大数据** | §1.9.4 归纳偏置 + §6.3 数据规模实验 |
| **三种训练路线怎么选** | §9.8 + §10.9 对照表 |
| **当 backbone 接下游** | §11 Hybrid 与下游用法 → §12 与 CNN 对比 |
| **读官方/常用实现** | §13 代码对照 |
| **知道后续怎么演进** | §14 相关工作 |

### 1.9 核心原理深讲

#### 1.9.1 图像分类 = Patch 序列建模

传统 CNN 隐含假设：**邻近像素强相关**、**特征应平移等变**、**层次化从局部到全局**。  
ViT 换了一个问法：

```
输入:  图像 I ∈ R^{H×W×C}
输出:  类别 y ∈ {1, ..., K}

中间表示:  token 序列  Z ∈ R^{N×D}，N = HW/P²

每个 token 对应一个 P×P patch 的嵌入向量，而非每个像素一个 activation map
```

**关键约束**：序列长度 N 固定（同分辨率同 P），与物体在图中的位置无关——**位置信息全靠 position embedding + attention 学出来**。

```
         224×224 图像                    196 个 patch token（示意）
    ┌──┬──┬──┬── ... ──┐              t_1  t_2  t_3 ... t_196
    │  │  │  │         │    Patch      ↓    ↓    ↓       ↓
    ├──┼──┼──┼── ... ──┤    Embed  →  [CLS, t_1, t_2, ..., t_196] + pos
    │  │  │  │         │              └─────── Transformer ───────┘
    └──┴──┴──┴── ... ──┘                              ↓
                                                    [CLS] → 分类
```

#### 1.9.2 信息如何在网络中流动（四段式）

```
阶段          输入是什么                    输出是什么                   这一步「学会」什么
─────────────────────────────────────────────────────────────────────────────────────────
① Patch Embed 像素块 P×P×C                  N 个 D 维向量                局部 RGB 模式 → 语义嵌入
② +PE + CLS   N 个 patch 向量               (N+1) 个 token               「第 i 块在图的哪」
③ Encoder     (N+1) 个 token                同形状，深层表示              patch 间全局关系、物体部件
④ Head        [CLS] 向量 z_0^L              K 维 logits                  全图语义 → 类别
```

**直觉**：Patch Embed 把 **局部视觉词** 建好；Encoder 做 **全文阅读理解**；[CLS] 是 **读后问答题的标准答案槽位**。

#### 1.9.3 Self-Attention：一个 patch 如何「看见」全图

以 **第 58 号 patch**（大致在图像中部）为例：

```
输入序列 z ∈ R^{197×768}

单层 Multi-Head Self-Attention（简化为单头）:
  Q = z W_Q,  K = z W_K,  V = z W_V

  对 token 58:
    score_j = (Q_58 · K_j) / √d_head     j = 0..196（0 是 [CLS]）
    α_j = softmax(score_j)

    output_58 = Σ_j α_j · V_j
```

```
197 个 token（含 CLS）          patch#58 的 attention 权重（示意）
┌─────────────────┐          ┌─────────────────┐
│  CLS  p1  p2 ...│          │ 0.05 0.01 0.02..│  ← 也可能关注 CLS
│  ...  p58 ...   │   →      │ ...  0.12 0.08..│  ← 同类区域、关键部件
│  ...  p196      │          │ ...  0.03 0.01  │
└─────────────────┘          └─────────────────┘
                                      │
                                      ▼
                            output_58 聚合全图相关信息
```

**与 CNN 的区别**：一层 MSA 后 **每个 patch 已接触全图**；但 **有效识别** 仍要多层 MLP+MSA 组合，且需要 **大量数据** 学会「该关注哪里」——没有 CNN 那种硬编码的局部连接。

#### 1.9.4 归纳偏置：ViT 缺什么、大数据补什么

| 归纳偏置 | CNN | ViT（纯） |
|----------|-----|-----------|
| **局部性** | 3×3 卷积只看邻域 | 一层就看全局；局部需多层或数据学 |
| **平移等变** | 权重共享 + 卷积 | patch 位置靠 PE；非严格等变 |
| **层次结构** | 池化/步长自然形成 pyramid | 同分辨率深堆；层次在特征语义里隐式学 |

```
ImageNet-1k 仅预训练:
  ResNet 利用偏置 → 样本效率高 → 更快收敛、更高精度

JFT-300M 预训练:
  ViT 容量大、scaling 好 → 大数据下反超
  归纳偏置少 → 更少「错误先验」，更灵活
```

论文 Figure 2 经典结论：**ViT-H 在足够大数据上验证误差随计算量下降比 ResNet 更陡**——这是 ViT 成为通用 vision backbone 的理论支点。

#### 1.9.5 [CLS] token：聚合机制

```
z_0^0 = x_class  （可学习向量，与图像无关，随 batch 广播）

经 L 层 Encoder 后 z_0^L 的更新路径:
  每一层 self-attention 中，[CLS] 作为 query 可以 attend 到所有 patch
  同时所有 patch 也可以 attend 到 [CLS]

分类:  ŷ = MLP(LayerNorm(z_0^L))
```

**为何不直接用 GAP(patch tokens)?**  
GAP 对所有 patch 平等平均；[CLS] 是 **可学习的查询向量**，理论上可以学会 **加权聚合**（类似 attention pooling）。论文报告两者性能接近；社区默认保留 [CLS] 以与 BERT/JFT 预训练权重兼容。

#### 1.9.6 位置编码与分辨率迁移

预训练 **224×224，P=16** → grid **14×14**，pos_embed 形状 `(1, 197, D)`：

```
pos_embed[0]     → [CLS] 的位置向量
pos_embed[1:197] → 14×14 patch 网格，按 row-major 展平
```

微调 **384×384** → grid **24×24**，需要 **25 个新位置**（含 CLS 仍 1 个）：

```
做法（论文 Appendix）:
  1. 把 pos_embed[1:] reshape 成 (14, 14, D)
  2. 双线性插值到 (24, 24, D)
  3. flatten 回 (576, D)，与 [CLS] pos 拼接 → (577, D)
  4. 通常 pos_embed 可微调或固定；分类头换/改 K 类
```

**这是 ViT 工程上最重要的 trick 之一**：同一套 JFT 预训练权重可在 **不同输入分辨率** 下微调，无需从头训练位置关系。

#### 1.9.7 三大机制如何协同（总图）

```
                    ┌──────────────────────────────────────┐
                    │         一张 224×224 训练图           │
                    └──────────────────────────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
   Patch + PE                    Transformer                   预训练/微调
   「token 化 + 在哪」            「全局混合」                   「scale 策略」
         │                            │                            │
   Conv P×P → N tokens           L×(MSA+MLP)                  JFT/21k 预训练
   + learnable pos               197 tokens 互看               384/512 微调
   + [CLS]                       [CLS] 读出语义                 pos 插值
         │                            │                            │
         └────────────────────────────┼────────────────────────────┘
                                      ▼
                              CE loss on [CLS] logits
                                      │
                                      ▼
                         梯度更新 embed / encoder / head
```

**一句话串起来**：Patch 把图变成 **句子**；Transformer 做 **全局语法/语义组合**；大规模预训练 + 分辨率微调 让 **弱偏置的大模型** 在下游任务上超过 **强偏置的 CNN**。

---

## 2. Stage 1：Patch Embedding

Patch Embedding 是 ViT 的「**stem**」，替代 ResNet 的前几层 conv。

### 2.1 从像素到 patch 序列

设输入 `x ∈ R^{H×W×C}`，patch 边长 **P**，则：

```
patch 个数:  N = (H/P) × (W/P)
每个 patch 展平:  x_p ∈ R^{P²·C}
嵌入:           z_p = x_p E + b_E，E ∈ R^{(P²C)×D}
```

**ViT-B/16，224×224×3**：

```
P = 16,  N = 14×14 = 196
P²·C = 256×3 = 768
D = 768（hidden size）
每个 patch → 768-d 向量
```

### 2.2 实现：Conv2d 等价形式（官方常用）

```python
# 概念等价：nn.Conv2d(in_ch=3, out_ch=D, kernel_size=P, stride=P)
# 输入: (B, 3, 224, 224)
# 输出: (B, D, 14, 14) → flatten → (B, 196, D)
```

| 实现方式 | 说明 |
|----------|------|
| **Conv2d(P, stride=P)** | 一次完成「切块 + 线性投影」，GPU 友好；**timm / JAX 官方默认** |
| **unfold + Linear** | 逻辑更清晰；大 P 时与 Conv 等价 |
| **Hybrid 前接 CNN** | 如 ResNet stage4 输出 14×14×1024，再 Conv 1×1 → D |

### 2.3 Patch 大小 P 的权衡

| P | N (224 图) | 序列长度 | 计算 | 语义粒度 |
|---|------------|----------|------|----------|
| **32** | 7×7=49 | 短 | 便宜 | 粗，细节少 |
| **16** | 14×14=196 | 中 | 标准 | **论文默认** |
| **14** | 16×16=256 | 较长 | 较贵 | ViT-H/14 用 |
| **8** | 28×28=784 | 很长 | 很贵 |  rarely 用于 224 分类 |

**Self-Attention FLOPs 近似 ∝ N²·D** → P 减半 → N 约 4 倍 →  attention 约 **16 倍** 开销（同 H,W）。

---

## 3. Stage 2~3：序列构造与 Transformer Encoder

### 3.1 序列构造

```
x_patch = PatchEmbed(image)           # (B, N, D)
cls_token = expand(class_embedding)   # (B, 1, D)
x = concat([cls_token, x_patch], dim=1)   # (B, N+1, D)
x = x + pos_embed                       # pos_embed: (1, N+1, D) 可学习
x = Dropout(x)
```

- **pos_embed**：随机初始化，与 patch embed 一起训练；**不含** 显式 sin/cos（主配置）。
- **Dropout**：标准，rate 见 §8。

### 3.2 Encoder Block 结构（Pre-LN）

论文 Section 3.2：**Layer Norm 放在每个子块之前**（Pre-LN），残差连接在子块之后：

```
z' = z + MSA(LN(z))
z'' = z' + MLP(LN(z'))
```

**单层 Block 展开**：

```
输入 z: (B, N+1, D)

── Multi-Head Self-Attention ──
  z_norm = LayerNorm(z)
  Q,K,V = Linear(z_norm)  →  reshape → H 个头
  Attention(Q,K,V) = softmax(QK^T / √d_h) V
  MSA_out = Concat(heads) W_O

  z' = z + Dropout(MSA_out)

── MLP（FFN）──
  z_norm' = LayerNorm(z')
  MLP(x) = Linear(D → D_ff) → GELU → Dropout → Linear(D_ff → D)
  z'' = z' + Dropout(MLP(z_norm'))

输出 z'': (B, N+1, D)
```

| 组件 | ViT-B | ViT-L | ViT-H |
|------|-------|-------|-------|
| 层数 L | 12 | 24 | 32 |
| D (hidden) | 768 | 1024 | 1280 |
| Heads H | 12 | 16 | 16 |
| d_h = D/H | 64 | 64 | 80 |
| D_ff (MLP 中间维) | 3072 | 4096 | 5120 |
| 激活 | GELU | GELU | GELU |

### 3.3 Multi-Head Attention 在视觉序列上的含义

```
Head h 可能学到不同模式（非论文显式监督，事后可视化常见）:
  - 某些 head：邻近 patch 高权重（近似局部）
  - 某些 head：远距离 patch（对称部件、上下文）
  - 某些 head：[CLS] 与物体相关 patch 强连接
```

**197×197 attention matrix** 每层每头各一份 → 可视化是 ViT 可解释性常见手段（Attention Rollout 等）。

### 3.4 Encoder 输出

```
z_L = Encoder^L(...Encoder^1(x))
z_L[0]   → [CLS] 最终表示，送分类头
z_L[1:]  → patch 表示，可用于分割/检测 dense 预测
```

---

## 4. Stage 4：分类头（Representation Head）

### 4.1 微调阶段（ImageNet-1k）

```
y = LayerNorm(z_L[0])      # 有些实现省略额外 LN，直接用 z_L[0]
logits = Linear(D → K)       # K=1000
```

微调时常 **替换** JFT 预训练的 18k 类 head，或 **重初始化** 最后一层；**Encoder + patch embed + pos embed** 加载预训练。

### 4.2 预训练阶段（JFT-300M）

```
K_pre ≈ 18,291 类（JFT 标签空间）
同样取 [CLS] → Linear(D → K_pre)
```

### 4.3 蒸馏头（ViT 原文 Section 4.2，可选）

大模型或 CNN **teacher** 生成 soft label，student ViT 在 **[CLS] logits** 上加 **distillation loss**：

```
L = (1-α) · CE(y, label) + α · KL(softmax(z/τ), softmax(teacher/τ))
```

ViT 原文仅在 JFT 实验里验证蒸馏有效；**DeiT** 在此基础上把蒸馏做成 **ImageNet-1k 训 ViT 的核心配方**（独立 `[DIST]` token、完整训练策略），详见 **§9**。

---

## 5. 维度与计算量推导（ViT-B/16 @ 224）

### 5.1 张量形状追踪

```
(B, 3, 224, 224)
    → PatchEmbed → (B, 196, 768)
    → +[CLS]     → (B, 197, 768)
    → Encoder×12 → (B, 197, 768)
    → head       → (B, 1000)
```

### 5.2 单层 Block 参数量（量级）

```
MSA:
  W_Q, W_K, W_V, W_O: 各 D×D  → 4D² ≈ 4×768² ≈ 2.36M

MLP:
  D×D_ff + D_ff×D ≈ 2×768×3072 ≈ 4.72M

LayerNorm + 偏置: 相对可忽略

单层合计 ≈ 7M 参数
12 层 ≈ 85M（与论文 ViT-B 总参 ~86M 一致，含 embed + head）
```

### 5.3 Attention FLOPs（单层、单样本）

```
N = 197, D = 768, H = 12, d_h = 64

QK^T:  H × (N × d_h × N)  ≈ O(N²·D)
AV:    同上量级
FFN:   O(N·D·D_ff)  → 通常与 attention 同阶或更大取决于 N
```

**N=196 时 ViT-B 与 ResNet-50 FLOPs 同量级**；分辨率或 P 变小导致 N 增大时 ViT 更快变慢于 CNN。

---

## 6. 预训练：数据、目标与 schedule

### 6.1 预训练数据

| 数据集 | 规模 | 论文用途 |
|--------|------|----------|
| **JFT-300M** | ~300M 图，18k 类 | 主预训练，最强结果 |
| **ImageNet-21k** | ~14M 图，21k 类 | 公开可复现的强预训练 |
| **ImageNet-1k** | 1.28M，1k 类 | 小数据 baseline，ViT 偏弱 |

### 6.2 预训练超参（典型，论文 Table 3 / Appendix）

```
优化器: Adam
β1=0.9, β2=0.999
batch size: 4096（大 batch 关键）
学习率: 线性 warmup + cosine decay
  peak lr 约 3e-3（随 batch 缩放）
weight decay: 0.1（AdamW 式 decoupled WD）
label smoothing: 0.1
训练步数: 300k steps（JFT）
分辨率: 224 或 384
```

### 6.3 为何 ViT 依赖大数据（论文 Figure 3）

```
设定: 从 ImageNet-1k 子采样 {10%, 20%, ..., 100%}

观察:
  - 100% 数据: ViT-L 仍略逊于 BiT ResNet
  - JFT 预训练后: ViT-L 明显优于 BiT

解释:
  小数据 → MSA 容易过拟合全局关系，缺 CNN 正则
  大数据 → 大容量 ViT 学到更通用表示，scaling 优势体现

后续两条公开可复现路线（无需 JFT）:
  **DeiT（§9）**: ImageNet-1k + CNN 蒸馏，一阶段训强 ViT
  **MAE（§10）**: ImageNet-1k 无标签掩码预训练 + 微调，表示质量更高
```

---

## 7. 微调：分辨率、head 与正则

### 7.1 标准 ImageNet 微调流程

```
1. 加载预训练权重（patch embed, pos embed, encoder）
2. pos_embed 若分辨率变化 → 2D 插值（§1.9.6）
3. 分类头 Random init（K=1000）
4. 更高分辨率训练（如 384），更小 batch，更长 fine-tune
5. 常用 **较小 lr**（如 1e-4 ~ 3e-4）、cosine decay、early stopping 按 val
```

### 7.2 层-wise learning rate decay（社区常见，非原文必选项）

下游检测/分类微调 ViT backbone 时，常对 **浅层 patch embed 用小 lr、深层用大 lr** 或逐层 decay，稳定微调。**timm** 等库内置 `layer_decay`。

### 7.3 微调 vs 线性 probe

| 方式 | 做法 | 用途 |
|------|------|------|
| **Linear probe** | 冻结 Encoder，只训 head | 衡量表示质量 |
| **Full fine-tune** | 全网微调 | 下游 SOTA |
| **Partial freeze** | 冻结前 k 层 | 小数据防过拟合 |

论文报告：**大数据预训练 + full fine-tune** 是 ViT 最强组合。

---

## 8. 训练时的输入处理与增强

### 8.1 预处理

```
训练 / 评估（ImageNet）:
  1. Resize: 短边缩放到 S（224 预训练；微调 384 等）
  2. RandomCrop(S) 或 CenterCrop(S)  — 训练用 Random
  3. RandomHorizontalFlip
  4. 归一化: mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]
     （论文用 [-1,1] 缩放；实现细节因库而异，timm 常用 ImageNet 标准 mean/std）
```

### 8.2 正则化（与 CNN 训练的关键差异）

| 技术 | ViT 中的用法 | 原因 |
|------|--------------|------|
| **Weight Decay** | 0.1（较大） | 大模型防过拟合 |
| **Dropout** | attention + MLP 中 0.1 | 小数据尤其重要 |
| **Stochastic Depth** | 训练时随机 drop 整层 | 论文/后续 DeiT 常用 |
| **Mixup / CutMix** | 微调 ImageNet 广泛使用 | 弥补 ViT 数据效率 |
| **Label Smoothing** | 0.1 | JFT 预训练默认 |
| **Repeated Aug** | 同一图多次增强不同 crop | 大 batch 训练稳定 |

### 8.3 一个 training step 样本定义

```
一个 sample = 一张图 + 单标签（单标签分类）

与检测不同:
  - 无 anchor / 无 region proposal
  - 整张图 → 一个 [CLS] logits → 一个 CE loss
  - batch 内 B 张图独立，无跨图 attention（标准 ViT）
```

---

## 9. DeiT：ImageNet-1k 上的 ViT 训练配方（知识蒸馏）

> **定位**：ViT 原文解决「Transformer 能否做视觉分类」；DeiT 解决「**没有 JFT，只有 ImageNet-1k，ViT 怎么训到 ResNet 同级**」。  
> 论文：Touvron et al., *Training data-efficient image transformers & distillation through attention*, ICML 2021. [arXiv:2012.12877](https://arxiv.org/abs/2012.12877)  
> 代码：[facebookresearch/deit](https://github.com/facebookresearch/deit)

### 9.1 要解决的问题

```
ViT 原文在 ImageNet-1k 从头监督训练:
  ViT-B/16  ~74.5% Top-1  <<  ResNet-152 ~79.8%

原因（与 §1.9.4 一致）:
  ViT 几乎无卷积式归纳偏置 → 需要更多数据「自己学」局部性与层次性
  1.28M 标注图不足以支撑 ~86M 参数的 ViT 从头收敛

DeiT 的策略:
  ① 不改 ViT 主体（仍是 Patch + Encoder + [CLS]）
  ② 加 **知识蒸馏**：让强 CNN teacher 把「暗知识」传给 ViT student
  ③ 加 **更强数据增强与正则**（与蒸馏配套）
  → 无需 JFT，仅 ImageNet-1k 即可训强 ViT
```

### 9.2 相对 ViT 的结构改动：Distillation Token

DeiT **不**改 Transformer Block 内部，只改 **输入序列** 和 **损失**：

```
ViT 序列:
  [CLS, patch_1, ..., patch_196] + pos_embed     长度 197

DeiT 序列:
  [CLS, DIST, patch_1, ..., patch_196] + pos_embed   长度 198
         ↑
    新增可学习 token，专用于接收 teacher 的 soft label
```

| Token | 读出方式 | 监督信号 |
|-------|----------|----------|
| **[CLS]** | `head_cls(LN(z[0]))` | 硬标签 **Cross-Entropy**（真实类别） |
| **[DIST]** | `head_dist(LN(z[1]))` | **KL 蒸馏**（对齐 teacher soft logits） |
| **patch** | 无独立 head | 仅通过 self-attention 辅助表示学习 |

**为何单独 [DIST]，而不是像 ViT 原文那样只蒸馏 [CLS]？**

```
若 [CLS] 同时做 CE + 蒸馏:
  两类目标可能冲突（硬标签要求「这是狗」，teacher 说「70% 狗 20% 狼」）
  梯度在同一 token 上打架 → 收敛慢

[CLS] 专注硬分类，[DIST] 专注模仿 teacher
  → 分工明确，论文称为 "distillation through attention"
  → self-attention 让 [DIST] 与 patch 交互，间接改善全局表示
```

推理时：DeiT 通常 **只用 [CLS] 分支** 输出；[DIST] 仅在训练中存在（或两分支 logits 平均，实现可选）。

### 9.3 Teacher–Student 蒸馏流程

```
                    同一张训练图 x
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
   Teacher（冻结）                    Student DeiT（训练）
   RegNetY-16GF 等 CNN                ViT-T/S/B
          │                               │
          ▼                               ▼
   logits_teacher ∈ R^1000          z_cls, z_dist
   softmax(·/τ)                     head_cls → pred_cls
                                    head_dist → pred_dist
          │                               │
          └─────────── KL 对齐 ───────────┘
                    CE 对齐 GT label → pred_cls
```

**Teacher 选择（论文）**：

| Teacher | Top-1 | 说明 |
|---------|-------|------|
| RegNetY-16GF | 82.9% | 默认 teacher，CNN 强、与 ViT 结构差异大，蒸馏收益明显 |
| EfficientNet-B4 | 83.0% | 亦可 |

Teacher **不参与反传**，推理开销仅在训练阶段（可离线缓存 soft label，但官方实现多为 online teacher）。

### 9.4 损失函数（逐项）

```
L = L_cls + λ · L_distill

L_cls = CE( head_cls(z[CLS]), y_true )

L_distill = τ² · KL( log_softmax(pred_dist/τ), softmax(teacher/τ) )
            （对 [DIST] 输出与 teacher 做 KL；τ 为温度）

推理: ŷ = argmax head_cls(z[CLS])     # 不用 [DIST]
```

| 超参 | 典型值 | 含义 |
|------|--------|------|
| **τ（temperature）** | 1.0 或 3~4 | 越大 soft label 越平滑，类间关系信号越强 |
| **λ（distill 权重）** | 0.5 ~ 1.0 | 蒸馏项占比；DeiT 默认与 CE 同量级 |
| **label smoothing** | 0.1 | 仍用于 CE 分支 |

**蒸馏在学什么（直觉）**：

```
硬标签:  「这是 class 207（金毛）」→ one-hot

Teacher soft: 「207: 0.85, 208: 0.08, 254: 0.03, ...」
  → 207 与 208（相近犬种）有相似度
  → ViT 从 [DIST] 学到 **类间结构**，比 one-hot 信息 richer
  → 等价于 CNN 的归纳偏置以「软监督」形式注入 ViT
```

### 9.5 训练增强与正则（与 ViT/JFT 的差异）

DeiT 的成功 **不只有蒸馏**，而是一整套 **data-efficient recipe**：

| 技术 | DeiT 用法 | 作用 |
|------|-----------|------|
| **Rand-Augment** | 默认开启 | 强增强，扩增有效数据 |
| **Mixup** | α=0.8 | 混合样本，平滑决策边界 |
| **CutMix** | α=1.0 | 裁剪粘贴，强迫看局部 |
| **Random Erasing** | 0.25 | 随机擦除，防过拟合 |
| **Stochastic Depth** | 0.1~0.4（随模型变大） | 随机 drop 层，正则 |
| **AdamW** | lr 5e-4, wd 0.05 | 与 ViT JFT 的 wd=0.1 略有不同 |
| **Cosine schedule** | 300 epoch | 长 schedule 利于 ViT 收敛 |
| **Repeated Aug** | 每 epoch 多视图 | 等效增大 batch 多样性 |

```
对比 ViT 原文 ImageNet-1k 从头训:
  ViT:  相对标准增强 + 大 batch 预训练套路移植 → 易过拟合
  DeiT: 强增强 + 蒸馏 + DropPath → 1k 上可收敛
```

### 9.6 ImageNet-1k 精度（DeiT 主结果）

| 模型 | 参数量 | FLOPs | Top-1 | 对比 |
|------|--------|-------|-------|------|
| DeiT-Ti | 5M | 1.3G | **72.2** | 小模型超 ResNet-18 级 |
| DeiT-S | 22M | 4.6G | **79.8** | 接近 RegNetY-16GF teacher |
| DeiT-B | 86M | 17.5G | **81.8** | **无 JFT**，超 ViT-B 1k 从头 ~7 点 |
| ViT-B/16（1k 从头） | 86M | 17.6G | ~74.5 | 缺蒸馏与 recipe |

> DeiT-B **81.8%** 仍低于 ViT-B **JFT 预训练 + 384 微调（~77.9 是 JFT+384；21k 预训练 84.0）**——DeiT 换的是 **数据效率**，不是 beat 一切 JFT 上限。

### 9.7 一个 Training Step Walkthrough（DeiT-S）

```
Step 0  输入
        图像 x（224×224），标签 y=207（金毛）
        Mixup 可能已混合另一张图 → 软标签 y'

Step 1  Teacher 前向（no_grad）
        teacher(x) → logits_t ∈ R^1000

Step 2  Student 前向
        patch embed → [CLS, DIST, patches] + pos
        12 层 Encoder → z[CLS], z[DIST]

Step 3  双头输出
        pred_cls = head_cls(z[CLS])
        pred_dist = head_dist(z[DIST])

Step 4  损失
        L_cls = CE(pred_cls, y')
        L_dist = KL(pred_dist, softmax(teacher(x)/τ))
        L = L_cls + λ L_dist

Step 5  反传
        只更新 student；teacher 冻结
        梯度经 [DIST] 与 [CLS] 的 self-attn 回传到 patch token 与 Encoder
```

### 9.8 DeiT vs ViT 原文训练：对照总表

| 维度 | ViT（1k 从头） | DeiT |
|------|----------------|------|
| **预训练数据** | 仅 ImageNet-1k | 仅 ImageNet-1k |
| **序列 token** | [CLS] + 196 patch | **[CLS] + [DIST] + 196 patch** |
| **监督** | CE only | **CE + KL 蒸馏** |
| **Teacher** | 无 | **RegNetY-16GF（冻结）** |
| **增强** | 标准 | **RandAug + Mixup + CutMix + ER** |
| **DeiT-B Top-1** | ~74.5 | **81.8** |
| **核心贡献** | 架构 | **训练 recipe + distillation token** |

### 9.9 代码与 timm 权重

```python
import timm

# DeiT 结构与 ViT 相同接口，权重不同（1k 蒸馏训练）
model = timm.create_model('deit_base_patch16_224', pretrained=True)
# DeiT-S / deit_tiny_patch16_224 / deit_base_distilled_patch16_384 等

logits = model(x)   # 推理默认走 [CLS] 头
```

| timm 名称 | 对应论文 |
|-----------|----------|
| `deit_tiny_patch16_224` | DeiT-Ti |
| `deit_small_patch16_224` | DeiT-S |
| `deit_base_patch16_224` | DeiT-B |

官方仓库 `models.py` 中 `DistilledVisionTransformer` 实现 `[CLS]` / `[DIST]` 双 token 与双 head。

---

## 10. MAE：ViT 的自监督预训练（掩码自编码）

> **定位**：DeiT 仍需要 **1000 类标注**；MAE 用 **无标签** 图像做预训练，再微调分类——把 ViT 推向 BERT 式「先预训练、后微调」范式。  
> 论文：He et al., *Masked Autoencoders Are Scalable Vision Learners*, CVPR 2022. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)  
> 代码：[facebookresearch/mae](https://github.com/facebookresearch/mae)

### 10.1 要解决的问题

```
ViT 监督预训练的瓶颈:
  ① 强依赖 JFT 等超大规模 **标注** 数据（标注成本高）
  ② ImageNet-1k 从头训 ViT 表示质量差（Even DeiT 也要 300 epoch 强 recipe）

MAE 的思路（类比 BERT）:
  ① 随机 **遮住大部分 patch**
  ② 让模型 **重建被遮住的像素**
  ③ 预训练 Encoder 学会视觉语义 → 微调时换分类头即可

关键设计: **非对称** Encoder/Decoder
  Encoder 只看 **少量可见 patch**（省算力）
  Decoder 轻量，负责 **填空**
```

### 10.2 整体架构（预训练阶段）

```
Input Image (B, 3, H, W)   224×224, P=16 → N=196 patch
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Step A: Patch Embed + 位置编码（与 ViT 相同）            │
│  196 个 patch token，每个 D=768                          │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Step B: Random Mask（论文默认 mask ratio = 75%）         │
│  保留 ~25% 可见 patch（约 49 个）                         │
│  丢弃被 mask 的 token（不送入 Encoder）                   │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Step C: MAE Encoder（标准 ViT Encoder，L=12）           │
│  输入: 仅可见 patch + 可选 [CLS]（MAE 默认 **不用 CLS**）  │
│  输出: 可见 token 的 latent 表示                          │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Step D: MAE Decoder（轻量，L_dec=8, D_dec=512）          │
│  输入: Encoder 输出 + mask token 占位符 + 完整 pos embed  │
│  序列恢复为 N=196 长度（可见处用 encoder 输出，mask 处可学习 mask token）│
│  输出: 每个 patch 的像素预测（P²×C 维）                   │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Loss: MSE( pred_pixels, target_pixels )  **仅在被 mask 的 patch 上计算**
```

**与 ViT 分类的最大结构差异**：

| 组件 | ViT 分类 | MAE 预训练 |
|------|----------|------------|
| **[CLS]** | 有，用于分类 | **无**（纯 patch 重建） |
| **Encoder 输入** | 全部 196 patch | **~25% 可见 patch** |
| **Decoder** | 无 | **有**（预训练专用，微调时 **丢弃**） |
| **监督** | 类别标签 | **像素重建**（自监督） |

### 10.3 掩码策略

```
对每个样本独立采样二值 mask  m ∈ {0,1}^N，N=196

mask ratio r = 0.75  →  约 147 个 patch 被 mask，49 个可见

采样方式: 均匀随机（无 block mask、无 MAE 式 structured mask）

可见 patch 集合 V，被 mask 集合 M:
  Encoder 输入长度 = |V| ≈ 49  （而非 196 → 预训练加速 ~3× 以上）
```

```
196 patch 网格示意（×=mask，o=可见）:

  o × × o × ×
  × × o × × ×
  × o × × o ×
  ...（约 25% 为 o）
```

**为何 75% 这么高？**

```
BERT 语言 mask ~15% 仍有效；图像冗余更高
论文消融: 50% mask 不如 75%；75%~80% 都很好
过高 mask → 重建任务变难 → Encoder 必须学更强语义才能「猜」缺失区域
```

### 10.4 重建目标：归一化像素

```
对每个 patch，目标为 P×P×C 像素向量

patch 内像素做 per-patch 归一化（减均值除标准差）再算 MSE:
  减少 patch 间亮度/contrast 差异带来的 trivial 解
  迫使模型学 **结构/纹理/形状** 而非全局均值

Loss = (1/|M|) Σ_{i∈M} || pred_i - norm(pixel_i) ||²
       只在 **被 mask** 的 patch 上求和
```

### 10.5 Encoder 与 Decoder 的分工

```
Encoder（ViT-B 规格，预训练后 **保留**）:
  L_enc = 12, D = 768, heads = 12
  只看可见 token → 学 **压缩语义表示**

Decoder（预训练后 **整个扔掉**）:
  L_dec = 8, D_dec = 512（比 Encoder 窄、浅）
  输入 full sequence（encoder 输出 + mask tokens 插回原序）
  Linear → P²C 重建像素

非对称的原因:
  ① Encoder 是下游唯一需要的 backbone → 投入大参数
  ② 重建是辅助任务 → Decoder 够用即可 → 省 FLOPs
  ③ 高 mask 比下 Encoder 输入已很短
```

**Decoder 如何恢复序列顺序**：

```
1. Encoder 输出 49 个 latent + 对应 pos embed
2. 对 mask 位置填入 **可学习 mask token**（共享向量）+ 该位置 pos embed
3. 按 **原始 patch 顺序** 排成 196 长序列
4. Decoder self-attention 让可见与 mask 位置交互 → 预测 mask 处像素
```

### 10.6 微调阶段：从 MAE 到分类 ViT

```
预训练结束:
  保存 Encoder 权重（patch_embed + pos_embed + 12 blocks）
  丢弃 Decoder

微调 ImageNet-1k 分类:
  1. 加载 MAE Encoder
  2. 加 [CLS] token + 分类 head（**随机初始化**）
  3. pos_embed: MAE 无 CLS → 微调时 **concat 新 CLS 的 pos**（零 init 或学习）
  4. 全图 **无 mask**，标准 ViT 前向
  5. CE loss，较小 lr，通常 **100~800 epoch**（实现常用 100~300）
  6. 可选 layer-wise lr decay、更高分辨率（384/448）
```

```
两阶段对比:

  MAE 预训练:  无标签，mask 75%，Encoder+Decoder，MSE 像素
  微调:        有标签，全 patch，仅 Encoder+[CLS]+head，CE

  下游检测/分割: 同样加载 MAE Encoder 权重初始化 backbone（ViTDet 等）
```

### 10.7 一个 Training Step Walkthrough（MAE 预训练）

```
Step 0  输入无标签图像 x: (3, 224, 224)

Step 1  Patch embed → 196 tokens × 768-d，加 pos embed

Step 2  随机 mask 75% → 保留 49 个可见 index

Step 3  Encoder（12 层）只在 49 token 上 self-attention
        输出 49 个 latent

Step 4  构造 Decoder 输入 196 长:
        可见位置 = encoder 输出；mask 位置 = mask_token + pos

Step 5  Decoder（8 层）→ Linear → 每 patch 预测 16×16×3=768 维像素

Step 6  Loss = MSE( pred, target ) 仅在 147 个 mask patch 上
        反传更新 Encoder + Decoder + mask token

（微调阶段则完全不同: 无 mask，有 [CLS]，CE，见 §10.6）
```

### 10.8 精度与 scaling（论文主结果）

| 配置 | 预训练 | 微调 | Top-1 |
|------|--------|------|-------|
| ViT-B | 无（1k 监督） | 1k 300ep | ~76 |
| ViT-B | **MAE 1600ep（1k 无标签）** | 1k 100ep | **83.6** |
| ViT-L | MAE 1k | 1k | **85.9** |
| ViT-H | MAE 1k | 1k | **87.8** |

> **83.6%（ViT-B + MAE）** 超过 DeiT-B **81.8%**，说明 **自监督预训练** 学到的表示可媲美甚至超过蒸馏监督；但 MAE 是 **两阶段**（预训练 epoch 多 + 再微调）。

### 10.9 MAE vs DeiT vs ViT 监督：三种路线对照

```
                    ┌─────────────────────────────────────────┐
                    │         ViT 怎么训才强？（三条主路）      │
                    └─────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
   ① ViT 原文监督                  ② DeiT                       ③ MAE
   JFT / 21k 标注                  1k 标注 + 蒸馏               1k 无标签预训练
          │                           │                           │
   需要私有大数据                   需要 teacher CNN              需要长预训练
   单阶段微调                      单阶段 300ep                  两阶段 pre+finetune
   ViT-H + JFT 最强               DeiT-B 81.8                   ViT-B MAE 83.6
```

| 维度 | ViT + JFT/21k | DeiT | MAE |
|------|---------------|------|-----|
| **标注需求** | 极大（JFT）或大（21k） | ImageNet-1k | **预训练无标签**；微调 1k |
| **阶段** | 预训练 + 微调 | **一阶段** end-to-end | **预训练 + 微调** |
| **核心 trick** | Scale | **[DIST] + teacher KL** | **75% mask + 非对称 dec** |
| **Encoder** | 标准 ViT | 标准 ViT + 多 1 token | 标准 ViT（预训练无 CLS） |
| **算力** | JFT 不可复现 | 1× 300ep 分类训练 | 预训练 1600ep + 微调 |
| **检测 backbone** | 常用 | 可用 | **ViTDet 等主流初始化** |

**选型建议**：

```
有 MAE / 21k 预训练权重     → 优先加载（下游最常见）
只有 1k、要一阶段训          → DeiT recipe 或 Swin（改结构）
复现 ViT 原文                → 21k 或 JFT 监督预训练
```

### 10.10 代码与 timm / 官方权重

```python
# 官方 MAE 微调后的 ViT（timm 常带 mae 或 fcmae 前缀，随版本更新）
import timm
model = timm.create_model('vit_base_patch16_224.mae', pretrained=True)

# 官方预训练 checkpoint 结构
# checkpoint['model'] 含 encoder; 不含 decoder 用于下游
```

官方 [facebookresearch/mae](https://github.com/facebookresearch/mae)：

| 脚本 | 用途 |
|------|------|
| `main_pretrain.py` | ImageNet 无标签 MAE 预训练 |
| `main_finetune.py` | 加载 MAE encoder，加 CLS head 微调 |
| `models_mae.py` | `MaskedAutoencoderViT` |

### 10.11 与 BEiT 等 MIM 方法的简要对比

| 方法 | 重建目标 | 与 MAE 关系 |
|------|----------|-------------|
| **BEiT** | 离散 visual token（dVAE tokenizer） | 重建 **code** 而非像素；需额外 tokenizer |
| **MAE** | **归一化像素** | 更简单，无 tokenizer |
| **SimMIM** | 像素（Swin 上） | 思想同 MAE，backbone 换 Swin |

MAE 的核心贡献是证明：**极简 mask + 像素 MSE + 非对称 ViT** 即可 scale 到 ViT-H。

---

## 11. Hybrid 模型与作为 Backbone 的用法

### 9.1 Hybrid ViT（CNN + Transformer）

```
Input (B, 3, 224, 224)
    │
    ▼
ResNet50（截断，无 avgpool/fc）
    → 特征图 (B, 1024, 14, 14)   # 步长 16
    │
    ▼
Conv 1×1 或 flatten + Linear → (B, 196, D)
    │
    ▼
+ [CLS] + pos_embed → Transformer Encoder → 分类
```

**动机**：在小中型数据上，CNN 提供 **局部归纳偏置**，Transformer 负责 **全局混合**——ImageNet-1k 上 Hybrid 可略优于纯 ViT。

### 9.2 下游密集预测（检测 / 分割）典型用法

ViT 原论文只做 **分类**；作为 backbone 时需 **取出 patch token 序列**：

```
Encoder 输出 z_L[1:]  →  (B, N, D)
reshape → (B, H/P, W/P, D)
permute → (B, D, H/P, W/P)   # 当作 CNN 特征图

例: 224 输入, P=16 → (B, 768, 14, 14)  单尺度特征
```

| 下游 | 常见接法 |
|------|----------|
| **DETR / Mask R-CNN** | 单尺度需 FPN 或改 multi-scale ViT（ViTDet 用简单 FPN + 中间层） |
| **Semantic Segmentation** | SETR：直接上采样 patch 特征；或 UPerNet |
| **DeiT / Swin** | 后续工作改 patch merging / window attention 降低 N² |

**与 DETR 文档衔接**：DETR 默认 ResNet；若换 ViT backbone，即 **用 ViT 替换 Stage 1 CNN**，后接 1×1 投影 + flatten + Transformer（检测用 Encoder-Decoder，分类用仅 Encoder）。

### 9.3 多尺度与 ViT 的结构性矛盾

```
纯 ViT 分类:  单一分辨率特征图 (H/P × W/P)
CNN + FPN:    天然多尺度 {P2, P3, P4, P5}

解决方向（后续论文，非 ViT 原文）:
  - 取不同 Encoder 层输出作 pyramid
  - Swin Transformer: shifted window + merge
  - ViTDet: 简单 FPN on ViT feature
  - MAE: 预训练更好表示，减轻 backbone 负担
```

---

## 12. 与 ResNet 的结构对比

```
                    ResNet-50                ViT-B/16
────────────────────────────────────────────────────────────
Stem              7×7 conv + pool          PatchEmbed P=16
Stage 1~4         残差 bottleneck×N        Transformer×12
感受野            逐层扩大                  第1层即全局
参数量            ~25M                     ~86M
归纳偏置          强（局部+平移）           弱
数据需求          1M 级可训                需 21M~300M 预训练更稳
输出              (B,2048,h,w) 或向量       (B,D) [CLS] 或 (B,D,h',w')
```

| 场景 | 更倾向 |
|------|--------|
| 小数据集、少算力 | ResNet / EfficientNet |
| 有 21k/JFT 预训练权重 | ViT |
| 要接 Transformer 检测头 | ViT / Swin（与 DETR 等同族） |
| 移动端 | ViT 通常不如轻量 CNN（除非 MobileViT 等） |

---

## 13. 代码对照（读实现时）

### 13.1 官方 JAX（google-research/vision_transformer）

| 模块 | 文件/符号 |
|------|-----------|
| Patch Embed | `models/vit.py` → `VisionTransformer` |
| Encoder Block | `Encoder1DBlock` |
| 位置插值 | `resize_pos_embedding` |
| 配置 | `model_configs.py` → `vit_b16`, `vit_l16`, `vit_h14` |

### 13.2 PyTorch timm（社区最常用）

```python
import timm
model = timm.create_model('vit_base_patch16_224', pretrained=True)
# forward: model(x) → (B, 1000)
# features: model.forward_features(x) → (B, 197, 768)
```

| timm 命名 | 含义 |
|-----------|------|
| `vit_base_patch16_224` | ViT-B, P=16, 224 预训练 |
| `vit_large_patch16_384` | ViT-L, 384 微调权重 |
| `vit_base_patch16_224.mae` | ViT-B，**MAE 预训练 + 1k 微调**（§10） |
| `deit_base_patch16_224` | DeiT-B，**蒸馏训练**（§9） |
| `deit_small_patch16_224` | DeiT-S |

### 13.3 与 PyTorch nn.Module 伪代码骨架

```python
class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, dim=768, depth=12,
                 num_heads=12, mlp_ratio=4, num_classes=1000):
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, dim * mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)                          # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[:, 0])
        return self.head(x)
```

---

## 14. 相关工作与演进

| 方向 | 代表工作 | 相对 ViT 的改进 | 详见 |
|------|----------|-----------------|------|
| **数据效率（有标签）** | DeiT (2021) | 蒸馏 + 更强增强，ImageNet-1k 可训 | **§9** |
| **自监督预训练** | MAE, BEiT (2022) | 掩码重建，降低标注需求 | **§10** |
| **局部 + 层次** | Swin (2021) | Window attention + patch merge，适合 FPN | [SwinTransformer.md](./SwinTransformer.md) |
| **检测专用** | ViTDet (2022) | 简单 FPN、中间层、常 **MAE 初始化** | — |
| **更优 PE** | RoPE, 2D sin-cos | 外推与相对位置 | — |
| **统一架构** | ViT → CLIP → SAM | 同一 Transformer backbone 多模态扩展 | — |

**读 ViT 之后建议**：

- 1k 一阶段训 ViT：**§9 DeiT**  
- 无标签预训练再微调：**§10 MAE**  
- 检测 backbone：**Swin**、[ViTDet](https://arxiv.org/abs/2111.09886)、本仓库 **DETR**  
- 效率：**EfficientViT**、**MobileViT**

---

## 15. 小结速查卡

```
┌─────────────────────────────────────────────────────────────────┐
│ ViT-B/16 @ 224 一句话                                            │
│ 196 patch + 1 CLS → 12层 Transformer → [CLS] → 1000 类          │
├─────────────────────────────────────────────────────────────────┤
│ 必记公式   N = (H/P)×(W/P)    Attention O(N²)                   │
│ 必记 trick 微调分辨率 → pos_embed 2D 插值                        │
│ 必记结论   小数据弱、大数据强；预训练是 ViT 的灵魂                  │
│ 训练三路   JFT/21k 监督 | DeiT 1k+蒸馏(§9) | MAE 无标签+微调(§10) │
│ 下游用法   z_L[1:] → (B,D,H/P,W/P) 当特征图                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 参考文献

- Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, ICLR 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.（Transformer 原架构）
- Devlin et al., *BERT*, NAACL 2019.（[CLS] token 来源）
- Touvron et al., *Training data-efficient image transformers & distillation through attention (DeiT)*, ICML 2021. [arXiv:2012.12877](https://arxiv.org/abs/2012.12877) → **§9**
- He et al., *Masked Autoencoders Are Scalable Vision Learners (MAE)*, CVPR 2022. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377) → **§10**
- Liu et al., *Swin Transformer*, ICCV 2021. → [SwinTransformer.md](./SwinTransformer.md)
