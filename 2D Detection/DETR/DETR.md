# DETR

> 本文档用于整理 DETR（End-to-End Object Detection with Transformers）论文精读笔记。  
> 重点：**核心网络结构**（CNN + Transformer Encoder-Decoder + Object Queries）、**集合预测与二分图匹配**，以及**训练/推理中的样本定义与处理方式**。  
> **想快速建立整体印象**：先读 **§1.5 ~ §1.9**，再按需深入后续章节。

---

## 1. 概览

### 1.1 基本信息

| 项目 | 内容 |
|------|------|
| 论文 | End-to-End Object Detection with Transformers |
| 作者/机构 | Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko（Facebook AI Research） |
| 发表 | ECCV 2020 |
| 任务 | 目标检测（端到端、无 anchor、无 NMS 的集合预测范式） |
| 代码 | [facebookresearch/detr](https://github.com/facebookresearch/detr) |

### 1.2 核心思想（一句话）

**把目标检测重新表述为「直接集合预测」问题：用固定数量的可学习 Object Query 通过 Transformer Decoder 与图像特征交互，输出 N 个预测；再用匈牙利算法做预测与 GT 的一一匹配，从而去掉 anchor、NMS、RoI 等手工组件。**

DETR 的关键突破不是「又一个更快的检测器」，而是证明了：

1. **Transformer 的全局自注意力** 可以替代 RPN / anchor / dense head 中的局部归纳偏置；
2. **二分图匹配损失** 让网络学会「每个 query 只负责一个物体」，天然抑制重复框；
3. **端到端可微**：从像素到 `(class, box)` 集合，一条损失链训练到底。

### 1.3 整体流水线

```
Input Image (batch, 3, H, W)     例如 H=W=800（短边 resize，长边 ≤1333）
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1: CNN Backbone（ResNet-50 / ResNet-101）        │
│  输出最后一层特征图 C=2048, 空间步长 32                   │
│  特征尺寸: H/32 × W/32 × 2048                            │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2: 1×1 Conv 投影 + 位置编码                       │
│  2048 → d_model=256；为每个空间位置加 positional encoding │
│  flatten → 序列长度 hw = (H/32)×(W/32)                   │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 3: Transformer Encoder（L=6 层）                  │
│  对 hw 个 token 做全局 self-attention + FFN              │
│  输出: 图像记忆 memory，形状 hw × d_model                 │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 4: Transformer Decoder（L=6 层）                  │
│  输入: N=100 个可学习 Object Query（与 batch 无关）       │
│  每层: self-attn(q↔q) → cross-attn(q→memory) → FFN       │
│  输出: N × d_model                                       │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 5: 预测头（共享 FFN，逐 query 独立前向）            │
│  分类: Linear → num_classes + 1（含 ∅ 空类）             │
│  回归: MLP(3层) → 4 维框 (cx, cy, w, h)，sigmoid 归一化   │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Output: 100 组 (class_logits, bbox)  →  训练时匈牙利匹配 GT
                                      →  推理时取 score 阈值，无需 NMS
```

### 1.4 COCO 精度参考（论文主结果）

| 配置 | Backbone | AP | AP50 | AP75 | 说明 |
|------|----------|-----|------|------|------|
| DETR | R50 | 42.0 | 62.4 | 44.2 | 300 epoch，标准 COCO train |
| DETR | R101 | 43.5 | 63.8 | 46.4 | 更深 backbone |
| DETR-DC5 | R50 | 43.3 | 63.1 | 45.9 | dilation 提高特征分辨率（步长 16） |

> DETR 在大物体（AP_L）上表现较好，小物体（AP_S）相对较弱——与特征步长 32、query 数有限有关，也是 Deformable DETR 等后续工作的主要动机。

### 1.5 快速理解：三个支柱

DETR 的整体设计可以压缩成 **三根柱子**。先记住它们，再读细节会轻松很多：

```
                    ┌─────────────────────────────────┐
                    │   DETR = 集合预测式目标检测      │
                    └─────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
   ① Object Query              ② Transformer                ③ 匈牙利匹配
   「100 个检测槽位」           「全局看全图、动态定位」        「谁对应谁，训练说了算」
          │                           │                           │
   固定 N=100 个可学习向量      Encoder: 图像 token 互相关     预测集合 ↔ GT 集合
   每个槽位最多输出 1 物体      Decoder: query 读 memory       最优一一配对
   用 ∅ 类表示「空槽位」        Cross-Attn = 定位机制          未配对 → 监督为 ∅
          │                           │                           │
          └───────────────────────────┴───────────────────────────┘
                                      │
                         结果：去掉 anchor、启发式正负样本、NMS
```

| 支柱 | 一句话 | 在 DETR 中的作用 |
|------|--------|------------------|
| **Object Query** | 用 100 个可学习「空位」等物体来填 | 固定容量的检测槽位，每个最多对应一个 GT |
| **Transformer** | Query 通过 attention 在全图找目标 | Encoder 建全局上下文，Decoder cross-attn 完成定位 |
| **匈牙利匹配** | 训练时动态决定哪个 query 负责哪个 GT | 集合级一对一监督，未配对槽位学习输出 ∅ |

**心智模型（类比）**：

给模型 **100 个固定座位**；每张图最多来 100 个物体「入座」；**Transformer 负责找座**，**匈牙利算法负责对号入座**，空座位输出「无人（∅）」。

### 1.6 一张图的 Walkthrough（3 个 GT 例子）

假设某张 COCO 图经 resize 后有 **3 个 GT**：`{猫, 狗, 车}`，DETR 固定输出 **100 个预测**。

#### 训练时（一个 iteration）

```
Step 0  输入
        图像 (3, 800, 1066)  +  GT = 3 组 (class, box)

Step 1  前向
        ResNet → 特征图 25×33
        Encoder → memory（825 个空间 token，每个 token 「看过」全图）
        Decoder → 100 个 query 向量，每个已通过 cross-attn 从 memory 聚合信息
        预测头  → 100 × (81 类 logits, 4 维 box)

Step 2  匈牙利匹配（no_grad，不参与反传）
        构造 100×3 代价矩阵 C[j,i] = 分类代价 + L1框 + GIoU
        例: query#7  对「猫」代价 0.4  → 最低
             query#23 对「狗」代价 0.5
             query#61 对「车」代价 0.3
        匈牙利输出: 7↔猫, 23↔狗, 61↔车
        其余 97 个 query → 监督标签为 ∅

Step 3  算损失并反传
        query#7:  CE→猫,  L1+GIoU→猫框
        query#23: CE→狗,  L1+GIoU→狗框
        query#61: CE→车,  L1+GIoU→车框
        query#其余: CE→∅（权重 ×0.1，避免 97 个空槽淹没梯度）
        梯度经 Decoder cross-attn → Encoder → ResNet

Step 4  多个 epoch 后
        某些 query 逐渐「专门化」（如大物体 query、左侧 query——无硬性规定，数据驱动）
        cross-attn 图在目标区域形成高响应 → 定位 + 分类同时学会
```

#### 推理时

```
Step 1  同样前向 → 100 组 (score, class, box)

Step 2  过滤
        对每个 query: score = max(80 类 softmax)
        保留 score > 0.7 的（假设只有 query#7=0.92, #23=0.88, #61=0.95 过线）

Step 3  输出
        3 个检测框，**不做 NMS**
        （若两个 query 偶尔框住同一物体，是已知局限）
```

### 1.7 常见困惑 FAQ

| 问题 | 简短回答 |
|------|----------|
| **没有 anchor，框从哪来？** | 每个 query 的 MLP **直接回归**归一化 `(cx,cy,w,h)`；Cross-Attn 先从 memory 聚合目标区域信息，再回归框——相当于「看了目标再画框」，不是相对 anchor 偏移。 |
| **为什么固定 100 个 query？** | COCO 单图物体数通常 < 100；100 是容量与效率的折中。物体 > 100 时无法全部匹配（极少见）。 |
| **100 个空 query 不会浪费吗？** | 会；所以 ∅ 类 loss 权重降到 0.1，且 self-attn 让 queries 分工。空槽是「集合预测」的代价。 |
| **匈牙利匹配不可微，怎么训练？** | 匹配在 `no_grad` 下只**分配标签**；loss 对**已匹配的 (query, GT) 对**正常反传。类似 EM 或伪标签：分配硬、梯度软。 |
| **Query 初始化随机，怎么知道去哪找？** | 初始确实盲目；靠 GT 匹配监督 + cross-attn 梯度，query 逐渐与特定模式（类/尺度/位置）绑定。前期 AP 低、收敛慢是正常现象。 |
| **Encoder 和 Decoder 分工？** | **Encoder**：混合全图 spatial token，建全局上下文；**Decoder**：100 个 query **读取** memory 并互相协调，输出物体级表示。 |
| **Cross-Attn 和 Self-Attn 各干什么？** | Cross-Attn（query→图像）：**定位**，query 在 825 个空间位置上加权取特征；Self-Attn（query↔query）：**去重协调**，避免多 query 抢同一物体。 |
| **为什么训练 300 epoch？** | Transformer 检测 head 无 anchor 归纳偏置，query  specialization 晚；辅助损失 + 长 schedule 才能收敛。Deformable DETR 可缩到 50 epoch 级。 |
| **真的完全不用 NMS 吗？** | 论文主实验不用；工程上偶发重复框可加 NMS，但非必需。核心是去 NMS 的设计空间，不是 100% 零重复。 |

### 1.8 文档阅读路线

| 你的目标 | 建议阅读 |
|----------|----------|
| **5 分钟懂原理** | §1.5 ~ §1.9（核心原理）+ §1.3 流水线 |
| **搞清网络结构** | §2 Backbone → §3 Encoder/Decoder → §3.6 维度推导 |
| **搞清训练怎么跑** | §5 匈牙利匹配 → §6 训练策略 → §1.6 Walkthrough |
| **搞清为何去 NMS** | §5.4 + §1.7 FAQ |
| **全景分割扩展** | §10 |
| **读官方代码** | §9 + §10.8 |
| **知道后续怎么演进** | §7.2 局限 + §8 相关工作 |

### 1.9 核心原理深讲

本节把 DETR 真正「新」的三块——**集合预测、Attention 定位、匈牙利训练**——拆开讲透。读完应能回答：**信息如何在网络里流动、每个模块在学什么、损失如何驱动行为**。

#### 1.9.1 检测 = 集合预测：问题如何被重新定义

传统检测隐含假设：图像上存在 **大量候选位置**，每个位置独立判断「是不是物体、是什么、框在哪」。  
DETR 换了一个问法：

```
输入:  图像 I
输出:  固定大小的集合  Ŷ = { ŷ_1, ŷ_2, ..., ŷ_N }，N=100

每个 ŷ_j = (c_j, b_j):
  c_j ∈ {1,...,K, ∅}     类别或「空」
  b_j ∈ [0,1]^4          归一化框 (cx, cy, w, h)

GT 同样是集合:  Y = { y_1, ..., y_M }，M 随图变化（通常 M << 100）
```

**关键约束**：每个 GT 物体 **最多** 被一个 query 匹配；每个 query **最多** 对应一个 GT（或 ∅）。  
这不是工程 trick，而是 **问题定义本身**：检测输出是一个 **集合**，不是一张密集 score map。

```
         图像中的真实物体（无序集合）              DETR 的输出（100 个槽位）
              ┌── 猫                              query#7  → (猫, box_7)
              ├── 狗         匈牙利匹配            query#23 → (狗, box_23)
              └── 车         ─────────►            query#61 → (车, box_61)
                                                 query#其余 → (∅, _)
```

集合预测的好处：**训练阶段就规定「一对一」**，推理时不必再靠 NMS 从重复候选里挑一个。

#### 1.9.2 信息如何在网络中流动（五段式）

```
阶段          输入是什么                    输出是什么                   这一步「学会」什么
─────────────────────────────────────────────────────────────────────────────────────────
① Backbone    像素 RGB                      局部语义特征图               边缘、纹理、局部模式
② 投影+位置   特征图 + (x,y) 坐标           hw 个 token 向量             每个格子「我在哪」
③ Encoder     hw 个 token                   memory（仍是 hw 个 token）   格子之间「谁和谁有关」
④ Decoder     100 query + memory            100 个物体级向量 hs          「每个槽位该关注哪、是什么」
⑤ 预测头      hs                            100×(class, box)             最终检测结果
```

**直觉**：Backbone+Encoder 把图变成「**带全局上下文的特征地图**」；Decoder 用 100 个 query 在这张地图上 **各取所需**；预测头把取到的信息 **解码** 成类别和框。

#### 1.9.3 Cross-Attention：定位机制逐步拆解

Cross-Attention 是 DETR 做检测的 **核心物理过程**：query 不直接「看像素」，而是在 **Encoder 输出的 hw 个空间 token** 上做加权聚合。

以 **query#7 要找猫** 为例（训练收敛后）：

```
memory 有 825 个 token，每个对应特征图 25×33 上的一个格子，带位置编码

Cross-Attention 计算:
  ① query#7 的向量 q_7  与  每个 memory token 的 key  k_i  算相似度
     score_i = q_7 · k_i / √32

  ② softmax 归一化 → 注意力权重 α_i，Σ α_i = 1
     若猫在特征图位置 (10, 15) 附近，这些格子的 α 会较大

  ③ 用 α 对 value v_i 加权求和 → 输出向量 h_7
     h_7 = Σ_i α_i · v_i

     h_7 相当于「从全图特征里，把猫所在区域的语义抽出来」
```

```
特征图 25×33（示意）          query#7 的 attention 权重（热力）
┌─────────────────┐          ┌─────────────────┐
│                 │          │  0.01 0.01 ...  │
│      ░░         │   →      │  0.02 0.85 0.80 │  ← 猫所在区域权重高
│      ░░         │          │  0.01 0.75 ...  │
│                 │          │  ...            │
└─────────────────┘          └─────────────────┘
                                      │
                                      ▼ 加权求和 memory
                               h_7 → 分类头 → 「猫」
                                   → 回归头 → (cx,cy,w,h)
```

**为何能回归框**：`h_7` 已聚合猫区域特征；MLP 从 `h_7` 直接输出 `(cx,cy,w,h)`。  
框的中心/宽高 **编码在聚合特征的语义里**——网络学会「看到猫的特征 → 输出猫的框」，而非「在 anchor 基础上偏移」。

**Self-Attention 的配合**：若 query#7 和 query#23 同时高响应「猫」，self-attn 让它们互相抑制，最终只有一个 query 匹配猫（配合匈牙利训练）。

#### 1.9.4 Object Query：从随机向量到「检测槽位」

Query 初始化时是 **与图像无关** 的随机 embedding，三张图共用同一组 100 个 query 权重。  
训练过程中发生的事：

```
Epoch 1~50（探索期）:
  query 输出混乱，分类/框都不准
  匈牙利仍强行分配：「当前最像猫的 query」↔ 猫 GT
  梯度把该 query 的 cross-attn 拉向猫区域

Epoch 50~150（分化期）:
  不同 query 的 attention 模式开始分化
  有 query 常匹配大物体，有 query 常匹配小物体（无显式约束，数据驱动）
  未常被匹配的 query 逐渐学会输出 ∅

Epoch 150~300（精调期）:
  匹配稳定，框和类都较准
  辅助层损失 + 最后一层共同收敛
```

**重要**：query **没有** 被人工指定「query#1 找人、query#2 找车」； specialization 是 **匹配监督 + attention 梯度** 自然涌现的。

#### 1.9.5 匈牙利匹配：手算一个小例子

假设简化场景：**N=4 个 query，M=2 个 GT**（猫、狗）。  
匹配代价矩阵 `C[j,i]`（越小越应该匹配）：

```
              GT: 猫    GT: 狗
query#1       8.2      7.9
query#2       0.6      9.1      ← 对猫代价最低
query#3       8.0      0.5      ← 对狗代价最低
query#4       7.5      7.8
```

匈牙利算法找 **总代价最小的一一匹配**：

```
选 query#2 ↔ 猫（0.6），query#3 ↔ 狗（0.5）
总代价 = 0.6 + 0.5 = 1.1

query#1、#4 未匹配 → 标签设为 ∅
```

**匹配在 `torch.no_grad()` 下执行**：算法本身不反传；但它决定了 **谁该学猫、谁该学狗、谁该学空**。  
下一步 loss 对 query#2 施加「分类=猫 + 框=猫框」，对 query#1/#4 施加「分类=∅」。

```
训练循环（概念）:

  pred = model(image)                    # 有梯度
  indices = hungarian_matcher(pred, gt)  # 无梯度，只返回 [(2,猫), (3,狗)]
  loss = criterion(pred, gt, indices)    # 有梯度，按 indices 填 label
  loss.backward()
```

#### 1.9.6 GIoU：框损失为何不只靠 L1

L1 只度量 **坐标数值差**，框大小差异大时梯度不稳定。GIoU 把 **两框重叠程度 + 最小外接矩形** 纳入：

```
给定预测框 B 与 GT 框 G:

IoU   = |B ∩ G| / |B ∪ G|

GIoU  = IoU - |C \ (B ∪ G)| / |C|
        其中 C 是同时包住 B 和 G 的最小矩形

loss  = 1 - GIoU     完全重合 → 0，不重叠 → 接近 2
```

**数值例子**（归一化坐标，同中心，预测框偏小）：

```
G: cx=0.5, cy=0.5, w=0.4, h=0.4     （GT 框）
B: cx=0.5, cy=0.5, w=0.2, h=0.2     （预测框，偏小）

IoU ≈ 0.25
GIoU ≈ 0.25 - 0.5625 = -0.31       → loss = 1 - (-0.31) = 1.31

若 B 扩大到与 G 一致: IoU=1, GIoU=1 → loss=0
```

匈牙利 **匹配代价** 和 **训练损失** 都用 GIoU，使「该配对的 query-GT 对」在 **位置、尺度、重叠** 上一致。

#### 1.9.7 分类损失与 ∅ 类：100 槽位如何平衡

100 个 query 里通常只有 ~3–10 个有物体，**正负极不平衡**。

```
对 N=100 个 query 的分类标签填充:

  匹配到 GT 的 M 个 query  →  真类 id（0~79）
  其余 (100-M) 个 query    →  ∅ 类（id=80）

Cross-Entropy 计算时:
  ∅ 类 loss 乘以 eos_coef=0.1
  避免 97 个「学背景」的梯度压过 3 个「学物体」的梯度
```

**回归 loss 只对 matched 的 M 个 query 计算**——空槽位不需要框，也没有框 GT。

#### 1.9.8 三大机制如何协同（总图）

```
                    ┌──────────────────────────────────────┐
                    │            一张训练图                 │
                    └──────────────────────────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
   Object Query                  Transformer                   匈牙利匹配
   「输出通道」                  「读图 + 聚合」                「监督分配」
         │                            │                            │
   100 个可学习槽位              Encoder: 全图 token 互看          pred ↔ GT 最优配对
   每槽位 1 组输出              Decoder: query cross-attn 定位    未配对 → ∅
         │                            │                            │
         └────────────────────────────┼────────────────────────────┘
                                      ▼
                         匹配上的 query: L_cls + L_L1 + L_GIoU
                         未匹配的 query: L_cls(∅) × 0.1
                                      │
                                      ▼
                         梯度回传 → query 更准定位、更准分类
                                  → Encoder 特征更适合被 query 读取
                                  → Backbone 提取更有判别力的语义
```

**一句话串起来**：Query 提供 **固定输出接口**；Transformer 负责 **从全图提取每个接口该看的特征**；匈牙利匹配告诉网络 **每个接口该学哪个 GT（或学空）**——三者缺一不可。

---

## 2. Stage 1：CNN Backbone

DETR 不自己设计 backbone，直接沿用 **ResNet**，并**冻结 BatchNorm**（使用 FrozenBN，小 batch 检测训练更稳定）。

### 2.1 结构选择

| 变体 | 输出通道 C | 下采样步长 | 特征图尺寸（输入 800×1066） |
|------|-----------|------------|------------------------------|
| ResNet-50 | 2048 | 32 | 25 × 33 |
| ResNet-101 | 2048 | 32 | 同上，更深 |
| ResNet-50 + DC5 | 2048 | **16**（dilation） | 50 × 67，计算更重 |

标准配置取 **ResNet 最后一 stage（conv5）** 的输出，**不做 FPN 多尺度融合**——单尺度特征 + Transformer 全局 attention 是论文的默认设定。

### 2.2 输入预处理

```
训练 / 推理:
  1. 短边 resize 到 800（多尺度训练时随机 480~800）
  2. 长边 max 1333
  3. ImageNet mean/std 归一化
  4. batch 内 padding 到相同 H×W（mask 标记有效区域）
```

### 2.3 从特征图到 Transformer 输入

Backbone 输出 `(B, 2048, H/32, W/32)` 后：

```
Step 1: 1×1 Conv + GroupNorm
  2048 → d_model = 256

Step 2: 展平空间维
  (B, 256, h, w) → (B, 256, hw) → (B, hw, 256)
  其中 hw = h × w，例如 25×33 = 825

Step 3: 加 Positional Encoding（与序列逐元素相加）
  每个空间位置 (x, y) 对应一个 d_model 维位置向量
```

> **设计要点**：Transformer 本身不带空间归纳偏置，**必须**显式注入 2D 位置信息；DETR 使用与原始 Transformer 相同的 **正弦位置编码**，对 `(x, y)` 分别用 sin/cos 函数编码后再拼接。

#### 2D 正弦位置编码（公式）

对特征图位置 `(x, y)`（x ∈ [0, w-1], y ∈ [0, h-1]），各用一半维度编码后 concat：

```
PE(x, 2k)   = sin( x / 10000^(2k/d_half) )
PE(x, 2k+1) = cos( x / 10000^(2k/d_half) )
PE(y, ...)  = 同上，用 y 代入

最终 pos(x,y) = concat( PE_x, PE_y )  ∈ R^d     d=256 时 x/y 各 128 维
```

加到 Encoder 的 Q/K 上（V 不加）；Decoder cross-attn 的 K/V（memory 侧）也加 pos。**Query 侧**用 `query_embed` 作为 query position，与 NLP 里 target embedding 角色类似。

---

## 3. Stage 2~4：Transformer Encoder-Decoder

这是 DETR 的**核心网络结构**，整体遵循「Attention Is All You Need」的 Encoder-Decoder 范式，但输入不是词 token，而是 **CNN 特征 token + 可学习 object query**。

### 3.1 维度与超参（默认 COCO 配置）

| 符号 | 含义 | 默认值 |
|------|------|--------|
| d_model | 隐藏维度 | 256 |
| n_heads | 多头注意力头数 | 8 |
| d_ff | FFN 中间层维度 | 2048 |
| L_enc | Encoder 层数 | 6 |
| L_dec | Decoder 层数 | 6 |
| N_queries | Object Query 数量 | 100 |
| dropout | — | 0.1 |

### 3.2 Encoder：全局上下文建模

```
输入:  flatten 特征 + pos enc     形状 (hw, B, d_model)  [实现中常用 seq-first]
       │
       ▼  × L_enc 层
┌──────────────────────────────────────────────────────────┐
│  Multi-Head Self-Attention                                │
│    Q = K = V = x + pos                                    │
│    每个空间位置 attend 到所有 hw 个位置 → 全局感受野       │
├──────────────────────────────────────────────────────────┤
│  Add & Norm                                               │
├──────────────────────────────────────────────────────────┤
│  FFN: Linear(d→2048) → ReLU → Linear(2048→d)             │
├──────────────────────────────────────────────────────────┤
│  Add & Norm                                               │
└──────────────────────────────────────────────────────────┘
       │
       ▼
输出 memory: (hw, B, d_model)   —— 供 Decoder cross-attention 读取
```

**Encoder 在做什么**：把 CNN 的局部卷积特征「混合」成带全局关系的表示。例如「左边的人」和「右边的球」可以在 self-attention 里建立长程关联。

#### 原理：Self-Attention 如何让每个格子「看见全图」

Encoder 每一层里，**每个 spatial token 都会与全部 hw 个 token 计算注意力**。  
对位置 `(x,y)` 的 token 而言：

```
输出 token(x,y) = Σ_{所有 (x',y')}  α(x,y ← x',y') · V(x',y')

α 由 Q(x,y) 与 K(x',y') 的相似度决定（softmax 后 Σα=1）
```

**一层 self-attn 后**：`(x,y)` 的特征已混入全图信息。  
**堆 6 层后**：每个 token 理论上能感知 **任意距离** 的物体关系——遮挡、共现、大小对比等，都在 memory 里编码好了。

```
仅 CNN（3×3 卷积）:  感受野有限，远处物体互不可见
Encoder self-attn:   任意两格可直接交互 → 「人举着球」这种关系可被建模
```

Decoder 的 cross-attn 读到的 memory，已是 **全局混合后的特征地图**，不是原始 CNN 局部特征。

### 3.3 Object Queries：Decoder 的「检测槽位」

DETR 引入 **N=100 个可学习的 embedding**，记为 `Q ∈ R^{N×d_model}`：

```
nn.Embedding(100, 256)   # 与图像内容无关，随机初始化，训练中学习

语义（事后解释）:
  每个 query 逐渐专门化到某类、某尺度、某空间区域
  例如 query #17 可能学会「找画面中央的大人」
  未匹配到 GT 的 query 学会输出 ∅（no object）
```

Object Query 的关键属性：

| 属性 | 说明 |
|------|------|
| 定义方式 | **数据驱动学习**的 slot，非手工尺度/比例 |
| 数量 | 固定 N=100 |
| 与位置关系 | 通过 cross-attention **动态**从 memory 读特征 |
| 空预测 | 未匹配 query → ∅ 类 |

### 3.4 Decoder：Query 读 Memory、互相抑制

```
输入:
  tgt = zeros 或 learnable query embedding     (N, B, d_model)
  memory = Encoder 输出                          (hw, B, d_model)
  pos enc 同时加到 memory 的 K/V 上

       ▼  × L_dec 层
┌──────────────────────────────────────────────────────────┐
│  Self-Attention（query 之间）                             │
│    Q,K,V 来自 tgt；query 互相可见 → 避免多个 slot 抢同一物体│
├──────────────────────────────────────────────────────────┤
│  Cross-Attention（query → 图像）                          │
│    Q 来自 tgt，K/V 来自 memory+pos                        │
│    每个 query 在 hw 个空间位置上做 weighted sum            │
├──────────────────────────────────────────────────────────┤
│  FFN                                                     │
└──────────────────────────────────────────────────────────┘
       │
       ▼
输出: (N, B, d_model)  →  接预测头
```

**Cross-Attention 是检测的「定位机制」**：某个 query 若与某块特征高度相关，其 attention map 会在该物体区域形成峰值——可视化 attention 可看到 query 逐渐「看向」目标。

#### 原理：Decoder 一层内三件事的顺序

每一层 Decoder 按 **Self-Attn → Cross-Attn → FFN** 执行，顺序不可乱：

```
① Self-Attention（query ↔ query）
   目的: 100 个槽位互相通信，「这个物体我已经占了」
   效果: 抑制多个 query 同时盯同一区域

② Cross-Attention（query → memory）
   目的: 每个 query 从 hw 个空间 token 里 **按需取特征**
   效果: 完成定位——attention 峰值落在 GT 物体所在特征格

③ FFN
   目的: 非线性变换，把聚合特征映射到更易分类/回归的空间
   效果: 为下一层或最终预测头准备表示
```

6 层堆叠 = **6 轮「协调 → 定位 → 提炼」**；浅层 attention 较散，深层逐渐聚焦目标。

#### 原理：query 向量如何同时承载「类」与「框」

同一个 `h_i` 分叉进两个头，**共享底层语义、任务特定解码**：

```
hs[i] ──┬── Linear(256→81)     → 分类 logits（是谁 / 是不是 ∅）
        └── MLP(256→4)         → bbox（在哪）

共享前提: cross-attn 抽出的 h_i 已包含「这个位置有猫、形状如此」的信息
分类头: 读「是什么」
回归头: 读「框参数」——同一语义向量的不同线性/非线性投影
```

### 3.5 整体数据流示意

```
ResNet 特征图 (h×w 个位置)
    │ flatten + pos
    ▼
┌─────────────┐
│  Encoder    │  hw tokens ──self-attn──► 全局 memory
└─────────────┘
         │
         │ cross-attn
         ▼
┌─────────────┐
│  Decoder    │  100 queries ──self-attn──► 互相协调
└─────────────┘
         │
         ▼
   100 × (class, box)
```

### 3.6 逐层 Attention 维度推导

下面用 **COCO 默认配置 + 一张具体尺寸的图** 做完整维度追踪。符号约定：

| 符号 | 含义 | 本例取值 |
|------|------|----------|
| B | batch size | 2 |
| H, W | 输入图像高宽（padding 后） | 800, 1066 |
| h, w | 特征图高宽（步长 32） | 25, 33 |
| hw | 序列长度 h×w | **825** |
| N | Object Query 数 | **100** |
| d | d_model | **256** |
| h_heads | 注意力头数 | **8** |
| d_k | 每头维度 d / h_heads | **32** |
| d_ff | FFN 中间维 | **2048** |

> **实现约定**：PyTorch 官方代码 `models/transformer.py` 采用 **seq-first** 布局 `(seq_len, B, d)`，与 NLP Transformer 一致；下面张量形状均按此写法。

#### 3.6.1 进入 Transformer 前

```
Backbone 输出:           (B, 2048, h, w)        = (2, 2048, 25, 33)
1×1 Conv 投影:           (B, d, h, w)           = (2, 256, 25, 33)
flatten + transpose:     (hw, B, d)             = (825, 2, 256)   ← Encoder 输入 src
Positional Encoding:     (hw, B, d)             = (825, 2, 256)   ← pos，与 src 相加用于 Q/K
Object Query Embedding:  (N, d) → expand       = (100, 2, 256)   ← Decoder 输入 tgt
```

#### 3.6.2 Multi-Head Attention 通用公式

单层 MHA 内部（单头视角，最终 concat 8 头）：

```
输入 X: (L, B, d)     L = 序列长度（Encoder 中 L=hw；Decoder self-attn 中 L=N）

Linear 投影（共享权重，一次矩阵乘再拆分）:
  W_q, W_k, W_v ∈ R^{d×d}
  Q = X · W_q          → (L, B, d)
  K = X · W_k          → (L, B, d)
  V = X · W_v          → (L, B, d)

拆成 h_heads 头，每头 d_k = d / h_heads = 32:
  Q → reshape → (B, h_heads, L, d_k)    本例 (B, 8, L, 32)
  K → (B, h_heads, L, d_k)
  V → (B, h_heads, L, d_k)

注意力权重:
  A = softmax( Q · K^T / √d_k )        → (B, h_heads, L, L)

输出:
  O = A · V                            → (B, h_heads, L, d_k)
  concat heads → (L, B, d)
  O · W_o                              → (L, B, d)    输出投影
```

**Scaled Dot-Product** 中除以 `√32 ≈ 5.66`，防止点积过大导致 softmax 梯度消失。

#### 3.6.3 Encoder 单层（Self-Attention + FFN）

设输入 `x^{(0)} = src + 0`（第一层前），每层结构相同：

```
────────────────── Self-Attention ──────────────────
输入 x:                    (hw, B, d)     = (825, 2, 256)
Q = K = x + pos:           (825, 2, 256)
V = x:                     (825, 2, 256)

reshape 为多头:
  Q, K, V:                 (B, 8, 825, 32)

注意力矩阵:
  A = QK^T / √32:          (B, 8, 825, 825)    ← O(hw²) 内存/算力瓶颈

加权求和:
  A · V:                   (B, 8, 825, 32)
  merge → 输出投影:         (825, 2, 256)

残差 + LayerNorm:
  x' = LN(x + Dropout(out))                (825, 2, 256)

────────────────── FFN ──────────────────
FFN(x') = W2 · ReLU(W1 · x'):
  W1: 256 → 2048    中间: (825, 2, 2048)
  W2: 2048 → 256    输出: (825, 2, 256)

残差 + LayerNorm:
  x'' = LN(x' + Dropout(FFN))              (825, 2, 256)
```

**6 层 Encoder 堆叠**后得到 **memory**，形状不变：

```
memory: (hw, B, d) = (825, 2, 256)
```

**参数量（单层 Encoder，粗算）**：

| 模块 | 参数量 |
|------|--------|
| Self-Attn Q/K/V/O | 4 × 256² ≈ 262K |
| FFN | 256×2048 + 2048×256 ≈ 1.05M |
| LayerNorm × 2 | 可忽略 |
| **单层合计** | **~1.3M** |
| **6 层 Encoder** | **~7.8M** |

#### 3.6.4 Decoder 单层（Self-Attn + Cross-Attn + FFN）

Decoder 第 ℓ 层输入 `tgt^{(ℓ)}` 形状 `(N, B, d) = (100, 2, 256)`。

**① Self-Attention（query ↔ query）**

```
Q = K = tgt + query_pos:   (100, 2, 256)    query_pos 通常与 query_embed 相同
V = tgt:                   (100, 2, 256)

多头:
  Q, K, V:                 (B, 8, 100, 32)

注意力矩阵:
  A_self = QK^T / √32:     (B, 8, 100, 100)   ← N²=1e4，远小于 hw²

输出 + 残差 + LN:          (100, 2, 256)      → tgt'
```

**② Cross-Attention（query → 图像 memory）**

```
Q = tgt' + query_pos:      (100, 2, 256)
K = memory + pos:          (825, 2, 256)
V = memory:                (825, 2, 256)

多头:
  Q:                       (B, 8, 100, 32)
  K, V:                    (B, 8, 825, 32)

注意力矩阵（定位核心）:
  A_cross = QK^T / √32:    (B, 8, 100, 825)   ← 每个 query 对 hw 个空间位置打分

  可视化: A_cross[b, head, n, :] reshape 成 (25, 33) 即 query #n 的空间注意力图

加权求和:
  A_cross · V:             (B, 8, 100, 32)
  merge + 投影:            (100, 2, 256)

残差 + LN:                 (100, 2, 256)      → tgt''
```

**③ FFN**

```
同 Encoder FFN:  (100, 2, 256) → (100, 2, 2048) → (100, 2, 256)
残差 + LN:       (100, 2, 256)
```

**6 层 Decoder 输出** `hs`：`(N, B, d) = (100, 2, 256)`，转置后接预测头 → `(B, 100, 256)`。

#### 3.6.5 预测头维度

```
hs:                         (B, N, d)           = (2, 100, 256)

分类头 Linear(d → K+1):
  logits:                   (B, N, K+1)         = (2, 100, 81)

回归头 MLP(d → d → d → 4):
  bbox (sigmoid 前):        (B, N, 4)           = (2, 100, 4)
```

#### 3.6.6 全链路维度一览表

以 **B=2, 800×1066, ResNet-50, N=100** 为例：

| 阶段 | 张量 | 形状 | 说明 |
|------|------|------|------|
| 输入图像 | `image` | (2, 3, 800, 1066) | padding 后 |
| Backbone | `features` | (2, 2048, 25, 33) | 步长 32 |
| 投影 | `src` | (825, 2, 256) | flatten 后 seq-first |
| Enc Self-Attn | `A` | (2, 8, 825, 825) | 每层，680K 元素/头 |
| Enc 输出 | `memory` | (825, 2, 256) | 6 层后 |
| Dec 输入 | `tgt` | (100, 2, 256) | query embed |
| Dec Self-Attn | `A_self` | (2, 8, 100, 100) | query 互相关 |
| Dec Cross-Attn | `A_cross` | (2, 8, 100, 825) | **检测定位** |
| Dec 输出 | `hs` | (100, 2, 256) | 6 层后 |
| 分类 | `pred_logits` | (2, 100, 81) | 含 ∅ |
| 回归 | `pred_boxes` | (2, 100, 4) | cxcywh ∈ [0,1] |

#### 3.6.7 计算量对比（量级）

```
Encoder 单层 Self-Attn:
  QK^T:  O(B · h_heads · hw² · d_k)  ≈  B · 8 · 825² · 32  ≈  1.7×10⁸ / batch

Decoder 单层 Cross-Attn:
  QK^T:  O(B · h_heads · N · hw · d_k)  ≈  B · 8 · 100 · 825 · 32  ≈  2.1×10⁷ / batch

比值: Encoder Self-Attn / Decoder Cross-Attn ≈ hw / N ≈ 825/100 ≈ 8×（单层）
      但 Encoder 有 6 层、hw² 增长更快 → **大图上 Encoder 是绝对瓶颈**

DC5（步长 16）: hw 变为 4× → Encoder Self-Attn 计算约 **16×**（825→3300, hw² 增 16 倍）
```

#### 3.6.8 与标准 NLP Transformer 的差异

| 项目 | NLP Seq2Seq | DETR |
|------|-------------|------|
| Encoder 序列 | 词 token 数（~512） | hw（~800+，随图变大） |
| Decoder 序列 | 目标句长度（变长） | **固定 N=100** |
| Cross-Attn 的 K/V | 源句 token | **CNN 空间特征 grid** |
| 位置编码 | 1D sin/cos | **2D sin/cos**（x、y 各编码再 concat） |
| 输出 | 每步一个词 | 每 query 一组 (class, box) |

---

## 4. Stage 5：预测头（Prediction FFN）

Encoder-Decoder 输出的每个 query 向量 `h_i ∈ R^256` 独立过两个头：

### 4.1 分类头

```
Linear(256 → num_classes + 1)

输出: logits_i ∈ R^{K+1}
  前 K 维: COCO 80 类
  最后 1 维: ∅（no object / background）

训练目标: 匹配到 GT 的 query → 对应类 one-hot
          未匹配的 query → ∅ 类
推理: softmax 后丢弃 ∅，或取 argmax ≠ ∅ 且 score > 0.7
```

**注意**：DETR **没有** sigmoid 多标签分类；是标准的 **softmax 单类预测**，与「每个 query 最多对应一个物体」的集合假设一致。

### 4.2 边界框回归头

```
MLP: Linear(256→256) → ReLU → Linear(256→256) → ReLU → Linear(256→4)

输出: (cx, cy, w, h) 经 sigmoid → 全部在 [0, 1]

坐标含义（相对整图归一化，与 GT 一致）:
  cx, cy = 框中心相对图像宽高的比例
  w, h   = 框宽高相对图像宽高的比例

DETR 直接预测绝对归一化框（sigmoid 后 ∈ [0,1]），无 reference box、无偏移量回归。
```

#### 原理：归一化框如何与图像坐标互转

```
训练 / 推理内部统一用 (cx, cy, w, h)，相对 **整图** 宽高归一化:

  cx = box_center_x / W_image
  cy = box_center_y / H_image
  w  = box_width     / W_image
  h  = box_height    / H_image

推理还原到像素（xyxy）:
  x1 = (cx - w/2) × W_image
  y1 = (cy - h/2) × H_image
  x2 = (cx + w/2) × W_image
  y2 = (cy + h/2) × H_image

例: 图像 800×1066，预测 cx=0.5, cy=0.3, w=0.2, h=0.15
  中心 ≈ (533, 240)，宽约 213 px，高约 160 px
```

sigmoid 保证输出恒在 [0,1]，与 GT 归一化方式一致，匹配 loss 可直接比较。

### 4.3 辅助损失（Auxiliary Decoding Loss）

训练时对 **Decoder 每一层** 的输出都接同样的预测头并算匹配损失（权重相同）：

```
L_total = L_final + Σ_{layer=1}^{L_dec-1} L_aux_layer

作用: 加深监督，缓解 Transformer 训练难、收敛慢的问题
推理: 只用最后一层输出
```

---

## 5. 核心机制：集合预测与二分图匹配

这是 DETR **集合预测训练**的核心：不用 IoU 阈值分配正负样本，而是 **匈牙利算法** 在预测集合与 GT 集合之间找最优双射。

### 5.0 集合预测 vs 逐点预测（原理）

```
逐点预测（密集 head）:
  特征图每个位置都输出 (class, box)
  需要规则决定: 哪些位置算正样本、哪些算负样本、重复框怎么处理

集合预测（DETR）:
  只输出 N 个 (class, box)，N 固定
  需要规则决定: N 个预测里谁对应哪个 GT、谁该输出 ∅
  → 匈牙利匹配就是这个规则，且是 **全局最优** 的一对一分配
```

DETR 把「样本分配」从 **手工 heuristic** 变成 **与 loss 一致的最优化问题**：匹配代价用的就是分类+框的误差，匹配结果直接用于算 loss。

### 5.1 问题表述

```
一张图:
  GT 集合:  Y = {y_i}_{i=1}^{M}     M 可变，通常 M << 100
  预测集合: Ŷ = {ŷ_j}_{j=1}^{N}     N=100 固定

目标: 找一个匹配 σ，使 matched pairs 的损失最小
      未匹配的 N-M 个预测 → 监督为 ∅
```

### 5.2 匹配代价（Hungarian Cost）

对每一对 (预测 j, GT i)，定义 **不参与梯度** 的匹配代价：

```
C_match(j, i) = λ_cls · (-p_j[c_i]) + λ_box · L_box(b_j, b_i) + λ_giou · L_GIoU(b_j, b_i)

其中:
  p_j[c_i]  = 预测 j 对 GT 类别 c_i 的 softmax 概率
  b_j, b_i  = 归一化 (cx,cy,w,h)
  L_box     = L1 距离
  L_GIoU    = 1 - GIoU（Generalized IoU）

默认权重: λ_cls=1, λ_box=5, λ_giou=2
```

在 `N × M` 代价矩阵上用 **匈牙利算法**（`scipy.optimize.linear_sum_assignment`）求最小总代价的 **一一匹配**。

```
        GT_1   GT_2   GT_3
 pred_1  0.3   8.2    9.1
 pred_2  7.5   0.4    8.0
 pred_3  6.1   7.8    0.5
 ...

匹配结果例: pred_1↔GT_1, pred_2↔GT_2, pred_3↔GT_3
其余 97 个 pred → ∅
```

#### 原理：匹配代价三项各管什么

```
-λ_cls · p_j[c_i]     分类项：预测 j 对 GT 类 c_i 的 softmax 概率越大，代价越小
                       → 倾向「类已经猜对」的 query 去匹配该类 GT

L_box(b_j, b_i)        L1 框坐标差：中心、宽高数值接近的配对代价小

L_GIoU(b_j, b_i)       1-GIoU：框重叠越多代价越小
                       → 即使 L1 接近，框形状差很多时 GIoU 仍会惩罚

λ_box=5, λ_giou=2 > λ_cls=1  →  匹配时 **位置比分类更重要**
                                  先把框对齐，再管类是否猜对
```

### 5.3 训练损失（Set Loss）

匹配确定后，对 **所有 N 个 query** 计算：

```
L = λ_cls · L_ce(class) + λ_box · L_L1(box) + λ_giou · L_GIoU(box)

分类 L_ce:
  匹配到的 query: 交叉熵 → 真类
  未匹配的 query: 交叉熵 → ∅ 类
  （∅ 类权重通常降低，如 eos_coef=0.1，缓解 100 槽位中前景极少的不平衡）

回归 L_L1 + L_GIoU:
  **仅对匹配到 GT 的 query** 计算
  框坐标: L1(b_pred, b_gt)
  GIoU: 1 - GIoU(b_pred, b_gt)，尺度不变、对框质量更敏感
```

默认: `λ_cls=1, λ_box=5, λ_giou=2`

#### 原理：Set Loss 如何填 label 并反传

匹配完成后，为 **每个 query** 构造监督（示意 M=3, N=100）：

```
query index    匹配结果        分类 label    是否算 box loss
─────────────────────────────────────────────────────────
#7             ↔ 猫           类「猫」       ✓ L1 + GIoU
#23            ↔ 狗           类「狗」       ✓
#61            ↔ 车           类「车」       ✓
#1,#2,...#99   未匹配         ∅            ✗（不算框 loss）

L_total = 100 个 query 的分类 loss（∅ 降权）
        + 3 个 matched query 的 L1 + GIoU
        + 5 层 Decoder 辅助 loss（同样流程）
```

**梯度路径**：loss → 预测头 → Decoder hs → cross-attn → memory → Encoder → backbone。  
匹配本身无梯度，但 **一旦 query#7 被分配为「猫」**，后续 CE/GIoU 就会拉动 query#7 的 cross-attn 去猫的 memory 区域。

### 5.4 为何可以去掉 NMS

密集预测范式下，同一物体可能被多个候选高分命中，通常需要 NMS 去重。

DETR 的设计：

1. **固定 N 个 slot**，训练时用匈牙利匹配强制 **每个 GT 最多匹配一个 query**；
2. **Self-attention 让 queries 互相「商量」**，减少多个 query 同时高置信抢同一物体；
3. 未用到的 query 被推向 **∅ 类**。

因此推理时通常 **直接取 score > 阈值 的预测即可**，论文主实验 **不做 NMS**。实践中若 query 数 > 图像物体数，偶尔仍可能出现重复框，但整体上重复预测远少于未做集合匹配的密集 head。

---

## 6. 训练策略与处理方式

### 6.1 优化配置

| 项目 | 设置 |
|------|------|
| 优化器 | AdamW |
| 骨干 LR | 1e-5 |
| Transformer LR | 1e-4 |
| weight decay | 1e-4 |
| batch size | 64（2 GPU × 32 或类似） |
| 训练轮数 | **300 epoch**（COCO） |
| LR 调度 | Step 降 LR：200 epoch 时 ×0.1 |
| 梯度裁剪 | max_norm = 0.1 |

> **收敛慢**是 DETR 早期最大痛点：前 100 epoch AP 很低，需要更长 schedule；这也是后续 Deformable DETR、Conditional DETR 的重要动机。

### 6.2 数据增强

```
标准 COCO 检测增强:
  - RandomHorizontalFlip
  - 多尺度: 短边 random 480~800，长边 max 1333
  - Normalize
```

### 6.3 每张图的 GT 处理

```
COCO annotation → 过滤 crowd、无效框
每张图:
  boxes:  (M, 4)  xyxy → 转为 (cx,cy,w,h) 并除以 [W,H,W,H] 归一化到 [0,1]
  labels: (M,)    类别 id 0~79

若 M > 100: 理论上无法完全匹配（极少见）；COCO 单图物体数通常 < 100
若 M = 0:  所有 query 监督为 ∅
```

### 6.4 推理流程

```
1. 图像 → resize / normalize → backbone → transformer → 100 预测
2. 对每个 query: score = max softmax over 80 类（不含 ∅）
3. 保留 score > threshold（默认 0.7）的预测
4. 框: sigmoid 输出已是 [0,1]，乘回原图 W,H 得到像素坐标
5. （可选）极少数场景可加 NMS，论文主结果不用
```

### 6.5 复杂度粗算

```
设 hw = H/32 × W/32

Encoder self-attention:  O(hw² · d)     全局，大特征图时贵
Decoder cross-attention: O(N · hw · d)  N=100 相对小
Decoder self-attention:  O(N² · d)      可忽略

大图上 **Encoder 的 O(hw²)** 是主要计算瓶颈；Decoder 侧 N=100 固定，开销相对可控。
```

---

## 7. 特点、优势与局限

### 7.1 主要特点

| 特点 | 说明 |
|------|------|
| **端到端简洁** | 无 RPN、anchor、NMS、RoI Pooling |
| **全局推理** | Self/Cross-Attention 建模长程依赖 |
| **集合预测** | 输出是集合而非密集 heatmap |
| **匹配驱动训练** | 匈牙利分配替代 heuristic 正负样本 |
| **大物体强** | COCO AP_L 表现较好 |
| **扩展性好** | 同一框架可接 panoptic segmentation（DETR + mask head） |

### 7.2 主要局限（论文与社区共识）

| 局限 | 原因 | 后续改进方向 |
|------|------|--------------|
| **训练极慢** | 300 epoch、匹配不稳定、query  specialization 晚 | Deformable DETR、DN-DETR |
| **小物体弱** | 步长 32、单尺度特征、query 数有限 | 多尺度特征、Deformable attention |
| **Encoder 计算贵** | O(hw²) attention | 稀疏 attention、局部特征 |
| **query 与 GT 数不对称** | N=100 固定，空 slot 多 | 动态 query、两阶段 query 初始化 |
| **匹配不可微** | 匈牙利在 no_grad 下 | 部分工作探索 soft matching |

### 7.3 DETR-DC5 变体

```
在 ResNet 最后一 block 使用 dilation=2:
  特征步长 32 → 16
  hw 变为 4 倍 → Encoder 计算约 ×16
  AP 提升（尤其 AP_S），但速度明显下降
```

### 7.4 核心设计决策速查（Why DETR?）

| 设计选择 | 做法 | 为什么 | 代价 |
|----------|------|--------|------|
| 候选机制 | 100 个 Object Query | 集合预测，无需手工 anchor 尺度/比例 | 容量上限 100；空槽多 |
| 特征交互 | Transformer 全局 Attention | 长程依赖、物体间关系（遮挡、上下文） | Encoder O(hw²) 贵 |
| 多尺度 | 单尺度 conv5（无 FPN） | 结构简洁，证明 Transformer 可行 | 小物体 AP 偏低 |
| 框回归 | 直接 sigmoid(cxcywh) | 与匹配损失一致，无 anchor 参照 | 早期定位难学 |
| 样本分配 | 匈牙利二分匹配 | 端到端集合监督，天然一对一 | 匹配硬、收敛慢 |
| 重复框 | Self-attn + ∅ + 匹配 | 训练阶段抑制，推理免 NMS | 偶发重复 |
| 空预测 | 第 K+1 类 ∅ | 统一 softmax，未匹配 query 有明确监督 | 需 eos_coef 平衡 |
| 定位损失 | L1 + GIoU | L1 回归 + GIoU 对框形状更敏感 | 匹配代价需调 λ |
| 深度监督 | 每层 Decoder 辅助 loss | 缓解 Transformer 训练难 | 训练更慢，推理只用最后一层 |
| BN | FrozenBN on backbone | 小 batch 检测训练稳定 | 与标准 ImageNet BN 不同 |

---

## 8. DETR 系列演进（后续工作）

```
DETR (2020) ──► Transformer + 集合匹配，去 NMS / anchor
     │
     ├── Deformable DETR (2021): 可变形 cross-attn，多尺度，收敛快 ~10×  → 详见 [DeformableDETR.md](./DeformableDETR.md)
     ├── Conditional DETR (2021): 条件 query，缓解 slow convergence
     ├── DINO (2022): 对比去噪 query，SOTA 级 DETR 系
     └── RT-DETR (2023): 实时化 hybrid encoder
```

**相关思想（非检测专属）**：

- **Set Prediction Loss**：匈牙利匹配训练集合输出（源于匹配预测类任务）；
- **Transformer in Vision**：ViT 用 patch 序列；DETR 用 CNN feature token + detection queries。

---

## 9. 实现细节备忘（读源码时可对照）

官方实现 [facebookresearch/detr](https://github.com/facebookresearch/detr) 中常见对应关系：

| 论文概念 | 代码模块（大致） |
|----------|------------------|
| Backbone | `models/backbone.py` → ResNet + Joiner |
| 1×1 投影 | `input_proj` Conv2d |
| Positional Encoding | `models/position_encoding.py` |
| Transformer | `models/transformer.py` |
| Object Query | `query_embed` nn.Embedding |
| 分类 / 框头 | `class_embed`, `bbox_embed` |
| 匈牙利匹配 | `models/matcher.py` |
| Set Loss | `SetCriterion` in `models/detr.py` |

**读代码建议顺序**：

1. `build_model` → 看清维度如何从 `(B,3,H,W)` 到 `(B,100,K+1)` 和 `(B,100,4)`  
2. `HungarianMatcher` → 理解 `cost_class + cost_bbox + cost_giou`  
3. `SetCriterion` → `labels` 如何填 ∅、`boxes` 如何只对 `indices` 算 loss  
4. `forward` 里 `aux_outputs` → 辅助层损失  

---

## 10. Panoptic Segmentation 扩展（DETR + Mask Head）

DETR 论文 **Section 7** 在同一套集合预测框架上扩展 **全景分割（Panoptic Segmentation）**： countable 的 **things**（人、车等，每实例一个 mask）与 amorphous 的 **stuff**（天空、路面等，按类合并）统一输出。核心思路是 **检测分支 + 实例 mask 点积头 + stuff 语义头 + 推理融合**，仍保持端到端训练、匈牙利匹配。

### 10.1 Panoptic 任务回顾

COCO Panoptic 把像素分成两类标注：

| 类型 | 含义 | 例子 | DETR 由谁负责 |
|------|------|------|---------------|
| **Things** | 可数实例，每物体独立 id | 人、车、猫 | Object Query + **Mask Head** |
| **Stuff** | 不可数区域，同类合并 |  sky、grass、wall | Encoder 上的 **Semantic Head** |

评价指标 **PQ（Panoptic Quality）** = 对每个 (class, segment) 匹配的 **SQ（Segmentation Quality，IoU）** 与 **RQ（Recognition Quality）** 的综合。

### 10.2 扩展后整体结构

在检测 DETR 基础上增加两个分支，**共享同一 backbone + Transformer Encoder**：

```
Input Image
    │
    ▼
ResNet + Transformer Encoder  ──► memory (hw, B, d)  ──► reshape ──► (B, d, h, w)
    │                                      │
    │                                      ├──► Semantic Head（stuff）
    │                                      │         逐像素 K_stuff 类 logits
    │                                      │         输出: (B, K_stuff, h, w)
    ▼                                      │
Transformer Decoder + 检测头               │
    │                                      │
    ▼                                      │
100 × (class, box)                         │
    │                                      │
    ▼                                      │
Mask Head: query embedding · 逐像素 embedding
    │         m_i = MLP(h_i)               │
    │         M_i(x,y) = σ( m_i · E(x,y) ) │
    ▼                                      ▼
100 张实例 mask (h×w)              stuff 语义分割图 (h×w)
    │                                      │
    └────────────── 推理融合 ──────────────┘
                        ▼
              Panoptic 输出（每像素: class + instance_id）
```

> **设计哲学**：实例 mask 不另建 RoI Align 分支，而是复用 **Decoder query 向量** 与 **Encoder 空间特征** 的 **双线性点积（dot-product mask）**——与 MaskFormer 后来的 mask classification 思想一脉相承。

### 10.3 Mask Head：实例 mask 如何产生

对每个匹配到 GT 的 query，从其 Decoder 输出 `h_i ∈ R^d` 预测 mask，**不**对 bbox 做 crop。

#### 10.3.1 结构

```
Decoder 输出 hs:           (B, N, d)         N=100, d=256

Mask Embedding MLP:
  hs → Linear → ReLU → Linear → ReLU → Linear
  mask_embed:              (B, N, d)         每个 query 一个 d 维 mask 向量 m_i

Encoder 空间特征（reshape）:
  memory → (B, d, h, w)   与检测共用同一组投影特征 E

逐 query 点积（官方实现用 einsum）:
  M_i = σ( einsum('bd,bdhw→bhw', m_i, E) )
  或等价: masks = torch.einsum('bqc,bchw→bqhw', mask_embed, memory_2d)

输出:
  pred_masks:              (B, N, h, w)      低分辨率，步长 32
  上采样: bilinear → (B, N, H, W)  仅推理 / 可视化；训练常在低分辨率算 loss
```

**设计要点**：实例 mask 由 **cross-attn 后的 query 向量** 与 **Encoder 空间特征** 点积得到，**无需**对 bbox 做 crop / RoI 操作；N 张 mask 一次 einsum 并行输出。

#### 10.3.2 维度推导（接 3.6 节符号）

```
mask_embed:     (B, N, d)      = (2, 100, 256)
memory_2d:      (B, d, h, w)   = (2, 256, 25, 33)

点积:
  (B, N, d) × (B, d, h, w)  →  (B, N, h, w)  = (2, 100, 25, 33)

每个 query 一张 25×33 的 coarse mask；100 个 query 共 100 张，未匹配的 query 学「空 mask」。
```

### 10.4 Stuff 分支：Semantic Segmentation Head

Things 由 query 集合覆盖；**stuff 类数量多、无实例边界**，论文在 **Encoder 输出** 上接轻量 **语义分割头**：

```
输入: memory reshape → (B, d, h, w)

Semantic Head（通常 3 层 Conv）:
  Conv 3×3, d → d
  Conv 3×3, d → d
  Conv 1×1, d → K_stuff        K_stuff = COCO stuff 类数（53）

输出: stuff_logits (B, K_stuff, h, w)

训练: 像素级 Cross-Entropy，仅对 GT 中 stuff 像素监督
      （things 区域在 stuff 标签中通常 ignore 或标为 other）
```

**为何不用 query 预测 stuff**：stuff 无实例可数性，用密集 per-pixel 分类更简单；things 与 stuff 在推理阶段再融合。

### 10.5 训练：匹配与损失

#### 10.5.1 匈牙利匹配（加入 mask 代价）

检测匹配仍基于 `(class, box)`，扩展版在代价矩阵中加入 **mask 相似度**：

```
C_match(j, i) = λ_cls · (-p_j[c_i])
              + λ_box · L1(b_j, b_i) + λ_giou · L_GIoU(b_j, b_i)
              + λ_mask · L_mask( M_j, M_i^GT )

L_mask 常用:  -Dice(M_j, M_i^GT)  或  Focal 代价的负值
              在低分辨率 (h×w) 上与 downsample 后的 GT mask 比较

默认权重（官方 segmentation 配置）:
  λ_cls=1, λ_box=5, λ_giou=2, λ_mask=1
```

匹配仍在 **no_grad** 下用匈牙利算法；**同一个 σ** 同时决定 class/box/mask 的监督对应关系。

#### 10.5.2 分割损失（仅 matched query）

```
对匹配到 GT 实例的 query:

L_mask_focal = Sigmoid Focal Loss(M_pred, M_gt)
L_mask_dice  = 1 - Dice(M_pred, M_gt)

L_seg = λ_focal · L_mask_focal + λ_dice · L_mask_dice

未匹配 query: 不参与 mask loss（与 box 回归相同，只监督匹配对）
```

Stuff 分支独立损失：

```
L_stuff = CrossEntropy(stuff_logits, stuff_gt_map)    像素级
```

总损失：

```
L = L_det_set + L_seg_matched + L_stuff + Σ L_aux（各 Decoder 层的辅助检测/分割损失）
```

#### 10.5.3 GT Mask 处理

```
COCO instance annotation → 每张实例一张二值 mask (H, W)
训练时下采样到 (h, w) 与 pred_masks 对齐:
  nearest 或 bilinear + threshold

Panoptic GT:
  things → 逐实例 mask + class，送入匈牙利匹配
  stuff  → 合并成语义图 (H, W)，每像素 stuff 类 id
```

### 10.6 推理：Panoptic 融合流程

检测与分割预测后，需要 **启发式融合** 成唯一像素标注 `(semantic_class, instance_id)`：

```
Step 1: 筛选 things 实例
  保留 score > τ（论文 panoptic 实验 τ≈0.85，高于检测的 0.7）
  按 score 降序排列 query

Step 2: 粘贴实例 mask（Painter's algorithm）
  依次把每个 instance mask 画到 panoptic 画布
  后画/高分实例覆盖先画/low 分重叠区
  每个 things 像素: (class, instance_id)

Step 3: 填充 stuff
  对未被 things 覆盖的像素，取 stuff_logits argmax 得 stuff 类
  stuff 像素: (class, instance_id=0)

Step 4: 冲突处理
  若 things mask 与 stuff 预测重叠，以 **things 实例优先**
  （things 由 query 直接监督，边界通常更准）
```

```
像素最终标签示例:

  ┌─────────────────────────────┐
  │ sky (stuff)     │ sky       │
  │ grass (stuff)   │ person#1  │
  │ grass           │ person#1  │
  │ road (stuff)    │ car#2     │
  └─────────────────────────────┘
```

### 10.7 实验结果参考（COCO Panoptic val）

| 方法 | Backbone | PQ | SQ | RQ | 说明 |
|------|----------|-----|-----|-----|------|
| **DETR** | R50 | **43.4** | 79.7 | 52.5 | 全景扩展 |
| DETR-DC5 | R50 | 44.6 | — | — | 更高分辨率 mask |

DETR 在 **things 分割（PQ_th）** 上提升明显，得益于 query 与全局 context；**stuff** 略依赖额外 semantic head，非纯 query 端到端。

### 10.8 源码对照（Segmentation 扩展）

官方 repo 中 `segmentation` 分支 / `models/segmentation.py`：

| 概念 | 模块 |
|------|------|
| Mask dot-product | `MLP` → `mask_embed` + `einsum` with `hs` and memory |
| Focal + Dice | `sigmoid_focal_loss`, `dice_loss` in `util/misc.py` |
| Stuff head | `sem_seg_head` on encoder output |
| 匹配 mask 代价 | `HungarianMatcher` 中 `cost_mask` |
| Panoptic 融合 | `panoptic.py` 推理脚本 |

**阅读顺序**：`models/detr.py`（`forward` 里 `pred_masks`）→ `SetCriterion`（`loss_masks`）→ `engine.py` panoptic 评估。

### 10.9 小结：Panoptic 扩展带来的启示

1. **Query 不仅能预测 box，还能通过点积「解码」出空间 mask**，无需 RoI 裁剪；  
2. **匈牙利匹配自然扩展到 mask**，一套 σ 对齐 class / box / mask；  
3. **Things + Stuff 职责分离**：实例用集合预测，区域用语义头，推理再融合——后来 **MaskFormer** 用统一 query 预测两类，是更彻底的「纯集合」路线；  
4. **分辨率瓶颈仍在步长 32 的 coarse mask**，DC5 与后续 Mask2Former 的多尺度特征是为弥补这一点。

---

## 11. 小结

DETR 把目标检测从「**在密集候选上做单点分类+回归**」转变为「**用固定 query 槽位做集合预测 + 二分图匹配训练**」。网络结构上，**ResNet 提特征 → Transformer Encoder 建全局上下文 → Decoder Query 通过 Cross-Attention 定位物体 → FFN 出 class/box**（§3.6 给出了逐层 Attention 张量形状）；训练上，**匈牙利匹配** 替代 anchor IoU 分配，**∅ 类** 吸收空 query，从而 **理论上摆脱 NMS**。扩展到全景分割时，**Mask Head 用 query 向量与 Encoder 特征点积出实例 mask**，stuff 由 **Semantic Head** 补充，推理阶段做 **things-stuff 融合**（§10），在同一框架内达到 COCO Panoptic 强基线。其代价是 **训练周期长、小物体与计算效率** 仍需后续 Deformable DETR 等大量工作补齐——但 as a paradigm，DETR 开启了「检测 = 集合预测 + Transformer」这一整条研究线。

---

## 参考文献

1. Carion N., et al. **End-to-End Object Detection with Transformers.** ECCV 2020.  
2. Vaswani A., et al. **Attention Is All You Need.** NeurIPS 2017.  
3. Lin T-Y., et al. **Feature Pyramid Networks for Object Detection.** CVPR 2017.  
4. Rezatofighi H., et al. **Generalized Intersection over Union.** CVPR 2019.  
5. Zhu X., et al. **Deformable DETR: Deformable Transformers for End-to-End Object Detection.** ICLR 2021.
