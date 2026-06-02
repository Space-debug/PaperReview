# DINO

> 本文档用于整理 DINO（DETR with Improved DeNoising Anchor Boxes）论文精读笔记。  
> 重点：**对比去噪训练 CDN**、**混合 Query 选择（Mixed Query Selection）**、**Look Forward Twice 框更新**，以及在 Deformable DETR + DN-DETR + DAB-DETR 基线上的 **SOTA 级收敛与精度**。  
> **想快速建立整体印象**：先读 **§1.4.1（版本体系）**、**§1.5 ~ §1.9**，再按需深入 **§3（三大创新）**。

> 前置阅读：[DETR.md](./DETR.md) → [DeformableDETR.md](./DeformableDETR.md) —— 理解集合预测、MSDeformAttn、two-stage query 选择后再读本篇。

---

## 1. 概览

### 1.1 基本信息

| 项目 | 内容 |
|------|------|
| 论文 | DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection |
| 作者/机构 | Hao Zhang, Feng Li, Shilong Liu 等（HKUST、清华、IDEA Research） |
| 发表 | ICLR 2023（arXiv 2022.03） |
| 任务 | 端到端目标检测（DETR 系 SOTA，首次在 COCO leaderboard 超越传统检测器） |
| 代码 | [IDEA-Research/DINO](https://github.com/IDEA-Research/DINO) |
| 基线 | **Deformable DETR** + **DN-DETR** 去噪 + **DAB-DETR** 动态 anchor query |

### 1.2 核心思想（一句话）

**在 Deformable DETR 的 MSDeformAttn + 两阶段 query 选择之上，用 CDN 对比去噪（正/负噪声 query）、Mixed Query Selection（位置来自 Encoder、内容仍可学习）、Look Forward Twice（层间框梯度多传一步）三项改进，在仍保持端到端集合预测的前提下，把 12 epoch 训练推到 49+ AP。**

DINO 相对「Deformable DETR + DN」的三项 **论文级** 新贡献：

| 创新 | 解决什么问题 |
|------|--------------|
| **CDN（Contrastive DeNoising）** | DN 只会「重建 GT」，不会「拒绝远 anchor」→ 加 hard negative 噪声 query，学预测 no object |
| **Mixed Query Selection** | Deformable 两阶段把 content 也绑死在 Encoder 特征上 → 只让 **anchor/位置** 来自 top-K，**content 保持可学习** |
| **Look Forward Twice（LFT）** | Deformable refine 时 **detach ref（look forward once）** → 第 i 层 box 监督还来自第 i+1 层 offset，加强 cascade |

### 1.3 整体流水线

```
Input Image
    │
    ▼
Backbone（ResNet / Swin）+ 多尺度特征（4 或 5 scale）
    │
    ▼
Deformable Encoder（MSDeformAttn ×6）→ memory
    │
    ├──────────────────────────────────────────┐
    │ Mixed Query Selection                     │
    │  Encoder 末层 top-K 特征 → 初始化 anchor (x,y,w,h)  │
    │  content query = 可学习 nn.Embedding（非 Encoder 特征）│
    └──────────────────────────────────────────┘
    │
    ▼
Decoder（6 层）× 两路 query 并行
    │
    ├── 匹配分支（900 queries）：Self-Attn → MSDeformAttn → FFN
    │       → 逐层 refine anchor（LFT）→ cls / box head
    │       → 匈牙利匹配 GT → Set Loss
    │
    └── CDN 分支（训练 only）：GT 加噪 → 正/负 denoise queries
            → 同一 Decoder 前向（拼接 query）
            → 正样本重建 GT；负样本 → background（focal loss）
    │
    ▼
推理：仅匹配分支 900 queries → 阈值过滤，无 NMS
```

### 1.4 COCO 精度参考

> **多版本说明**：DINO 官方按 **尺度数 × backbone × 训练 schedule × 预训练数据** 组合出多个 checkpoint；下文 **§1.4.1** 系统梳理。本节表为论文/官方 README 主结果摘要。

| 配置 | Backbone | Epoch | AP | AP_S | AP_M | AP_L | 说明 |
|------|----------|-------|-----|------|------|------|------|
| DN-Deformable-DETR | R50 | 12 | 43.4 | 24.8 | 46.8 | 59.4 | 直接前作 |
| **DINO-4scale** | R50 | 12 | **49.0** | **32.0** | 52.3 | 63.0 | 4 尺度，速度更快（~23 FPS） |
| **DINO-5scale** | R50 | 12 | **49.4** | **32.3** | 52.5 | 63.9 | 5 尺度，精度略高 |
| DINO-4scale | R50 | 24 | 50.4 | — | — | — | 2× schedule |
| DINO-5scale | R50 | 24 | 51.3 | 34.5 | 54.2 | 65.8 | 2× schedule |
| DINO-4scale | R50 | 36 | 50.9 | — | — | — | 3× schedule |
| DINO-5scale | R50 | 36 | 51.2 | — | — | — | 3× schedule |
| DINO-4scale | Swin-L | 12 | 56.8 | — | — | — | 需单独下载 Swin 权重 |
| DINO-5scale | Swin-L | 12 | 57.3 | — | — | — | 同上 |
| DINO-5scale | Swin-L | 36 | 58.5 | — | — | — | 更长 schedule |
| **DINO-5scale** | Swin-L | 12 | **63.2** | — | — | — | **Objects365 预训练 → COCO 微调**，val SOTA |
| DINO-5scale | Swin-L | — | **63.3** | — | — | — | 同上配置，**test-dev** |

### 1.4.1 官方配置版本体系（多版本怎么区分）

很多人说的「DINO 多个版本」，在 **同一篇检测论文 / IDEA-Research/DINO 仓库** 里，主要指下面 **四个正交维度** 的组合，而不是改动了 CDN / Mixed QS / LFT 等核心算法：

```
                    ┌──────────────────────────────────────┐
                    │  DINO 检测模型 = 固定算法 + 可换配置   │
                    └──────────────────────────────────────┘
           ┌────────────┬────────────┬────────────┬──────────────┐
           ▼            ▼            ▼            ▼              ▼
      ① 尺度数      ② Backbone   ③ Schedule   ④ 预训练数据    ⑤ Query 初始化模式
     4-scale       R50          12 epoch     ImageNet        standard（Mixed QS，默认）
     5-scale       Swin-L       24 epoch     + Objects365    no / 其他（消融用）
                   ...          36 epoch     （SwinL SOTA）
```

#### 维度 ①：4-scale vs 5-scale（最常提的版本名）

| 项目 | **DINO-4scale** | **DINO-5scale** |
|------|-----------------|-----------------|
| 特征层数 L | **4** | **5** |
| 典型 stride | 8 / 16 / 32 / 64 | **4** / 8 / 16 / 32 / 64 |
| 相对 Deformable | 与 Deformable 默认 4 尺度一致 | **多一层 stride-4 高分辨率** |
| 12 epoch R50 AP | 49.0 | **49.4**（+0.4） |
| 速度 | **~23 FPS**，更偏部署 | 计算更重，AP 略高 |
| query 数 | **900**（官方默认） | **900** |
| 算法 | CDN + Mixed QS + LFT **相同** | 相同 |

**怎么选**：复现论文主结果用 **5-scale**；要速度/算力友好用 **4-scale**，精度差距不大。

#### 维度 ②：Backbone

| Backbone | 用途 | 备注 |
|----------|------|------|
| **ResNet-50** | 论文主表、消融 Table 4 | ImageNet 预训练，脚本 `scripts/DINO_train_dist.sh` |
| **Swin-L** | 冲高 AP | 需下载 [Swin-L 22k 权重](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth)；`scripts/DINO_train_submitit_swin.sh` |

**注意**：换 backbone **不改变** DETR 系训练范式（仍 CDN + 匈牙利 + 无 NMS），只换特征提取器与 FPN 通道。

#### 维度 ③：训练 Schedule（1× / 2× / 3×）

| 名称 | epoch | 5-scale R50 AP（论文） | 说明 |
|------|-------|------------------------|------|
| **1×** | **12** | **49.4** | 论文主打「快速收敛」 |
| **2×** | **24** | **51.3** | lr drop 策略随 schedule 延长 |
| **3×** | **36** | **51.2** | 5-scale R50 收益趋缓；Swin-L 仍可从 57.3→58.5 |

官方 README 按 **12 / 24 / 36 epoch** 分别发布 checkpoint（4-scale 与 5-scale 各一套）。

#### 维度 ④：预训练数据（SOTA 配置）

| 流程 | 典型 AP | 说明 |
|------|---------|------|
| ImageNet → COCO | R50 ~49；Swin-L ~57 | 常规定义 |
| **Objects365 → COCO** | Swin-L **63.2** val / **63.3** test-dev | 论文 **leaderboard SOTA** 配置；**不是**改网络结构，是 **额外检测预训练** |

#### 维度 ⑤：`two_stage_type`（Query 初始化模式，消融 vs 默认）

代码参数 `two_stage_type` 控制 Query 从哪来——**默认 DINO 即 Mixed QS**：

| 取值 | 含义 | 对应论文概念 |
|------|------|--------------|
| **`standard`** | Encoder top-K 初始化 **anchor**，content 用 `nn.Embedding` | **Mixed Query Selection（默认 DINO）** |
| `no` | 不做 Encoder top-K，静态 query | 接近 DN-DETR / 静态 init |
| （消融）Pure two-stage | pos+content 皆来自 Encoder | Deformable two-stage，Table 4 对照行 |

**日常说的「DINO 模型」= `standard` + CDN + LFT**；Table 4 里的 Pure QS 行是 **对照实验**，不是默认发布版。

#### 官方命名与脚本对照

| 官方名称 | 典型脚本 / 配置关键字 | 12 epoch R50 AP |
|----------|----------------------|-----------------|
| DINO-4scale | `DINO_4scale.py` / `dino_4scale` | 49.0 |
| DINO-5scale | `DINO_5scale.py` / `dino_5scale` | 49.4 |
| DINO-4scale Swin-L | `DINO_train_submitit_swin.sh` | 56.8 |
| DINO-5scale Swin-L | 同上 + 5 levels | 57.3 |
| DINO-5scale Swin-L O365 | O365 预训练后再 COCO | **63.2** |

#### 容易混淆的「其他 DINO」（不是本文默认所指）

| 名称 | 与检测 DINO 关系 |
|------|------------------|
| **Grounding DINO** | 同团队后续工作：在 DINO 架构上接 **文本**，做 open-vocabulary 检测 |
| **DINOv2**（Meta） | **完全不同**：自监督视觉 foundation model，与 IDEA 这篇 **检测 DINO 无关** |
| MMDetection `dino-*-improved` | 第三方复现时的训练技巧增强版，**非原论文 Table 1 主结果** |

> 读源码 / 复现时：先确认是 **4scale 还是 5scale**、**R50 还是 Swin-L**、**12 还是 24/36 epoch**、是否 **O365 预训练**——四项对齐后再比 AP。

### 1.5 与 DETR 系差异总览

| 模块 | DETR | Deformable DETR | DN-DETR | **DINO** |
|------|------|-----------------|---------|----------|
| Attention | 全局 | MSDeformAttn | +Deformable | +Deformable |
| Query 形式 | 可学习 embedding | ref point + content | **4D 动态 anchor**（DAB） | 同 DAB |
| Query 初始化 | 静态 nn.Embedding | two-stage：pos+content 皆来自 Encoder | 静态 + DN 支路 | **Mixed**：pos 来自 Encoder，**content 可学习** |
| 训练辅助 | 无 | 无 | **DN 去噪**（单噪声重建） | **CDN**（正+负对比） |
| 框 cascade | 无 | refine + **detach**（look forward once） | +DN loss | **Look Forward Twice** |
| Query 数 | 100 | 300 | 300（+DN） | **900**（见 §1.5.1） |
| 典型 epoch | 300~500 | 50 | 12~50 | **12~24** 即 SOTA 级 |

**不变的部分**：集合预测、匈牙利匹配、无 NMS、MSDeformAttn 读特征方式。

#### 1.5.1 为何 900 个 matching queries？

论文 Appendix F.3.1 的原始设计：

```
DN-DETR:  300 decoder queries × 3 patterns = 900 槽位（Pattern 复制 query，计算量相当）
DINO:     沿用「900 槽位」设定，与 optimized 基线一致
```

**官方代码实现**（`DINO_5scale.py`）：

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_queries` | **900** | matching 分支槽位数 |
| `num_patterns` | **0** | 不启用 Pattern Embedding，**直接 900 个 query**（与论文 300×3 等价槽位，实现更简） |
| `num_select` | **300** | **推理**时从 900 预测中取 cls 分数 top-300 输出（`PostProcess`） |

```
训练: 900 queries 全部参与匈牙利匹配（与 GT 一对一）
推理: 900 queries 前向 → 按 max class score 取 top-300 → 阈值过滤 → 无 NMS

为何训练 900、推理 300？
  更多 query 提升 crowded 场景召回与匹配灵活性
  推理筛 top-300 控制输出数量与速度（与 Deformable two-stage 常用 300 一致）
```

4-scale 与 5-scale **均为 900** queries；差异在 **特征层数**，不在 query 数。

### 1.6 快速理解：三根增量柱子

DINO 在 Deformable + DAB + DN 之上，新增三根 **正交** 柱子：

```
                    ┌─────────────────────────────────┐
                    │  DINO = 强基线 + 三项训练/初始化改进 │
                    └─────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
   ① CDN 对比去噪              ② Mixed Query Selection      ③ Look Forward Twice
   「近 anchor 重建 GT          「位置来自 Encoder、           「第 i 层 box 监督
     远 anchor 学 background」    语义仍可学习」                  多看一步 Δb_{i+1}」
          │                           │                           │
   稳定早期匹配 + 抑制重复框    避免粗糙 Encoder 特征绑死 content   加强 cascade 框优化
   训练 only，推理去掉          继承 two-stage 空间先验          仍 detach 传给下一层输入
          │                           │                           │
          └───────────────────────────┴───────────────────────────┘
                                      │
              继承：MSDeformAttn + 匈牙利 + 900 queries + 12 epoch SOTA
```

| 柱子 | 解决什么 | 不改什么 |
|------|----------|----------|
| **CDN** | DN 只会「拉近」，不会「推远」→ 重复框、错配 anchor | 集合预测框架 |
| **Mixed QS** | Pure two-stage 把 content 也锁在 Encoder 特征上 | MSDeformAttn 读特征方式 |
| **LFT** | Look forward once 时第 i 层只吃本层 box loss | ref 仍 detach 进下一层 |

### 1.7 一张图的 Walkthrough（2 个 GT + CDN）

假设 5-scale 配置：**900 matching queries**，图上有 **2 个 GT**（猫、狗），`dn_number=1`（每组 CDN 为每个 GT 生成正+负 2 个 query）。

#### 训练时（一个 iteration）

```
Step 0  输入
        图像 + GT = {猫框, 狗框}

Step 1  Mixed Query Selection
        Encoder MSDeformAttn ×6 → memory
        memory 上 dense cls/bbox head → top-900 位置
        用 top-K 预测框 init **anchor (x,y,w,h)**（positional query）
        content query = 可学习 nn.Embedding(900, 256)，**与图像无关**

Step 2  构造 CDN queries（prepare_for_cdn）
        对「猫」: 正 query = 猫框 + 小噪声(λ1 内) → 监督重建猫
                  负 query = 猫框 + 中等噪声(λ1~λ2) → 监督 background
        对「狗」: 同上 → 共 4 个 CDN queries
        concat → Decoder 输入 = [4 CDN | 900 matching] = 904 queries

Step 3  Decoder 6 层（两路并行）
        Self-Attn: CDN 与 matching 有 mask（matching 不能抄 CDN 答案）
        MSDeformAttn: 全体 query 读多尺度 memory
        逐层 refine anchor；LFT 让第 i 层还吃第 i+1 层 box 监督

Step 4  双分支 Loss
        CDN 支路: 2 正 → L1+GIoU+focal(猫/狗)；2 负 → focal(∅)，**无匈牙利**
        Matching: 900 queries 匈牙利 ↔ 2 GT → 2 匹配 + 898 ∅
        L_total = L_match + L_cdn + 各层 aux

Step 5  12 epoch 内
        CDN 提供大量「有明确标签」的梯度 → 匹配更快稳定
        Mixed QS → matching query 第一步就在物体附近采特征
        LFT → 框 cascade 收敛更紧
```

#### 推理时

```
Step 1  去掉 CDN，仅 900 matching queries 前向
Step 2  最后一层 refined anchor + cls head → (score, class, box)
Step 3  score > 0.3 → 输出，无 NMS
        （PostProcess 先从 900 中取 cls top-300，见 §1.5.1）
```

### 1.8 常见困惑 FAQ（速查）

| 问题 | 简短回答 |
|------|----------|
| **DINO 相对 DN-DETR 改了什么？** | DN → **CDN**（加负样本）；Deformable two-stage → **Mixed QS**；refine → **LFT**。详见 §3。 |
| **CDN 推理要用吗？** | **不用**；仅训练拼接进 Decoder，推理只保留 900 matching queries。 |
| **Mixed 和 Pure two-stage 差在哪？** | Pure：pos+content 都来自 Encoder top-K；Mixed：**只有 anchor/位置** 来自 Encoder，content 仍是 `nn.Embedding`。 |
| **LFT 和 detach 矛盾吗？** | 不矛盾：`b_i'` 参与 LFT 第二跳反传；**传给第 i+1 层输入** 的仍是 `detach(b_i')`。 |
| **为何 900 queries？** | 论文设计 300×3=900 槽位；代码 `num_queries=900`，`num_patterns=0`；推理 `num_select=300` 取 top-300。见 **§1.5.1**。 |
| **还要编译 CUDA ops 吗？** | **要**；继承 Deformable MSDeformAttn，见 [DeformableDETR.md §11](./DeformableDETR.md)。 |
| **DINO 有多个版本吗？** | **有**；按 **4/5-scale × R50/Swin-L × 12/24/36 epoch × 是否 O365 预训练** 组合；算法核心（CDN+Mixed QS+LFT）不变。详见 **§1.4.1**。 |
| **4-scale 和 5-scale 差什么？** | 5-scale **多一层 stride-4** 特征；12 epoch R50：49.0 vs 49.4 AP；4-scale 更快（~23 FPS）。 |
| **和 Grounding DINO 区别？** | Grounding DINO 在 DINO 上接 **文本** 做 open-set；本文档是原版 **闭集检测** DINO。 |

> 更多细节见 **§10 FAQ** 与 **§11 源码**。

### 1.9 核心原理深讲（入口）

§1.5~§1.7 建立直觉后，用 **§9 核心原理深讲** 做完整串讲（协同机制、12 epoch 原因、演进位置、局限）。此处只保留三问索引：

| 问题 | 跳转 |
|------|------|
| 训练为何仍不稳？ | §3.1 CDN |
| 两阶段 init 为何不够？ | §3.2 Mixed QS |
| cascade 框为何弱？ | §3.3 LFT |
| 三项如何协同 / 为何 12 epoch？ | **§9.1 ~ §9.2** |
| 工程局限？ | **§7.2** |

---

## 2. 前置基础：DINO 站在哪些肩膀

### 2.1 DAB-DETR：Query = 4D 动态 Anchor

```
每个 decoder query 显式拆成:
  positional query = 4D anchor box (x, y, w, h)   逐层 refine
  content query    = 256-d 语义向量                 分类依据

相对 DETR 纯 embedding：anchor 与框回归统一，层间 cascade 自然
DINO 完全沿用此 formulation
```

### 2.2 DN-DETR：去噪训练稳定匹配

```
问题: 二分图匹配在训练早期不稳定 → query specialization 慢

做法: 把 GT 加噪后作为额外 decoder 输入（DN queries）
  噪声 |Δx|<λw/2, |Δy|<λh/2, |Δw|<λw, |Δh|<λh
  监督: 重建原始 GT 类别 + 框（L1 + GIoU + focal）

DN queries 不参与匈牙利匹配，走独立 DN loss
推理时去掉 DN 支路

DINO 继承 DN 框架，升级为 CDN（§3.1）
```

### 2.3 Deformable DETR：MSDeformAttn + Query Selection

```
MSDeformAttn: 见 DeformableDETR.md §3

Query Selection（论文称 two-stage）:
  Encoder memory 上 dense 预测 → top-K 位置
  用 Encoder 特征生成 decoder 的 positional + content queries
  并 init reference boxes

Iterative refine + detach ref = Look Forward Once（§3.3 对比 LFT）

DINO 采用 query selection，但改为 Mixed（§3.2）
```

---

## 3. 三大核心创新（原理细讲）

### 3.0 并行 Walkthrough：相对 DN / Deformable 差在哪

同一训练步、同一 GT「猫框」，三种做法对照：

```
┌─────────────────────────────────────────────────────────────────────────┐
│ (A) DN-DETR：单档去噪，只教「重建」                                        │
├─────────────────────────────────────────────────────────────────────────┤
│  GT 猫框 + 噪声(λ 内) → 1 个 DN query → 监督: 类=猫, 框=原 GT           │
│  未教: 稍远的 anchor 应输出 ∅ → 多个 query 仍可能挤在猫附近                │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ (B) CDN（DINO）：两档噪声，正+负对比                                       │
├─────────────────────────────────────────────────────────────────────────┤
│  正: 噪声 < λ1  → 重建猫                                                 │
│  负: λ1 < 噪声 < λ2 → focal → ∅（hard negative 靠近 GT 更有教益）         │
│  同一 GT 同时有「该留」与「该拒」两种 anchor → 减轻重复框                  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ (C) Query 初始化：Pure two-stage vs Mixed（DINO）                          │
├─────────────────────────────────────────────────────────────────────────┤
│  Deformable Pure:  top-K memory 特征 → Linear → pos **和** content 皆来自特征 │
│  DINO Mixed:       top-K → **仅** init anchor/reference                   │
│                    content = nn.Embedding(900, d)  第一层 Decoder 再学语义  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ (D) 框 cascade：Look Forward Once vs Twice                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  Once:  loss 只监督 Update(b_{i-1}, Δb_i)；b_i=detach 传下一层              │
│  Twice: 额外监督 Update(b_i', Δb_{i+1}) → 第 i 层参数也受「下一步」框质量驱动   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 3.1 CDN：Contrastive DeNoising Training

#### 3.1.1 DN 的不足

```
DN-DETR 只教模型: 「噪声不太大的 anchor → 重建 GT」

未教: 「离 GT 较远的 anchor → 应预测 no object / background」

后果（论文 Fig.8）:
  多个 anchor 挤在同一物体附近 → 匹配混淆 → 重复框
  较远的错误 anchor 偶尔被匹配 → 低质量框
```

#### 3.1.2 CDN 机制：同心方框 + 正/负样本

对 **每个 GT box**，生成 **两个** denoise query（一组 CDN）：

```
                    外边界: 噪声尺度 λ2
              ┌─────────────────────────┐
              │    负样本区域            │
              │   ┌───────────────┐     │
              │   │ 正样本区域 λ1 │     │
              │   │   ┌─────┐     │     │
              │   │   │ GT  │     │     │
              │   │   └─────┘     │     │
              │   └───────────────┘     │
              └─────────────────────────┘

正 query:  噪声 < λ1  →  监督重建 GT（类 + 框）
负 query:  λ1 < 噪声 < λ2  →  监督为 background（focal loss）

同一 GT 同时出现「近」与「远」两种 anchor → 对比学习 → 学会区分
```

**超参**：`λ1 < λ2`；λ2 通常较小，hard negative 靠近 GT 更有教益。

#### 3.1.3 与 DN 的并行对照

| | DN-DETR | CDN（DINO） |
|--|---------|-------------|
| 噪声档 | 1 档（\|Δ\|<λ） | **2 档** λ1、λ2 |
| 每 GT query 数 | 1（重建） | **2**（正 + 负） |
| 负样本 | 无 | **有**，预测 background |
| 分类损失 | focal（重建类） | focal + **负样本 focal→∅** |
| 推理 | 无 DN 支路 | 无 CDN 支路 |

#### 3.1.4 训练时 query 如何拼接

```
设一张图 n 个 GT，DN number = g 组 CDN:

CDN queries 数 ≈ 2 × n × g（正负各半）
匹配 queries 数 = 900（5-scale 配置）

Decoder 输入 = concat(CDN queries, matching queries)
Self-Attn / MSDeformAttn 对 **全体** query 计算

CDN 部分:
  已知 GT 对应关系，**不跑匈牙利**，直接算重建/background loss
  通过 attn mask 控制 CDN 与 matching queries 的可见性（防信息泄漏）

Matching 部分:
  照常匈牙利 + Set Loss
```

#### 3.1.5 指标 ATD 与效果

论文定义 **ATD(k)**：匹配上的 anchor 与 GT 的 L1 距离 top-k 均值。CDN 使 **小物体** anchor 更贴近 GT（Fig.4），12 epoch AP_S **+7.5**（相对 DN-Deformable-DETR）。

#### 3.1.6 CDN 超参与噪声公式（官方默认）

**配置文件默认值**（`DINO_4scale.py` / `DINO_5scale.py`）：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `use_dn` | `True` | 启用 CDN 支路 |
| `dn_number` | **100** | 目标 CDN query **预算**（见下方动态分组） |
| `dn_box_noise_scale` | **0.4** | 框噪声幅度系数（对应论文 λ） |
| `dn_label_noise_ratio` | **0.5** | 标签噪声比例 |
| `dn_labelbook_size` | 91 | 与 COCO 类数一致 |

**论文框噪声约束**（对 GT 框 `(x,y,w,h)`，加噪后 `(x',y',w',h')`）：

```
|Δx| < λ · w/2 ,  |Δy| < λ · h/2
|Δw| < λ · w     ,  |Δh| < λ · h

CDN 两档:
  正样本:  λ = λ1  （较小，如 scale=0.4 的内档）
  负样本:  λ1 < 噪声 < λ2  （外档 hard negative）
```

**官方代码如何实现正负两档**（`prepare_for_cdn`）：

```
1. 先把 GT 框转为 xyxy，以 w/2、h/2 为各方向最大扰动基准 diff
2. rand_sign ∈ {-1,+1}，rand_part ∈ [0,1] 随机
3. 正样本 index = positive_idx:
     known_bbox_noised = bbox_xyxy + rand_sign * rand_part * diff * box_noise_scale
4. 负样本 index = negative_idx（= positive_idx + len(boxes)）:
     rand_part += 1.0   →  rand_part ∈ [1,2]
     → 噪声幅度落在「内档之外、外档之内」→ 实现 CDN 对比
5. 转回 cxcywh，clamp 到 [0,1]
```

**标签噪声**：以概率 `label_noise_ratio × 0.5` 把正 CDN query 的类别随机换成其他类（仍监督框），增强分类鲁棒性。

**自适应 DN 组数**（论文 Appendix F.1，`prepare_for_cdn` 内）：

```
配置 dn_number=100，进入函数后先 ×2（正负各一套）

若 dn_number >= 100:
  dn_number = dn_number // (max(GT数 per image) × 2)

目的: 固定 CDN query **总量** 预算，按每张图 GT 数 **动态** 决定组数
      避免 COCO 单图 1~80 个 GT 时 pad 浪费显存

例: 图上有 10 个 GT，dn_number 配置 100
    → 100×2=200 → 200//(10×2)=10 组
    → 每 GT 仍有正+负 2 个 query，总 CDN pad ≈ 10×10×2=200
```

> CDN loss 权重与 matching **相同**（`loss_ce_dn`、`loss_bbox_dn`、`loss_giou_dn` 系数均等于 matching 分支）。

---

### 3.2 Mixed Query Selection

#### 3.2.1 三种 Query 初始化（Fig.5）

```
(a) Static（DETR / DN-DETR）:
    pos + content 皆为可学习 embedding，与具体图像无关

(b) Pure Query Selection（Deformable DETR two-stage）:
    Encoder top-K 特征 → Linear → **pos + content 皆来自特征**
    reference box 由 aux bbox head 预测

(c) Mixed Query Selection（DINO）:
    Encoder top-K → 只 init **anchor / positional query**
    content query = **保持可学习 embedding**（与 DN 分支一致）
```

#### 3.2.2 为何 Mixed 优于 Pure

```
Pure 的问题:
  Encoder 特征未经 Decoder 精炼，可能:
    - 一个 cell 含多个物体
    - 只覆盖物体一部分
  把 content 也绑死 → 语义模糊，误导 Decoder

Mixed 的做法:
  pos/anchor 用 Encoder 空间先验（「从哪看」）
  content 仍数据驱动学习（「是什么」）
  第一层 Decoder 在 **好位置** 上 pooling **自学习语义**

消融（Table 4, 12 epoch）:
  Pure query selection:  46.5 AP
  Mixed query selection: 47.0 AP  (+0.5)
  小物体 AP_S: 29.6 → 31.1
```

#### 3.2.3 与 Deformable DETR two-stage 代码差异（概念）

```
Deformable DETR two-stage:
  topk propos → pos_trans + split → tgt 与 query_pos 都来自 memory

DINO mixed:
  topk → 只用于 anchor/reference 初始化
  self.tgt_embed = nn.Embedding(num_queries, d)   可学习，与图像无关
  query_pos 来自 anchor 坐标编码（DAB sine / MLP）
```

---

### 3.3 Look Forward Twice（LFT）

#### 3.3.1 Look Forward Once（Deformable DETR）

```
Decoder 第 i 层:
  Δb_i = bbox_embed[i](hs_i)
  b_i' = Update(b_{i-1}, Δb_i)
  b_i  = detach(b_i')              ← 传给下一层前截断梯度
  辅助 loss 只监督 b_i^{pred} = Update(b_{i-1}, Δb_i)

问题: 第 i 层参数只被 **本层 box loss** 优化
      更准的 Δb_{i+1} 无法帮助修正第 i 层的 **初始框 b_{i-1}**
```

#### 3.3.2 Look Forward Twice 公式

```
Δb_i   = Layer_i(b_{i-1})
b_i'   = Update(b_{i-1}, Δb_i)        # 未 detach 版
b_i    = Detach(b_i')                 # 仍 detach 传给 i+1 层输入
b_i^{pred} = Update(b_{i-1}, Δb_i)    # 本层预测（同 Once）

关键新增 — 用下一层 offset 再更新一次:
b_{i+1}^{pred} = Update(b_i', Δb_{i+1})   # 注意用 b_i' 而非 b_i

Loss: 第 i 层 bbox head 同时受到:
  - b_i^{pred}  的 GT 监督
  - b_{i+1}^{pred} 的 GT 监督（梯度穿过 Δb_{i+1} 回到 Layer_i 相关参数）

效果: 优化 Δb_i 时同时考虑「这一步 + 下一步」能否 jointly 得到好框
```

#### 3.3.3 图示对比

```
Look Forward Once:                Look Forward Twice:

  b_{i-1} ──► Layer_i ──► Δb_i     b_{i-1} ──► Layer_i ──► Δb_i
                │                         │
                ▼                         ├─► b_i' ──► Layer_{i+1} ──► Δb_{i+1}
            Update → b_i                  │              │
                │                         │              ▼
            detach                        │         Update(b_i', Δb_{i+1}) = b_{i+1}^{pred}
                ▼                         │              │
            下一层                         └─► loss(b_i^{pred}) + loss(b_{i+1}^{pred})
                                              （i 层参数也吃 i+1 预测的损失）
```

#### 3.3.4 消融

Table 4 Row 4→5：Mixed QS 47.0 → +LFT 47.4 AP（+0.4）。

#### 3.3.5 最后一层与 LFT 边界（Appendix I）

**配置开关**：`use_detached_boxes_dec_out = False`（默认）即启用 LFT；为 `True` 时退化为 look forward once。

**Decoder 内 `ref_points` 收集**（`deformable_transformer.py`）：

```
每层 refine 后:
  if use_detached_boxes_dec_out:
      ref_points.append(reference_points)           # detach 后的 ref，给下一层输入
  else:  # LFT
      ref_points.append(new_reference_points)       # 未 detach，供 loss 用 b_i' 做第二跳
  reference_points = new_reference_points.detach()  # 传给下一层输入仍 detach
```

**6 层 aux box loss 与 LFT 的对应**：

| 层 i | LFT 第二跳监督 | 说明 |
|------|----------------|------|
| 0 ~ 4 | 有 `b_{i+1}^{pred} = Update(b_i', Δb_{i+1})` | 第 i 层 bbox head 受本层 + 下一层预测监督 |
| **5（最后一层）** | **无 i+1** | 仅 `b_5^{pred}`；Appendix I：最后一层 LFT 在 factor 2（梯度回传）上占优 |

**为何不 Look Forward 3/4 次？** 论文 Table 12：LF3/LF4 整体 AP 低于 LFT（48.3/48.2 vs 49.0）——多看层梯度难收敛，**2 步是精度与稳定性的折中**。

---

## 4. 完整训练：双分支 Loss 与推理

### 4.1 总损失构成

```
L_total = L_match + L_cdn + L_aux + L_interm + L_enc
```

**Matching 分支**（900 queries，匈牙利匹配）：

| 损失项 | 权重（官方默认） | 说明 |
|--------|------------------|------|
| `loss_ce` | **1.0** | sigmoid focal loss，γ=2，α=`focal_alpha=0.25` |
| `loss_bbox` | **5.0** | L1，归一化 cxcywh |
| `loss_giou` | **2.0** | 1 − GIoU |
| 每层 aux | 同上，键名加 `_0`…`_4` | 6 层 Decoder 中间输出 |

**匈牙利匹配代价**（与 loss 权重 **不同**，optimized 基线关键改动）：

| 代价项 | 权重 |
|--------|------|
| `set_cost_class` | **2.0** |
| `set_cost_bbox` | **5.0** |
| `set_cost_giou` | **2.0** |

> 原 DN-Deformable-DETR 匹配与 loss 同权；DINO optimized 基线把 **matcher 分类权 2.0、loss 分类权 1.0** 分开，提升匹配质量。

**CDN 分支**（训练 only，`SetCriterion` 内 `dn_pos_idx` / `dn_neg_idx`）：

| 样本 | 监督 |
|------|------|
| 正 CDN query | `loss_ce_dn` + `loss_bbox_dn` + `loss_giou_dn`（权重同 matching） |
| 负 CDN query | 仅 `loss_ce_dn` → **background（∅）** |
| 各层 aux CDN | 键名 `*_dn_0`…`*_dn_4` |

**Encoder 中间监督**：

| 项 | 权重 |
|----|------|
| `loss_*_interm` | × `interm_loss_coef=1.0` |
| `loss_*_enc` | × `enc_loss_coef=1.0` |

```
L_match（与 DETR 相同逻辑）:
  对 900 个 matching queries 匈牙利匹配
  L_cls (focal) + L_box (L1 + GIoU)
  每层 Decoder aux loss（6 层）；LFT 时 ref_points 用未 detach 版参与 box 回归链

L_cdn（训练 only）:
  正 denoise queries:  L1 + GIoU + focal → GT 类/框
  负 denoise queries:  focal → background / no object
  不经过匈牙利
```

### 4.2 CDN 的 attn mask（防泄漏）

CDN queries **已知 GT 答案**，matching queries 不能通过 Self-Attn「抄」CDN。

**张量布局**（`prepare_for_cdn` 输出，CDN 在前）：

```
Decoder 输入顺序:  [ CDN pad (pad_size) | matching (900) ]
索引:               [ 0 … pad_size-1     | pad_size … pad_size+899 ]

attn_mask 形状: (pad_size + 900, pad_size + 900)
  True  = 禁止 attend
  False = 允许 attend
```

**三条 block 规则**（对应源码）：

```
规则 1 — matching 不可看 CDN（核心防泄漏）:
  attn_mask[pad_size:, :pad_size] = True
  → 所有 matching query 不能 attend 任何 CDN token

规则 2 — 不同 CDN 组互不可见:
  将 pad 按 dn_number 组划分，每组占 single_pad×2 槽（正+负）
  组 i 的 query 不能 attend 组 j≠i 的 CDN token

规则 3 — 组内 CDN 可见（同组正/负可交互）:
  组内 attn_mask 为 False（允许 self-attn 协调）
```

**示意**（2 组 CDN + 900 matching，pad_size=8 简化）：

```
         CDN₀  CDN₁  match(900)
CDN₀      ✓     ✗      ✗
CDN₁      ✗     ✓      ✗
match     ✗     ✗      ✓     ← matching 之间仍全连接（Self-Attn 去重）

✓=可见  ✗=mask（True）
```

**Cross-Attn（MSDeformAttn）**：CDN 与 matching **均可** attend Encoder memory（图像特征），泄漏风险只在 **CDN→matching 的 Self-Attn**。

> 实现细节与逐步 Walkthrough 见 **§11.1**。

### 4.3 推理

```
去掉 CDN 支路，仅 900 matching queries 前向
MSDeformAttn + 最后一层 refined anchor → cls / box
PostProcess: 900 预测中取 cls max score 的 top-300（num_select=300）
score > threshold（默认约 0.3）→ 无 NMS

与 DETR / Deformable 相同：端到端集合输出
```

---

## 5. 相对 Deformable DETR 的模块级对照

| 阶段 | Deformable DETR | DINO |
|------|-----------------|------|
| Backbone | 4-scale FPN | 4 或 **5-scale** |
| Encoder | MSDeformAttn ×6 | 同 |
| Query init | two-stage：pos+content 来自 Encoder | **Mixed**：pos 来自 Encoder，**content 可学习** |
| Decoder | Self-Attn + MSDeformAttn + refine(detach) | + **CDN 拼接** + **LFT** |
| 训练辅助 | 无 | **CDN 正负去噪** |
| Loss | Set + aux | Set + aux + **CDN** |
| Queries | 300 | **900** |

---

## 6. 消融与贡献量化（12 epoch, R50, 4-scale）

### 6.1 Optimized DN 基线是什么？（Table 4 Row 1→2）

消融表 Row 1「DN-DETR 43.4 AP」是 **原始 DN-Deformable-DETR**；Row 2「Optimized DN 44.9 AP」是 DINO 团队在 **尚未加入 CDN / Mixed QS / LFT** 前整理的 **强基线**（论文 Appendix A / F.3.3），相对原版有三处关键工程改动：

| # | 改动 | 原版 DN-Deformable-DETR | Optimized 基线 |
|---|------|-------------------------|----------------|
| 1 | **Decoder Attention** | Encoder 用 Deformable，Decoder 仍 **dense** | Encoder + Decoder **均 MSDeformAttn** |
| 2 | **Query 数** | 300 | **900**（300×3 槽位，见 §1.5.1） |
| 3 | **Matcher vs Loss 权重** | 与 DETR 相同 | matcher cls **2.0**，loss cls **1.0** |
| 4 | **Dropout** | 有 | **0** |
| 5 | **Encoder MSDeformAttn 初始化** | 未正确按 Deformable 方式 init | **修复 init**（小物体 AP_S 受益） |
| 6 | **Prediction head** | 每层 **不共享** head | **共享** bbox/cls head（参数 −1M，精度略升） |
| 7 | **DN 组数** | 固定 5~10 组，pad 浪费 | **自适应 DN 组**（Appendix F.1，§3.1.6） |

```
43.4 (原始 DN) → 44.9 (Optimized DN) → 46.5 (+Pure QS) → … → 47.9 (完整 DINO)
                 ↑ 工程优化            ↑ 算法组件逐项叠加
```

> Table 4 的 CDN / Mixed QS / LFT 增量，均相对 **44.9 的 Optimized DN** 或 **46.5 的 Strong baseline（+Pure QS）** 报告。

| Row | QS | CDN | LFT | AP | AP_S |
|-----|----|-----|-----|-----|------|
| DN-DETR 基线 | — | — | — | 43.4 | 24.8 |
| Optimized DN | — | — | — | 44.9 | 26.9 |
| + Pure QS | Pure | — | — | 46.5 | 29.6 |
| + **Mixed QS** | Mixed | — | — | 47.0 | 31.1 |
| + **LFT** | Mixed | — | ✓ | 47.4 | 29.9 |
| **DINO** | Mixed | ✓ | ✓ | **47.9** | **31.2** |

三项创新累计约 **+3.0 AP**（相对 optimized DN-DETR 44.9）；Mixed QS、CDN、LFT 均有独立贡献。

---

## 7. 训练配置要点

> 不同 **官方版本** 的差异见 **§1.4.1**；本节为各版本 **共用** 的训练超参。

| 项目 | 典型设置 |
|------|----------|
| 优化器 | AdamW |
| backbone LR | 1e-5 |
| 其余 LR | 1e-4 |
| batch size | 32（2×8 或类似） |
| 主 schedule | **12 / 24 / 36 epoch**（1× / 2× / 3×） |
| query 数 | 900（5-scale）；4-scale 亦用 900 queries |
| CDN | `dn_number` 多组；`dn_box_noise_scale` λ1/λ2 | 见 **§3.1.6** |
| MSDeformAttn | 同 Deformable；需编译 CUDA ops | 见 [DeformableDETR.md §11](./DeformableDETR.md) |
| lr schedule | 12ep: drop@11；24ep: drop@20；36ep: drop@30 | ×0.1 |
| `num_select` | 推理 top-**300** / 训练 **900** | §1.5.1 |

### 7.2 局限与工程注意

| 局限 / 代价 | 说明 |
|-------------|------|
| **CDN 动态显存** | CDN pad 随 batch 内最大 GT 数变化；GT 多的图 CDN query 更多，显存与 Self-Attn O(Q²) 上升 |
| **dn_number 饱和** | 配置 100 后增益有限（论文 Table 9）；超过约 100 收益变小甚至略降 |
| **仍可能重复框** | CDN 减轻但未消除；推理无 NMS，偶发多 query 框同一物体 |
| **4-scale 稳定性** | 官方 README：R50 4-scale AP 可能波动 **~0.4** |
| **5-scale 算力** | GFLOPS 860 vs 4-scale 279（论文 Table 2）；FPS 10 vs 24 |
| **CUDA 依赖** | MSDeformAttn 必须编译；无 CUDA 难以高效训练 |
| **LFT 非越深越好** | LF3/LF4 低于 LFT（§3.3.5）；多看层梯度难收敛 |
| **闭集检测** | 原版 DINO 不含文本；open-set 见 Grounding DINO |

### 7.3 数据增强

与 DETR / Deformable DETR **相同**（论文 Appendix F.3.4）：

```
训练:
  RandomResize — 短边 random ∈ [480, 800]，长边 ≤ 1333
  RandomSizeCrop + RandomHorizontalFlip（继承 DETR pipeline）

Swin-L + O365 预训练后 COCO 微调:
  更大尺度 — 短边 [720, 1200]，长边 ≤ 2000（冲 leaderboard 配置）
```

---

## 8. 维度推导（R50，query=900）

### 8.1 4-scale vs 5-scale 对照

| 项目 | **4-scale**（`DINO_4scale.py`） | **5-scale**（`DINO_5scale.py`） |
|------|-------------------------------|--------------------------------|
| `num_feature_levels` | **4** | **5** |
| `return_interm_indices` | `[1,2,3]`（C3~C5） | `[0,1,2,3]`（C2~C5 + 额外层） |
| 典型 stride | 8 / 16 / 32 / 64 | **4** / 8 / 16 / 32 / 64 |
| `num_queries` | 900 | 900 |
| batch size（默认） | 2 | 1 |
| 12ep R50 AP | 49.0 | 49.4 |
| GFLOPS / FPS | ~279 / **~24** | ~860 / **~10** |

5-scale **多一层 stride-4** → 序列 S 更长、小物体 AP_S 通常更高，但算力显著增加。

### 8.2 5-scale 张量形状（单张图 batch=1，短边 800）

| 阶段 | 张量 | 形状 | 说明 |
|------|------|------|------|
| 输入 | `image` | (1, 3, H, W) | 短边 800，长边 ≤1333 |
| Backbone | 多尺度特征 | 5 层 stride 4/8/16/32/64 | 较 Deformable 4-scale 多一层高分辨率 |
| Encoder 输入 | flatten tokens | (1, S, 256) | S = Σ H_l×W_l，5-scale 时 S 更大 |
| Encoder 输出 | `memory` | (1, S, 256) | 6 层 MSDeformAttn |
| Mixed QS | top-K index | (1, 900) | Encoder 末层 cls 分数 top-900 |
| | init `refpoints` | (1, 900, 4) | 来自 aux bbox，归一化 cxcywh |
| | `tgt` (content) | (900, 1, 256) | **nn.Embedding**，与 batch 广播 |
| CDN (train) | `dn_queries` | (1, 2×n×g, 256) | n=GT 数，g=dn_number；正负各半 |
| | 拼接后 query | (1, 2×n×g+900, 256) | CDN 在前或按实现 concat |
| Decoder 层 i | `hs[i]` | (1, 2×n×g+900, 256) | 6 层 hidden |
| | `ref_points[i]` | (1, 2×n×g+900, 4) | 逐层 refine + detach 输入 |
| 预测 | cls logits | (1, 900, K+1) | 仅 matching 部分参与匈牙利 |
| | boxes | (1, 900, 4) | 最后一层 refined box |

**CDN Self-Attn mask（概念）**：详见 **§4.2**；`attn_mask` 形状 `(pad_size+900, pad_size+900)`。

**LFT 与 aux loss**：6 层 Decoder 每层对 matching 算 cls/box aux；`use_detached_boxes_dec_out=False` 时 `ref_points` 存未 detach 版供 LFT（§3.3.5）。

### 8.3 4-scale 相对 5-scale 的 S 变化（量级）

```
设输入短边 800、长边 1066 量级:

4-scale flatten 后 S ≈ Σ(H_l×W_l)，约 1.5~2×10⁴ 量级
5-scale 多 stride-4 层 → S 再增约 30~40%（高分辨率 token 多）

Encoder/Decoder MSDeformAttn 复杂度 ∝ S（Encoder）或 ∝ Q（Decoder self-attn）
→ 5-scale 主要贵在 Encoder 序列更长 + 多一层 value
```

---

## 9. 核心原理深讲

### 9.1 三项创新如何协同

```
Mixed Query Selection     →  matching queries 起点更准（空间）
CDN                       →  训练早中期匹配更稳 + 学会拒绝坏 anchor
Look Forward Twice        →  cascade 框参数优化更充分
MSDeformAttn（继承）      →  读特征便宜、可 multi-scale
匈牙利 Set Loss（继承）   →  推理仍无 NMS

CDN 解决「训练动态」；Mixed QS 解决「初始化」；LFT 解决「层间 cascade」
三者正交，可叠加
```

### 9.2 为何 12 epoch 就够

```
Deformable DETR 50 epoch → 44.5 AP 的主要瓶颈:
  ① 匹配不稳定（DN/CDN 解决）
  ② content/pos 初始化差（Mixed QS 解决）
  ③ cascade 监督弱（LFT 解决）

DINO 12 epoch → 49.4 AP:
  CDN 让 encoder-decoder 早期即有大量 **有明确标签** 的 query 梯度
  Mixed QS 让 matching query 第一步就在有意义的位置采特征
  LFT 加深 box head 监督
  → AP 曲线前 12 epoch 斜率远大于 Deformable/DN（Fig.7）
```

### 9.3 DINO 在 DETR 演进中的位置

```
DETR → Deformable（MSDeformAttn + 多尺度 + two-stage）
     → DAB（4D anchor query）
     → DN（去噪稳定匹配）
     → DINO（CDN + Mixed QS + LFT）  ← 首个 COCO leaderboard SOTA 的 DETR 系
     → Grounding DINO / Mask DINO 等（检测+语言 / 分割）
```

> **注意**：Meta **DINOv2** 是自监督视觉模型，与本文检测 DINO **无关**。

### 9.4 信息流总图

```
  Image → Backbone(4/5 scale) → Encoder(MSDeformAttn) → memory
                │
                ├─ top-900 → init anchor only (Mixed QS)
                ├─ CDN queries (train only) ──┐
                └─ 900 learnable content ─────┼→ Decoder ×6 (Self + MSDeform + LFT)
                                              │
        L_match + L_cdn + L_aux + L_interm + L_enc
        推理: 去 CDN → 900 前向 → top-300 → 阈值
```

### 9.5 局限与工程代价

详见 **§7.2**（CDN 显存、4-scale 波动、5-scale 算力、无 NMS 重复框等）。

---

## 10. 常见困惑 FAQ

| 问题 | 回答 |
|------|------|
| **DINO 和 DN-DETR 关系？** | DINO **包含并升级** DN：DN 是单噪声重建；CDN 加负样本对比。代码里 `prepare_for_cdn` 实现 CDN。 |
| **Mixed QS 和 Deformable two-stage？** | 同用 Encoder top-K；Deformable **全用** memory 特征；DINO **只用** 位置/anchor，content 仍 `nn.Embedding`。 |
| **LFT 和 detach 矛盾吗？** | 不矛盾：`b_i'` 未 detach 参与 LFT 第二跳；传给 **下一层输入** 的仍是 `detach(b_i')`，稳定 recurrent。 |
| **900 queries 为何这么多？** | 论文 300×3=900 槽位；代码 `num_queries=900`；推理 `num_select=300`。见 **§1.5.1**。 |
| **CDN attn mask 干什么？** | 阻止 matching attend CDN；详见 **§4.2** 与 **§11.1**。 |
| **Optimized DN 是什么？** | Table 4 强基线：Decoder MSDeform + 900 query + matcher/loss 分权等，见 **§6.1**。 |
| **推理输出 300 还是 900？** | 前向 900；`PostProcess` 取 top-**300** 再阈值过滤。 |
| **推理要用 CDN 吗？** | **不用**；CDN 仅训练，推理只 900 matching queries。 |
| **还要编译 MSDeformAttn CUDA 吗？** | **要**；见 [DeformableDETR.md §11](./DeformableDETR.md)。 |
| **与 Grounding DINO 区别？** | Grounding DINO 在 DINO 上接 **文本** 做 open-set；本文是闭集检测 DINO。 |
| **DINO 有几个官方版本？** | 见 **§1.4.1**：4/5-scale、R50/Swin-L、12/24/36 epoch、O365 预训练等。 |
| **默认「DINO」指哪个？** | **DINO-5scale + R50 + 12 epoch + `two_stage_type=standard`**。 |
| **LFT 开关在哪？** | `use_detached_boxes_dec_out=False`（默认）= LFT；`True` = look forward once。 |

---

## 11. 源码阅读顺序

### 11.1 `prepare_for_cdn` 逐步 Walkthrough

文件：`models/dino/dn_components.py`

```
Step 1  解析参数
        targets, dn_number, label_noise_ratio, box_noise_scale = dn_args
        dn_number *= 2  （正负两套）
        若 dn_number >= 100: dn_number //= (max_gt_per_image * 2)

Step 2  收集 GT
        labels, boxes (cxcywh), batch_idx 拼成 flat 张量
        known_indice = 所有 GT 索引，repeat 2*dn_number 次

Step 3  标签噪声（可选）
        以 prob = label_noise_ratio*0.5 随机替换部分 known_labels

Step 4  框噪声 — CDN 核心
        positive_idx = 每组前半索引
        negative_idx = positive_idx + num_gt  （同一 GT 的负样本）
        在 xyxy 空间加 rand_sign * rand_part * diff * scale
        negative 的 rand_part += 1.0 → 更大噪声

Step 5  写入 padding 张量
        input_query_label / input_query_bbox: (bs, pad_size, ·)
        map_known_indice 把 CDN embedding 填到对应 slot

Step 6  构造 attn_mask (pad_size + num_queries, 同)
        [pad_size:, :pad_size] = True     # matching 不看 CDN
        循环 dn_number 组，组间互相 mask

Step 7  返回 dn_meta = {pad_size, num_dn_group}
        forward 里与 900 matching query concat 进 Decoder
```

**Step 8  `dn_post_process`**（`dino.py` forward 末尾）：

```
从 outputs_class/coord 切出前 pad_size 维 → CDN 预测
matching 部分从 pad_size: 开始 → 匈牙利
SetCriterion: dn_pos_idx / dn_neg_idx 分别算 pos 重建 loss 与 neg ∅ loss
```

### 11.2 文件索引

| 概念 | 文件（IDEA-Research/DINO） |
|------|------------------------------|
| CDN 准备 | `models/dino/dn_components.py` → `prepare_for_cdn` |
| CDN loss 分配 | `models/dino/dino.py` → `SetCriterion.forward` |
| Mixed QS / two-stage | `models/dino/dino.py` → `build_dino` |
| LFT / refine | `models/dino/deformable_transformer.py` Decoder；`use_detached_boxes_dec_out=False` |
| MSDeformAttn | 同 Deformable-DETR ops |
| 推理 top-300 | `models/dino/dino.py` → `PostProcess(num_select=300)` |
| 默认超参 | `config/DINO/DINO_5scale.py` |

**建议顺序**：

1. `config/DINO/DINO_5scale.py` → 默认超参一览  
2. `prepare_for_cdn` → §11.1  
3. `DINO.forward` → CDN + matching 双路 + `dn_post_process`  
4. `SetCriterion.forward` → matching / CDN / aux / interm loss  
5. `DeformableTransformerDecoder.forward` → LFT 与 `ref_points`  
6. `build_dino` → `two_stage_type`、`dn_number`、query 数配置  

---

## 12. 文档阅读路线

| 目标 | 章节 |
|------|------|
| 5 分钟懂 DINO | **§1.4.1（版本）** + **§1.6 ~ §1.9** + §1.3 流水线 |
| **选哪个 checkpoint / 配置** | **§1.4 + §1.4.1** |
| 与 DETR 系差在哪 | §1.5 → §3.0 → §5 |
| 搞懂 CDN | §3.1 + **§3.1.6** + §1.7 Walkthrough |
| 搞懂 Mixed QS | §3.2 |
| 搞懂 LFT | §3.3 + **§3.3.5** |
| 训练/推理 / loss 权重 | §4（**§4.1 权重表**） |
| 消融与贡献 | §6 + **§6.1 Optimized DN** |
| 维度与张量 | **§8** |
| 原理串讲 | §9 |
| 源码 | §11（含 **§11.1 prepare_for_cdn**） |
| MSDeformAttn CUDA | [DeformableDETR.md §11](./DeformableDETR.md) |
| Optimized DN 基线 | §6.1 |
| 局限 / 工程 | §7.2 |
| **DETR 系下一篇** | §13 |

---

## 13. DETR 系列演进（后续工作）

```
DETR (2020)
  → Deformable DETR (2021)     [DeformableDETR.md]
  → DAB-DETR / DN-DETR (2022)
  → DINO (2023)                [DINO.md] ← 本文
  → Grounding DINO (2023)      文本条件 open-vocabulary 检测，架构继承 DINO
  → Mask DINO (2023)           分割：DINO + mask head
  → DINO-X / 更新版检测器       IDEA 团队后续
```

| 方法 | 相对 DINO | 文档 |
|------|-----------|------|
| **Grounding DINO** | +文本 encoder，短语 grounding | 待整理 |
| **Mask DINO** | +mask 分支，统一分割 | 待整理 |
| **DINOv2（Meta）** | **无关**（自监督） | — |

**DETR 系文档链**：[DETR.md](./DETR.md) → [DeformableDETR.md](./DeformableDETR.md) → [DINO.md](./DINO.md)

---

## 14. 小结

DINO **不改变** DETR 的集合预测与端到端哲学，在 **Deformable + DAB + DN** 强基线上追加三项正交改进：

1. **CDN**：同一 GT 生成正/负噪声 query，既重建又 **拒绝** 远 anchor，减轻重复框与匹配混淆；  
2. **Mixed Query Selection**：Encoder top-K 只初始化 **anchor**，**content 可学习**，避免粗糙 Encoder 特征绑死语义；  
3. **Look Forward Twice**：层 i 的 box 监督 **多看一步** Δb_{i+1}，加强 cascade 框优化。

结果：**12 epoch 49.4 AP（R50）**，SwinL + Objects365 预训练 **63.3 AP test-dev**，证明 DETR 系可作为 **主流 SOTA 检测框架**。

---

## 参考文献

1. Zhang H., et al. **DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection.** ICLR 2023.  
2. Li F., et al. **DN-DETR: Accelerate DETR Training by Introducing Query DeNoising.** CVPR 2022.  
3. Liu S., et al. **DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR.** ICLR 2022.  
4. Zhu X., et al. **Deformable DETR.** ICLR 2021.  
5. Carion N., et al. **DETR.** ECCV 2020.
