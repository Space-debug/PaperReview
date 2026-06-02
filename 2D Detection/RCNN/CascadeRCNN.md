# Cascade R-CNN

> 本文档用于整理 Cascade R-CNN（Delving into High Quality Object Detection）论文精读笔记。  
> 重点：**训练/推理质量失配（quality mismatch）**、**多级检测头级联与递增 IoU 阈值**，以及相对 Faster / Mask R-CNN 的**结构与训练/推理差异**。

> 前置阅读：[RCNN.md](./RCNN.md) → [FastRCNN.md](./FastRCNN.md) → [FasterRCNN.md](./FasterRCNN.md) → [MaskRCNN.md](./MaskRCNN.md)

---

## 1. 概览

### 1.1 基本信息

| 项目 | 内容 |
|------|------|
| 论文 | Cascade R-CNN: Delving into High Quality Object Detection |
| 作者/机构 | Zhaowei Cai, Nuno Vasconcelos（UC San Diego） |
| 发表 | CVPR 2018 |
| 任务 | **高质量目标检测**（提升高 IoU 下的 AP，尤其 AP75 / AP90） |
| 代码 | [zhaoweicai/cascade-rcnn](https://github.com/zhaoweicai/cascade-rcnn)、Detectron2 `CascadeRCNN` |

### 1.2 核心思想（一句话）

**检测器在训练时用 IoU=0.5 定义正样本，但 COCO 评估看重 IoU≥0.75 的框；Cascade R-CNN 用多个检测头级联，逐级提高 IoU 阈值训练，并让每一级的输入分布与上一级推理输出对齐，从而专门优化「高质量检测」。**

### 1.3 整体流水线

```
Input Image
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Backbone + FPN + RPN（与 Faster / Mask R-CNN 相同）          │
│  输出 proposals + 共享多尺度特征                               │
└──────────────────────────────────────────────────────────────┘
    │
    ▼  proposals / 初始框 b⁰
┌──────────────────────────────────────────────────────────────┐
│  Stage 1 检测头（h¹）  训练 IoU 阈值 u₁ = 0.5                 │
│  RoIAlign → cls + bbox  →  refined 框 b¹                    │
└──────────────────────────────────────────────────────────────┘
    │
    ▼  用 b¹ 作为下一级 RoI（非原始 proposal）
┌──────────────────────────────────────────────────────────────┐
│  Stage 2 检测头（h²）  训练 IoU 阈值 u₂ = 0.6                 │
│  RoIAlign → cls + bbox  →  refined 框 b²                    │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 3 检测头（h³）  训练 IoU 阈值 u₃ = 0.7                 │
│  RoIAlign → cls + bbox  →  最终框 b³                        │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
NMS → 最终检测（bbox + class + score）
```

> **Cascade R-CNN 相对单阶段 Faster / Mask R-CNN**
>
> | 维度 | 单阶段 Faster/Mask R-CNN | Cascade R-CNN |
> |------|--------------------------|---------------|
> | 检测头数量 | 1 | **3（典型）**，参数独立 |
> | 训练 IoU 阈值 | 固定 0.5 | **0.5 → 0.6 → 0.7 递增** |
> | 每级输入框 | 始终 RPN proposals | **上一级回归后的框** |
> | 优化目标 | 兼顾 recall（0.5） | **逐级专精高 IoU 定位** |
> | AP75 / AP90 | 相对弱 | **显著提升** |
> | bbox AP (COCO) | ~36–40 | **~42+**（同 backbone） |

### 1.4 COCO 精度参考（test-dev，bbox）

| 模型 | AP | AP50 | AP75 | AP90 |
|------|-----|------|------|------|
| Faster R-CNN ResNet-101-FPN | 36.2 | 59.1 | 39.0 | — |
| Mask R-CNN ResNet-101-FPN | 39.8 | 62.3 | 43.4 | — |
| **Cascade R-CNN ResNet-101-FPN** | **42.8** | **62.1** | **46.3** | 明显提升 |
| **Cascade Mask R-CNN** | bbox 42.8 / **segm 38.4** | — | — | — |

- **AP50 几乎不变**，**AP75 提升 ~3–7 点** 是论文主卖点
- 说明级联主要改善 **定位质量**，而非「找更多物体」

---

## 2. 动机：Quality Mismatch（质量失配）

### 2.1 COCO 评估与训练目标的错位

```
COCO 主指标 AP:  IoU 阈值 0.5, 0.55, 0.6, ..., 0.95  取平均
→ 高 AP 需要 在 IoU=0.75、0.85 时仍有大量正确检测

经典 Faster R-CNN 训练:
  正样本:  proposal 与 GT  IoU ≥ 0.5
  回归目标:  把「勉强对齐」的框也当正样本去学

结果:
  分类器:  对 0.5 IoU 的框也能给高分 → recall 好
  回归器:  在松散正样本上训练 → 输出框常停在 0.5~0.6 IoU 一带
  评估:    AP75/AP90 很差
```

**图示理解**：

```
IoU 轴:  0.5────0.6────0.7────0.75────0.9
         │训练正样本│      │  COCO 高 AP 真正关心的区域  │
         └─ 单阶段大量正样本 ─┘      └─ 需要更紧的框 ─┘
```

### 2.2 直觉实验：直接把训练阈值提到 0.7？

| 方案 | 现象 |
|------|------|
| 单阶段，u=0.5（默认） | AP50 高，AP75 低 |
| 单阶段，u=0.7 训练 | AP75 仍差；AP50 **也掉** |
| **Cascade，0.5→0.6→0.7** | AP50 保持，**AP75 大幅升** |

**为何单阶段 u=0.7 失败？两大原因（论文核心）**：

#### （1）正样本过少 → 过拟合 / 拟合差

```
IoU ≥ 0.7 的 proposal 数量远少于 ≥ 0.5
→ 回归器见过的正样本分布窄
→ 对测试时大量 IoU∈[0.5,0.7) 的候选泛化差
```

#### （2）训练 / 推理分布不一致（Inference Distribution Mismatch）

```
训练（若单阶段 u=0.7）:
  输入框 = RPN proposals（质量参差不齐，很多 IoU<0.7）

推理:
  输入框 = 同一检测器上一轮的输出（分布已变）

单阶段用高 u 训练时，回归器假设输入已是「较准框」
实际推理第一步输入却是「粗糙 proposal」→ 失配
```

**Cascade 的解法**：第 t 级用 **第 t−1 级输出框** 训练，且 **u_t 逐级升高** → 每级看到的输入分布与推理时一致。

### 2.3 「高质量检测」定义

论文中 **quality** 指 **检测框与 GT 的 IoU**，不是分类置信度：

```
低 quality: IoU ≈ 0.5（刚及格）
高 quality: IoU ≥ 0.75（定位准）

Cascade 目标:  在保持 recall（AP50）的同时，把框推向高 IoU 区域
```

---

## 3. 网络结构详解

### 3.1 与 Faster R-CNN 的继承关系

```
保留:
  ✓ Backbone + FPN + RPN
  ✓ RoIAlign（7×7）+ FC cls + FC bbox
  ✓ bbox 参数化 (tx, ty, tw, th)（相对当前框）
  ✓ Smooth L1 回归损失、Softmax 分类损失
  ✓ 多任务 L = L_cls + L_box（每级各一份）

新增:
  ✓ N 个级联检测头 h¹, h², ..., h^N（默认 N=3）
  ✓ 每级独立参数，不共享 head 权重
  ✓ 每级独立 IoU 阈值 u_t
```

### 3.2 级联结构（数学表述）

记第 t 级（t = 1..N）：

```
输入框:   b^(t-1)     （t=1 时 b⁰ = RPN proposals）
特征:     x^t = RoIAlign(F, b^(t-1))
输出:     (p^t, b^t) = h^t(x^t)

b^t = ApplyDelta(b^(t-1), Δ^t)    ← 相对 b^(t-1) 回归，非相对原始 proposal
```

**三级级联数据流**：

```
RPN proposals  b⁰
      │
      ▼  h¹, u₁=0.5
     b¹  （已比 b⁰ 更贴 GT）
      │
      ▼  h², u₂=0.6   ← 在 b¹ 上重新 RoIAlign、重新采样正负样本
     b²
      │
      ▼  h³, u₃=0.7
     b³  → 最终检测框
```

### 3.3 每级 Head 是否共享特征？

```
Backbone + FPN:     全图共享，只算 1 次
RPN:                共享
RoIAlign:           每级对当前框 b^(t-1) 各做一次（框不同 → 采样区域不同）
检测头参数 h^t:     各级独立（3 套 FC cls + FC bbox）

→ 不是「同一特征过 3 遍同一 head」
→ 是「3 个专门化 head，输入分布逐级变干净」
```

### 3.4 Cascade Mask R-CNN

在 Mask R-CNN 上 **每个 stage 再加 mask head**（结构同 Mask R-CNN）：

```
Stage t:  RoIAlign 7×7 → cls + bbox
          RoIAlign 14×14 → mask (仅正样本算 L_mask^t)

推理:  用最后一级 b^N 与对应 mask
扩展:  segm AP 同样受益，尤其高 IoU mask
```

---

## 4. 训练：样本分配与损失（关键细节）

### 4.1 第 t 级的正负样本定义

对第 t 级，输入框为 **b^(t-1)**（训练时由上一级 **回归输出** 得到，非 GT 直接替代）：

| 类型 | 条件 | 说明 |
|------|------|------|
| **正样本** | max IoU(b^(t-1), GT) ≥ **u_t** | u_t 随 stage 升高 |
| **负样本** | max IoU(b^(t-1), GT) < **u_t** | 含 background |
| **忽略** | （可选）与 Faster R-CNN 类似低 IoU 忽略 | 实现中常沿用 0.1 以下忽略 |

**默认三级阈值**：

```
Stage 1:  u₁ = 0.5
Stage 2:  u₂ = 0.6
Stage 3:  u₃ = 0.7
```

**与单阶段的对比**：

```
单阶段:  所有回归都在「RPN 粗糙框」上，用 u=0.5 采样一次
Cascade:  Stage2 在「Stage1 精修框」上，用 u=0.6 采样
         Stage3 在「Stage2 精修框」上，用 u=0.7 采样
```

### 4.2 训练时 b^(t-1) 从哪来？（避免 teacher forcing 误解）

```
端到端联合训练时:

Stage 1:  b⁰ = RPN proposals（与 Faster R-CNN 相同）
Stage 2:  b¹ = Stage1 的 bbox head 对 b⁰ 的回归结果（前向得到）
Stage 3:  b² = Stage2 的 bbox head 对 b¹ 的回归结果

→ 每级回归器学习的是:
   「在给定质量的输入框上，如何进一步拉近 GT」
→ 与推理时分布一致（非用 GT 框代替中间级输入）
```

### 4.3 每级损失与总损失

```
L = Σ_{t=1}^{N}  ( L_cls^t + L_box^t )

Stage t 的 L_cls^t:  对采样 RoI 的 Softmax CE（含 background）
Stage t 的 L_box^t:  仅正样本，Smooth L1，相对 b^(t-1) 参数化

若有 mask（Cascade Mask R-CNN）:
L = Σ_t ( L_cls^t + L_box^t + L_mask^t )
L_mask^t 仅对 stage t 的正样本、GT 类通道
```

**Mini-batch**：每级通常仍采样 **128 RoI / iter**（与 Faster R-CNN 同量级），正负比约 1:3。

### 4.4 为何不能「单 head 迭代回归 3 次」？

| 做法 | 效果 |
|------|------|
| 同一 head 对 proposals 连回归 3 次 | 有限提升，**远不如** 独立 head + 递增 u |
| 3 个 head，但都用 u=0.5 | AP75 提升小 |
| **3 head + u={0.5,0.6,0.7}** | **最佳** |

原因：同一 head 同一 u 重复调用，**没有专门化**于不同 IoU 区间的输入分布；级联的价值在 **专用 head + 专用阈值**。

### 4.5 训练超参（COCO，ResNet-FPN）

| 超参 | 值 |
|------|-----|
| Stages N | 3 |
| IoU 阈值 | {0.5, 0.6, 0.7} |
| 优化器 | SGD，与 Mask R-CNN 类似 |
| 学习率 | 0.02，60k/80k step decay |
| Backbone | ResNet-50/101-FPN |
| RoIAlign | 7×7，sample_ratio=2 |

---

## 5. 推理流程（逐步）

```
输入: 测试图 I
────────────────────────────────────────────────────────────

1. FPN + RPN → proposals b⁰

2. Stage 1
   对每个 proposal: RoIAlign → h¹ → (score¹, Δ¹)
   b¹ = ApplyDelta(b⁰, Δ¹)
   （一般不在 stage1 后做最终 NMS，保留足够候选进入下级）

3. Stage 2
   以 b¹ 为 RoI 框（不是 b⁰）→ RoIAlign → h² → b²

4. Stage 3
   以 b² 为 RoI → h³ → b³, score³, class³

5. 最终后处理
   用最后一级 score³ + class³
   score 过滤 → 类内 NMS（IoU 通常 0.5）

6. 输出检测结果

────────────────────────────────────────────────────────────
Cascade Mask R-CNN: 最后一级同时输出 mask，贴回 b³
```

**推理注意**：

```
中间级分类分数一般不直接作为最终结果
最终类别 / 置信度以最后一级 h^N 为准

每级都在「更准的框」上重新提取特征 → 分类与回归 progressively 变准
```

---

## 6. 消融实验与关键结论

### 6.1 IoU 阈值与 stage 数

| 配置 | AP | AP75 | 说明 |
|------|-----|------|------|
| 1 stage, u=0.5 | 基准 | 低 | 标准 Faster R-CNN |
| 1 stage, u=0.7 | ↓ | 仍低 | 正样本少 + 分布失配 |
| 3 stage, 全 u=0.5 | 略升 | 略升 | 无递增阈值，收益有限 |
| **3 stage, u={0.5,0.6,0.7}** | **↑** | **↑↑** | 论文默认 |

### 6.2 级联 vs 迭代回归

```
迭代回归（1 head，多次 bbox refine）:  AP75 提升有限
Cascade（3 head，递增 u）:             AP75 提升显著

→ 「多级」+「专用 head」+「递增 u」缺一不可
```

### 6.3 对不同 IoU 指标的影响

```
AP50:   基本持平或略变（recall 主要由 RPN + stage1 保证）
AP75:   +3~7 点（主收益）
AP90:   明显提升（高 IoU 检测稀少但级联专门优化）
```

### 6.4 与 IoU-Net、BBox Refinement 等对比

论文讨论：事后用独立网络预测 IoU 再筛框（如 IoU-Net）是 **另一路 quality 建模**；Cascade 从 **训练级联 + 分布对齐** 入手，可与之互补或结合。

---

## 7. 设计思想总结（精读必记）

### 7.1 三个独立概念不要混

```
1. 递增 IoU 阈值 u_t
   → 定义每阶段「什么样算正样本」

2. 递增输入框质量 b^(t-1)
   → 每阶段回归器看到的输入越来越准

3. 独立 head 参数 h^t
   → 每个 IoU 段有专用分类/回归函数
```

### 7.2 与「难例挖掘」的区别

```
OHEM / 难例挖掘:  选 loss 大的样本
Cascade:          按 IoU 分段，让后级专门学「精修已较准的框」

目标不同:  Cascade 针对 evaluation metric 中的高 IoU 区间
```

### 7.3 对后续工作的影响

```
Cascade 思想被广泛用于:
  - HTC (Hybrid Task Cascade，实例分割)
  - Detectron2 CascadeRCNN / Cascade Mask R-CNN 标配
  - 一些 one-stage 的 refine 分支（思想类似，实现不同）

局限:
  - 推理要跑 N 遍 head，延迟 ≈ N × 单 head（常 N=3）
  - 结构更重，小模型 / 实时场景少用
```

---

## 8. 历史地位与局限

### 8.1 贡献

1. 系统分析 **quality mismatch**，解释「为何 AP50 高、AP75 低」；
2. **级联检测头 + 递增 IoU** 成为高 AP 两阶段检测标配；
3. **Cascade Mask R-CNN** 统一检测与实例分割的高质量定位；
4. 推动 COCO 竞赛方案从「单阶段调参」转向「多级 specialization」。

### 8.2 局限

| 局限 | 说明 |
|------|------|
| 计算量 | 3 个 head 串行，推理慢于单阶段 Faster/Mask R-CNN |
| 仍依赖 RPN | proposal 质量上限仍在；后由 Cascade RPN 等改进 |
| 固定 u 序列 | {0.5,0.6,0.7} 需针对数据集调节 |
| 非端到端实时 | 两阶段 + 三级，难以部署到极端实时场景 |

### 8.3 R-CNN 系列演进（更新）

```
R-CNN (2014)           SS + 独立 CNN
Fast R-CNN (2015)      RoI Pool + 联合训练
Faster R-CNN (2016)    RPN
Mask R-CNN (2017)      RoIAlign + mask
Cascade R-CNN (2018)   多级 head + 递增 IoU  ← 本文
Cascade Mask R-CNN     实例分割 + 高质量框
```

---

## 9. 精读备忘：易混淆点

### 9.1 每级回归参照框是谁

```
错误理解:  三级都相对 RPN proposal 回归
正确理解:  Stage t 的 Δ^t 相对 b^(t-1) 回归

b^t = Transform(b^(t-1), Δ^t)     非 Transform(b^0, ΣΔ)
```

### 9.2 为何 Stage1 仍用 u=0.5

```
Stage1 负责:  从大量粗糙 proposal 中稳住 recall + 粗定位
若 Stage1 就用 u=0.7:  正样本太少，级联没有足够入口

后两级在 increasingly tight 的框上提高 u
```

### 9.3 训练与推理级联是否一致

```
一致:  推理也是 b⁰→h¹→b¹→h²→b²→h³→b³
训练:  中间 b^t 由网络前向计算（非 GT 框），保证分布匹配
```

### 9.4 NMS 在哪做

```
一般仅在最后一级输出后做一次类内 NMS
中间级不做激进 NMS，避免过早丢掉尚可被后级救回的框
（具体实现可有 score threshold 微调）
```

### 9.5 Cascade R-CNN vs Cascade Mask R-CNN

```
Cascade R-CNN:       仅 bbox + cls 三级
Cascade Mask R-CNN:  每级再加 mask head，L_mask 也按 stage 分
推理 mask 通常取最后一级
```

### 9.6 与 Soft-NMS / Test-time augmentation 关系

```
正交:  Cascade 改模型结构与训练；TTA / Soft-NMS 是后处理
可叠加使用
```

---

## 10. R-CNN 系列串联总结（含 Cascade）

```
┌─────────────┬──────────┬───────────┬────────────┬────────────┬──────────────┐
│             │ R-CNN    │ Fast      │ Faster     │ Mask       │ Cascade      │
├─────────────┼──────────┼───────────┼────────────┼────────────┼──────────────┤
│ Proposal    │ SS       │ SS        │ RPN        │ RPN        │ RPN          │
│ RoI 采样    │ warp     │ RoI Pool  │ RoI Pool   │ RoI Align  │ RoI Align    │
│ 检测头      │ SVM+线性 │ 1× Softmax│ 1×         │ 1×+mask    │ **3× 级联**  │
│ 训练 IoU    │ 0.5      │ 0.5       │ 0.5        │ 0.5        │ **0.5/0.6/0.7**│
│ 核心卖点    │ CNN特征  │ 共享conv  │ 学习proposal│ 实例分割   │ **高IoU AP** │
└─────────────┴──────────┴───────────┴────────────┴────────────┴──────────────┘
```

**演进主线**：检测框架成熟后，从 **「能检出」（AP50）** 走向 **「检得准」（AP75+）**；Cascade 用 **分布对齐的级联 specialization** 解决 quality mismatch。

---

## 11. 参考资料

- 原论文：[Cascade R-CNN](https://arxiv.org/abs/1712.02626)（CVPR 2018）
- 前置：[Faster R-CNN](./FasterRCNN.md)、[Mask R-CNN](./MaskRCNN.md)
- 扩展：Cascade Mask R-CNN（同论文 / Detectron2）
- 相关：[IoU-Net](https://arxiv.org/abs/1711.08864)（IoU 引导的检测质量估计）
- 实现：[Detectron2 Cascade R-CNN](https://github.com/facebookresearch/detectron2)

---

*文档版本：初稿 | 对应论文 CVPR 2018 Cascade R-CNN*
