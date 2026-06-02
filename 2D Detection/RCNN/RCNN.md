# R-CNN

> 本文档用于整理 R-CNN（Rich feature hierarchies for accurate object detection and semantic segmentation）论文精读笔记。  
> 重点：**核心网络结构**、**四阶段检测流水线**，以及**训练/推理中的样本定义与处理方式**。

---

## 1. 概览

### 1.1 基本信息

| 项目 | 内容 |
|------|------|
| 论文 | Rich feature hierarchies for accurate object detection and semantic segmentation |
| 作者/机构 | Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik（UC Berkeley） |
| 发表 | CVPR 2014 |
| 任务 | 目标检测（Region-based，两阶段检测的开山之作） |
| 代码 | [rbgirshick/rcnn](https://github.com/rbgirshick/rcnn)（MATLAB + Caffe 时代实现） |

### 1.2 核心思想（一句话）

**不再用手工特征（HOG、SIFT），而是用深度 CNN 对每个候选区域提取高维语义特征，再用线性 SVM 做分类、线性回归做框精修。**

R-CNN 的关键突破不是「端到端一个网络搞定检测」，而是证明了：

1. **监督预训练 + 检测微调** 可以把 ImageNet 上学到的表示迁移到检测任务；
2. **CNN 特征** 在 region proposal 上远强于当时的手工特征；
3. **高容量模型 + 区域候选** 的组合，在 PASCAL VOC 上把 mAP 从 ~30% 拉到 50%+。

### 1.3 整体流水线（四阶段）

R-CNN **不是** 单网络端到端检测，而是典型的 **「候选区 → 特征 → 分类 → 回归」** 四段式：

```
Input Image (任意尺寸, 如 480×640×3)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1: Region Proposal（Selective Search）            │
│  每张图约 2000 个类别无关的候选框（矩形 region）          │
└─────────────────────────────────────────────────────────┘
    │
    ▼  对每个 region 独立处理（循环 2000 次）
┌─────────────────────────────────────────────────────────┐
│  Stage 2: Feature Extraction（CNN, AlexNet 变体）        │
│  候选框 → warp 成 227×227 → 前向传播 → 取 fc7 的 4096 维 │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 3: Category Classification（每类一个线性 SVM）    │
│  4096-d 特征 → 21 个 one-vs-rest SVM → 各类得分           │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 4: Bounding Box Regression（每类一个线性回归器）   │
│  对高分框用类别专属回归器精修 (x, y, w, h)                │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Output: 经过 NMS 后的检测框 + 类别 + 置信度
```

> **与后续方法对比**
>
> | 方法 | Proposal | 特征提取 | 分类/回归 | 是否端到端 |
> |------|----------|----------|-----------|------------|
> | **R-CNN** | Selective Search（外部） | 每个 region 单独过 CNN | SVM + 线性回归（独立训练） | 否，四阶段 |
> | **Fast R-CNN** | SS / EdgeBoxes | 整图一次 CNN，RoI Pooling 取特征 | Softmax + 回归（同一网络） | 部分 |
> | **Faster R-CNN** | RPN（网络内生成） | 整图共享 + RoI | 同上 | 是（proposal+检测一体） |
> | **YOLO 等单阶段** | 无显式 proposal | 整图 dense 预测 | 直接输出 | 是 |

### 1.4 PASCAL VOC 2007 精度参考

| 方法 | mAP | 说明 |
|------|-----|------|
| DPM（手工特征 + 滑动窗口） | 33.7 | R-CNN 之前 SOTA |
| R-CNN（VGG-16 特征） | 66.0 | 论文主结果之一 |
| R-CNN（AlexNet 特征） | 58.5 | 更快、更轻量的 backbone |
| R-CNN + bbox reg | +3~4 mAP | 框回归带来的稳定增益 |
| R-CNN + context padding | +2~3 mAP | 上下文填充带来的增益 |

> 在 VOC 2010/2012 上同样大幅领先当时所有方法；这也是深度学习检测时代的起点。

---

## 2. Stage 1：Region Proposal（Selective Search）

R-CNN **本身不训练** proposal 模块，直接调用 Uijlings 等人的 **Selective Search**（IJCV 2013）。

### 2.1 Selective Search 在做什么

目标：在**不依赖类别标签**的前提下，从图像中找出「可能是物体」的候选区域。

```
原图
  │
  ▼  初始过分割（Felzenszwalb 图分割）
  得到大量小色块（superpixels）
  │
  ▼  层次化合并
  基于颜色、纹理、尺寸、填充度等相似度，贪心合并相邻区域
  │
  ▼  多尺度
  在不同合并阶段取候选框 → 覆盖不同大小物体
  │
  ▼
输出约 2000 个 region proposals（矩形框）
```

### 2.2 为什么用 Selective Search

| 优点 | 缺点（R-CNN 时代的痛点） |
|------|--------------------------|
| 类别无关，Recall 高（~98% 物体被某个 proposal 覆盖） | CPU 上约 2s/图，慢 |
| 候选数远少于滑动窗口（2000 vs 数十万） | 框质量一般，大量冗余 |
| 与 CNN 解耦，R-CNN 只关心「框里有没有物体」 | 无法与 CNN 联合优化 → Faster R-CNN 的 RPN 动机 |

### 2.3 与 R-CNN 的接口

- 输入：原图 RGB
- 输出：一组 `(x1, y1, x2, y2)` 像素坐标矩形
- R-CNN 对**每个** proposal 一视同仁地提取特征，**不区分** proposal 来源或分数

---

## 3. Stage 2：CNN 特征提取（核心网络结构）

### 3.1 Backbone：AlexNet（Krizhevsky et al., NIPS 2012）

R-CNN 使用在 **ImageNet ILSVRC 2012** 上预训练好的 **AlexNet**，结构如下（与分类任务一致，直到 fc7）：

```
Input: warped region, 227 × 227 × 3
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  Conv1: 11×11, stride 4, 96 filters  → 55×55×96         │
│  ReLU + LRN + MaxPool 3×3, stride 2  → 27×27×96         │
├──────────────────────────────────────────────────────────┤
│  Conv2: 5×5, 256 filters              → 27×27×256        │
│  ReLU + LRN + MaxPool 3×3, stride 2  → 13×13×256        │
├──────────────────────────────────────────────────────────┤
│  Conv3: 3×3, 384 filters              → 13×13×384        │
│  ReLU                                                    │
├──────────────────────────────────────────────────────────┤
│  Conv4: 3×3, 384 filters              → 13×13×384        │
│  ReLU                                                    │
├──────────────────────────────────────────────────────────┤
│  Conv5: 3×3, 256 filters              → 13×13×256        │
│  ReLU + MaxPool 3×3, stride 2        → 6×6×256          │
├──────────────────────────────────────────────────────────┤
│  FC6: 4096-d, ReLU, Dropout                              │
│  FC7: 4096-d, ReLU, Dropout   ← R-CNN 取这里的输出作为特征 │
├──────────────────────────────────────────────────────────┤
│  FC8: 1000-d (ImageNet 1000 类)  ← 检测微调时被替换       │
└──────────────────────────────────────────────────────────┘
```

**R-CNN 使用的特征向量：fc7 输出的 4096 维向量。**

论文还实验了 **VGG-16**（Simonyan & Zisserman, 2014）作为 backbone，fc7 等价层输出同样是 4096-d，但 mAP 更高（66.0 vs 58.5），代价是前向更慢。

### 3.2 为什么取 fc7 而不是 conv5

| 特征层 | 维度 | 特点 |
|--------|------|------|
| **conv5** | 6×6×256 = 9216（需 flatten） | 更偏局部、低层；论文 ablation 显示 fc7 更好 |
| **fc7** | 4096 | 高层语义 + 全局池化后的综合表示，分类 SVM 效果最佳 |

直觉：检测需要的是「这个区域像不像某类物体」的**语义级**判断，fc7 经过多层 conv + 两个全连接，比 conv5 更适合做后续线性分类。

### 3.3 Region → CNN 输入：Warp 与 Context Padding

每个 proposal 的宽高比、尺寸都不同，CNN 却要求固定 **227×227**。处理方式：

```
Step 1: 对 proposal 做 context padding（上下文填充）
  - 在框的四周各扩 p 像素（论文默认 p = 16）
  - 超出图像边界的部分用「图像均值」填充（RGB 三通道均值）

Step 2: 强制 warp 到 227×227
  - 不保持宽高比，直接拉伸/压缩到正方形
  - 会有形变，但实验表明仍然有效

Step 3: 减 ImageNet 均值，送入 CNN
```

**Context padding 示意：**

```
原图                          带 padding 的 crop              warp 后
┌─────────────────┐          ┌───────────────┐            ┌─────────┐
│     ┌───┐       │          │  ┌─────────┐  │   resize   │         │
│     │obj│       │   →      │  │  ┌───┐  │  │   ────→    │ 227×227 │
│     └───┘       │          │  │  │obj│  │  │            │         │
└─────────────────┘          │  │  └───┘  │  │            └─────────┘
                             │  └─────────┘  │
                             └───────────────┘
                               p=16 像素上下文
```

| 设置 | mAP 影响 | 原因 |
|------|----------|------|
| 无 padding | 明显下降 | 物体边缘被裁切，丢失上下文 |
| p = 16 | 默认最优 | 约 1.5 倍 proposal 面积，兼顾上下文与计算 |
| 用 0 填充 vs 均值填充 | 均值更好 | 0 在 AlexNet 输入分布中属于 outlier |

### 3.4 推理时的计算瓶颈

```
一张图 ~2000 proposals × 每次 CNN 前向 ≈ 47s / 图（GPU）
                              ≈ 5s / 图（CPU，仅 CNN 部分）
```

**每个 region 独立前向、无法共享卷积** —— 这是 R-CNN 最大缺陷，也是 Fast R-CNN「整图一次卷积 + RoI Pooling」的直接动机。

---

## 4. Stage 3：类别分类（Linear SVM）

### 4.1 为什么不用微调后的 Softmax，而要再训 SVM

论文做了明确 ablation：**微调 CNN + 线性 SVM** 比 **微调 CNN + Softmax 直接当检测器** 更好。

原因概括：

1. **Softmax 与检测目标不完全一致**：微调时的 mini-batch 采样（见 5.2）和正负样本比例，是为了优化 log-loss，不是直接优化 mAP；
2. **SVM 的 hinge loss** 更契合「把正样本分数推高、负样本推低」的排序目标；
3. **Hard negative mining** 在 SVM 框架下更自然（见 4.3）。

因此 R-CNN 的实际做法是：

```
CNN 微调 → 固定 CNN，用 fc7 特征离线提取 → 对每个类训练一个 binary linear SVM
```

### 4.2 One-vs-Rest 线性 SVM

PASCAL VOC 有 20 个物体类 + 1 个 background，R-CNN 训练 **21 个二分类线性 SVM**：

```
类 k 的 SVM:  f_k(x) = w_k^T · x + b_k

x ∈ R^4096  （fc7 特征）
得分 > 0     → 倾向于类 k
```

推理时对每个 proposal 的 4096-d 特征，计算 20 个物体类的 SVM 得分；**background 不单独输出**，低分即视为背景。

### 4.3 训练样本定义（SVM 阶段，与微调阶段不同！）

| 类型 | IoU 条件 | 说明 |
|------|----------|------|
| **正样本** | IoU(proposal, GT) ≥ **0.5** | 与某 GT 重叠足够 |
| **负样本（hard negative）** | IoU ≤ **0.3** | 与所有 GT 重叠都很小 |
| **忽略** | 0.3 < IoU < 0.5 | 不参与 SVM 训练 |

> 注意：**0.3~0.5 区间在 SVM 训练中被丢弃**，避免「半正半负」样本带来标签噪声。  
> 这与微调阶段「忽略 0.1~0.5」的设定不同（见 5.2）。

### 4.4 Hard Negative Mining

```
1. 用当前 SVM 对所有 proposal 打分
2. 取被误判为正的负样本（高分负例）→ hard negatives
3. 加入训练集，重新训练 SVM
4. 迭代 1~2 轮（论文默认）
```

作用：PASCAL 中负样本远多于正样本，且大量「easy negative」（天空、草地）对边界学习帮助小；**hard negative** 迫使分类器学会区分「像物体但不是」的区域。

---

## 5. CNN 检测微调（连接 Stage 2 与 Stage 3 的桥梁）

在训 SVM 之前，需要把 ImageNet 预训练 CNN **适配到检测数据**。

### 5.1 替换分类头

```
原 FC8: 4096 → 1000（ImageNet 类数）
         ↓ 替换
新 FC8: 4096 → N+1（N = 物体类数，+1 = background）

PASCAL VOC: N+1 = 21
```

权重初始化：新 FC8 用 **零均值、0.01 标准差的高斯** 随机初始化；其余层从 ImageNet 预训练权重加载。

### 5.2 微调时的样本定义

| 类型 | IoU 条件 |
|------|----------|
| **正样本** | IoU ≥ **0.5** |
| **负样本** | IoU ∈ **[0.1, 0.5)** |
| **忽略** | IoU < **0.1** |

与 SVM 阶段的差异：

```
IoU 轴:  0────────0.1────────0.3────────0.5────────1.0
         │ 忽略   │  微调负样本   │ 丢弃  │  正样本  │
         │        │              │(SVM)  │(两者共用)│
```

微调需要「难一点的负样本」（0.1~0.5）来学边界；SVM 阶段则把 0.3~0.5 丢掉，避免标签歧义。

### 5.3 Mini-batch 采样策略

```
每个 SGD iteration:
  batch_size = 128
  其中正样本 = 32（25%）
       负样本 = 96（75%）

若当前 epoch 正样本不足 32，则重复采样正样本直至凑满
```

**为什么 25% 正样本**：检测数据中 GT 框很少，若按自然比例几乎全是负样本，模型会塌缩成「全预测背景」。

### 5.4 微调超参（论文默认）

| 超参 | 值 |
|------|-----|
| 初始学习率 | 0.001（为 ImageNet 预训练时的 1/10） |
| 每 layer 学习率倍率 | conv1: 1×，fc8: 100×（新层需更快学习） |
| 迭代次数 | 约 30k mini-batch（VOC trainval） |
| 数据增强 | 无（仅 warp + padding） |

### 5.5 特征缓存（训练 SVM 时的工程技巧）

微调完成后，对所有 train/val 图像的所有 proposal **前向一次 CNN**，把 fc7 特征**磁盘缓存**：

```
优点: SVM 训练、hard negative mining、bbox 回归训练可反复实验，无需重复 CNN
缺点: 存储巨大（VOC: 数百 GB 量级特征文件）
```

---

## 6. Stage 4：Bounding Box Regression

### 6.1 动机

Proposal 来自 Selective Search，定位往往不准；即使分类正确，IoU 也可能不够高。**类别专属的边界框回归器**在 scored 框上做一次精修。

### 6.2 参数化方式（与 Fast/Faster R-CNN 相同，成为后续标准）

对每个 GT 框 `(Px, Py, Pw, Ph)` 和 proposal `(Gx, Gy, Gw, Gh)`，学习 4 个归一化偏移：

```
tx = (Gx - Px) / Pw
ty = (Gy - Py) / Ph
tw = log(Gw / Pw)
th = log(Gh / Ph)
```

回归器预测 `(t̂x, t̂y, t̂w, t̂h) = f(fc7_features)`，推理时反变换：

```
Ĝx = Pw · t̂x + Px
Ĝy = Ph · t̂y + Py
Ĝw = Pw · exp(t̂w)
Ĝh = Ph · exp(t̂h)
```

**为什么 tw, th 用 log 空间**：尺度变化跨数量级，log 后更接近线性，回归更稳定。

### 6.3 训练设置

| 项目 | 设置 |
|------|------|
| 训练样本 | 仅 **正样本** proposal（IoU ≥ 0.5 且类标正确） |
| 回归器 | **每类一个** 线性回归（4 维输出） |
| 输入 | 该 proposal 的 fc7 特征（4096-d） |
| 损失 | 各维度 squared loss 之和 + L2 正则 |
| 推理 | 仅对 **SVM 得分 > 0** 且非 background 的框做回归 |

### 6.4 效果

- mAP 通常 **+3~4 点**
- 对 IoU 要求高的指标（如 AP@0.7）提升更明显
- 每类独立回归：避免类间框尺度差异互相干扰

---

## 7. 完整推理流程（逐步）

```
输入: 测试图像 I
────────────────────────────────────────────────────────────

1. Selective Search
   regions = selective_search(I)     # ~2000 boxes

2. 对每个 region r:
   a. context padding (p=16)
   b. warp → 227×227
   c. CNN forward → fc7 feature f_r ∈ R^4096

3. 对每个 region r、每个类 k ∈ {1..20}:
   score_k = w_k^T · f_r + b_k      # SVM 得分

4. 对每个类 k 独立做 NMS:
   - 取该类所有 score_k > threshold 的框
   - 按得分降序
   - Greedy NMS，IoU 阈值 = 0.3（VOC 标准）

5. 对每个保留框 (r, k):
   (Δx, Δy, Δw, Δh) = Regressor_k(f_r)
   refined_box = apply_delta(r, Δ)

6. 输出: {(refined_box, class_k, score_k)}
────────────────────────────────────────────────────────────
```

**NMS 要点**：按**类内**做 NMS，不同类的框可以重叠（如「人骑着马」）。

---

## 8. 训练流程总览（三阶段独立训练）

```
┌─────────────────────────────────────────────────────────────┐
│ Phase A: ImageNet 监督预训练（已有，直接下载权重）            │
│ AlexNet on ILSVRC 2012, top-1 error ~16.4%                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase B: 检测微调 CNN                                         │
│ 数据: VOC trainval 的所有 GT + 采样的 proposal               │
│ 损失: Softmax log-loss（21 类）                               │
│ 输出: 微调后的 CNN 权重                                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase C: 离线提取 fc7 特征 + 训练 21 个 linear SVM            │
│ Hard negative mining, hinge loss                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase D: 训练每类 bbox regressor（线性，输入 fc7）            │
│ 仅正样本，squared loss + L2                                   │
└─────────────────────────────────────────────────────────────┘
```

> **痛点**：A/B/C/D 分离 → 训练流程繁琐、超参多、特征磁盘占用大。Fast R-CNN 合并 B+C+D 为单网络多任务损失。

---

## 9. 消融实验与关键结论

### 9.1 预训练的重要性

| 初始化 | VOC 2007 mAP |
|--------|--------------|
| 随机初始化 CNN | ~43% |
| ImageNet 预训练 + 微调 | ~54%+ |

**无预训练 mAP 掉约 10+ 点** —— 检测数据量远小于 ImageNet，直接训 CNN 容易过拟合；预训练提供通用视觉表示。

### 9.2 各组件贡献（典型 ablation）

| 配置 | 相对增益 |
|------|----------|
| R-CNN baseline（AlexNet + SS） | 基准 ~58.5 mAP |
| + bbox regression | +3~4 mAP |
| + context padding p=16 | +2~3 mAP |
| 换 VGG-16 backbone | +7~8 mAP（更慢） |
| 微调 + SVM vs 仅 SVM（不微调） | 微调必要 |
| fc7 vs conv5 特征 | fc7 更好 |

### 9.3 「Detection without Selective Search」实验

论文还做了 **全图 dense 扫描 + CNN** 的实验（类似早期 sliding window，但用 CNN 特征）：

- 在多尺度、多位置滑动窗口上跑 CNN
- mAP 仍不如 region-based 方案

结论：**高容量 CNN + 高质量 region proposals** 的组合优于 brute-force 滑动窗口 —— 候选的质量和数量平衡很关键。

---

## 10. 语义分割扩展（论文第二部分）

R-CNN 框架可扩展到 **语义分割**，思路是 **Region-based Semantic Segmentation**：

```
1. 对 SS 产生的每个 region，用 CNN 判断「是否属于某类物体」
2. 对正类 region，用贪心区域填充（greedy region filling）从 region 内估计像素级 mask
3. 多个 region 的 mask 投票/合并得到全图分割
```

在 PASCAL VOC segmentation 上也取得当时 SOTA，说明 **fc7 特征的区域级语义** 可迁移到像素任务（虽较粗糙）。

---

## 11. R-CNN 的历史地位与局限

### 11.1 开创性贡献

1. **首次**将深度 CNN 成功用于目标检测，大幅超越 DPM；
2. 确立 **「预训练 + 微调」** 范式，影响后续所有 detection backbone；
3. 确立 **region proposal + 深度特征 + classifier + regressor** 的两阶段模板；
4. **Bounding box regression 参数化** 被 Fast/Faster R-CNN 沿用；
5. 引发 **AlexNet → VGG → ResNet** 在检测上的持续升级。

### 11.2 主要局限

| 局限 | 具体表现 | 后续改进 |
|------|----------|----------|
| **慢** | 2000 次独立 CNN 前向，47s/图 | Fast R-CNN：整图 1 次 conv |
| **训练复杂** | CNN、SVM、regressor 三件套分开训 | Fast R-CNN：多任务联合 |
| **存储大** | fc7 特征离线缓存数百 GB | Fast R-CNN：在线 RoI Pooling |
| **Proposal 与 CNN 分离** | SS 无法利用 CNN 特征 | Faster R-CNN：RPN |
| **Warp 形变** | 强制 227×227 拉伸 | SPP-net / RoI Pooling 减少形变 |

### 11.3 演进脉络

```
R-CNN (2014)     四阶段，每 region 独立 CNN，SVM 分类
    ↓
Fast R-CNN (2015)  共享 conv，RoI Pooling，Softmax+回归同一网络
    ↓
Faster R-CNN (2016)  RPN 生成 proposal，真正端到端两阶段
    ↓
Mask R-CNN (2017)  加 mask 分支，实例分割
    ↓
单阶段方法 (YOLO/SSD/RetinaNet...)  去掉 proposal，追求速度
```

---

## 12. 精读备忘：易混淆点

### 12.1 微调 vs SVM 的 IoU 阈值

| 阶段 | 正 | 负 | 忽略 |
|------|----|----|------|
| **CNN 微调** | ≥ 0.5 | [0.1, 0.5) | < 0.1 |
| **SVM 训练** | ≥ 0.5 | ≤ 0.3 | (0.3, 0.5) |

### 12.2 「2000 proposals」与训练样本数

- **2000** 是测试时对**每张图** SS 输出的 proposal 数；
- 训练时还会加入 **所有 GT 框**（保证每个物体至少有一个高 IoU 正样本）；
- SVM 训练对象是整个 trainval 上**所有图的所有 proposal**（数十万级），不是 2000×图数那么简单。

### 12.3 R-CNN 里的「CNN」不负责找框

CNN 只做 **「给定一块图像，提取特征 +（微调时）做 21 类 softmax」**；  
**找框**靠 Selective Search，**最终分类分数**靠 SVM，**精修**靠 bbox regressor。

---

## 13. 参考资料

- 原论文：[Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524)（arXiv 2013，CVPR 2014）
- Selective Search：[Uijlings et al., IJCV 2013](https://ivi.fnwi.uva.nl/isis/publications/2013/UijlingsIJCV2013)
- 作者后续：**Fast R-CNN**（2015）、**Faster R-CNN**（2016）—— 建议按顺序继续精读
- 官方实现：[rbgirshick/rcnn](https://github.com/rbgirshick/rcnn)

---

*文档版本：初稿 | 对应论文 CVPR 2014 R-CNN*
