# FCOS

> 本文档用于整理 FCOS（Fully Convolutional One-Stage Object Detection）论文精读笔记。  
> 重点：**逐像素四边距回归（l,t,r,b）**、**FPN 层级尺度限制 + 中心采样**、**Centerness 分支**，以及相对 RetinaNet / CenterNet 的**样本定义与推理方式**。

> 相关：[RetinaNet.md](./RetinaNet.md)（Focal Loss、FPN）· [CenterNet.md](./CenterNet.md)（中心热图）· [RCNN/FasterRCNN.md](./RCNN/FasterRCNN.md)

---

## 1. 概览

### 1.1 基本信息

| 项目 | 内容 |
|------|------|
| 论文 | FCOS: Fully Convolutional One-Stage Object Detection |
| 作者/机构 | Zhi Tian, Chunhua Shen, Hao Chen, Tong He（阿德莱德大学 / 中科院等） |
| 发表 | ICCV 2019 |
| 任务 | **Anchor-free** 单阶段检测；把检测做成「全卷积密集预测」 |
| 代码 | [tianzhi0549/FCOS](https://github.com/tianzhi0549/FCOS)、[Detectron2 FCOS](https://github.com/facebookresearch/detectron2) |

### 1.2 核心思想（一句话）

**在 FPN 每一层特征图的每个空间位置上，若该点落在某 GT 框内且靠近物体中心，则预测到框四边的距离 (l,t,r,b) 与类别；用 Centerness 抑制远离中心的低质量预测，证明 anchor 并非必要。**

| 对比 | 正样本定义 | 框参数化 |
|------|-----------|----------|
| RetinaNet | 与 GT IoU 匹配的 **anchor** | 相对 anchor 偏移 |
| CenterNet | **中心点** 热图峰 | wh + reg |
| **FCOS** | 落在 GT 内且 **中心采样区** 的 **网格点** | **l,t,r,b** 四边距 |

### 1.3 整体流水线

```
Input（短边 800 等）
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Backbone + FPN → P3, P4, P5, P6, P7（stride 8~128）          │
└──────────────────────────────────────────────────────────────┘
    │
    ▼  每个 (level, x, y) 位置
┌──────────────────────────────────────────────────────────────┐
│  共享卷积 trunk → 分支:                                       │
│    · cls:  K 类 sigmoid（Focal Loss）                         │
│    · bbox: 4 通道 l,t,r,b（IoU / GIoU Loss，仅正样本）         │
│    · centerness: 1 通道（BCE，仅正样本）                       │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
推理: score = cls × centerness → 阈值 → 解码框 → 类内 NMS
```

### 1.4 COCO 精度参考（test-dev，bbox）

| 模型 | Backbone | AP | AP50 | AP75 | APs |
|------|----------|-----|------|------|-----|
| RetinaNet-101-FPN | R101 | 39.1 | 59.1 | 41.7 | 21.8 |
| CenterNet-DLA-34 | DLA | 42.1 | — | — | — |
| **FCOS-R50-FPN** | R50 | 37.0 | 56.7 | 39.8 | 22.0 |
| **FCOS-R101-FPN** | R101 | **41.5** | 60.7 | 45.0 | 24.4 |
| FCOS-R101-FPN-DCN | R101-DCN | 44.7 | — | — | — |

- 首次在 COCO 上以 **无 anchor** 达到与 RetinaNet-101 同级 AP
- 后续 **ATSS、GFL、VFNet** 等多在其分配策略与 loss 上改进

---

## 2. 动机：为何要 FCOS

### 2.1 Anchor-based 的问题

```
RetinaNet / SSD:
  · 每位置多套 anchor（尺度×比例）→ 超参多、设计依赖数据集统计
  · 正负样本靠 IoU 匹配 → 启发式规则
  · 论文证明: anchor 可视为 FCOS 的特例（固定离散偏移），
    并非检测本质所需
```

### 2.2 全卷积检测的直觉

```
语义分割:  每个像素一个类标签
FCOS:      每个像素「若在物体内」→ 预测类 + 到框边的距离

→ 与分割同构，天然 anchor-free、head 结构简单
```

### 2.3 与 CenterNet 的分歧（同 anchor-free，不同正负样本）

| | CenterNet | FCOS |
|--|-----------|------|
| 谁为正 | **一个** 中心峰（高斯） | 框内 **一片** 网格点（+ 中心约束） |
| 定位 | 读 wh | 读 l,t,r,b |
| NMS | 热图 peak，常无 box NMS | **需要** IoU-NMS |
| 多尺度 | 常单输出 stride4 | **FPN 每层管不同尺寸** |

---

## 3. 网络结构详解

### 3.1 Backbone + FPN

与 RetinaNet 相同骨架：

```
ResNet / ResNeXt → C3,C4,C5
FPN → P3(s=8), P4(16), P5(32), P6(64), P7(128)

每层通道 256，后接 **共享** 的检测子网络（可与 RetinaNet head 类似）
```

### 3.2 三个输出分支

对 **每个 FPN 层、每个空间位置 (x,y)**：

#### （1）分类 cls

```
输出: K 维 sigmoid（每类独立，无 softmax background 通道）
损失: Focal Loss（γ=2, α=0.25，同 RetinaNet）
```

#### （2）回归 bbox — l, t, r, b

```
设特征图点映射到原图坐标 (x, y)，对应 GT 框 (x1,y1,x2,y2):

  l = x - x1    （到左边距离）
  t = y - y1    （到上边）
  r = x2 - x    （到右边）
  b = y2 - y    （到下边）

正样本条件: l,t,r,b > 0（点在框内）

解码框:
  x1 = x - l,  y1 = y - t,  x2 = x + r,  y2 = y + b
```

**为何用四边距而非 (cx,cy,w,h)**：

```
· 与网格点 (x,y) 自然对齐，梯度直观
· 点在框外时可为负，便于定义「不在物体内」
· 论文证明与 anchor 回归等价性时可统一分析
```

#### （3）Centerness（中心度）

```
仅正样本监督；推理与 cls 分数相乘

centerness* = sqrt( (min(l,r)/max(l,r)) × (min(t,b)/max(t,b)) )

∈ (0,1]:  点越靠近 GT 中心 → 值越大
           靠近框边角的点 → 值小

作用: 抑制「在框内但远离中心」的低质量预测
      （这类点 l≈0 或 r≈0，框很扁，IoU 对 GT 仍可能不低但定位差）
```

### 3.3 推理分数融合

```
final_score = sigmoid(cls_k) × sigmoid(centerness)

再阈值过滤（如 0.05）→ 解码 bbox → **类内 NMS**（IoU 0.5~0.6）
```

**与 RetinaNet 区别**：RetinaNet 无 centerness，FCOS 必须靠它压掉框边缘的 duplicate 检测。

---

## 4. 正负样本定义（关键细节）

FCOS 的样本分配 **不用 IoU 匹配 anchor**，而用 **几何规则**。

### 4.1 步骤一：点在框内

```
位置 (x,y) 对 GT box 计算 l*,t*,r*,b*
若 min(l*,t*,r*,b*) > 0 → 落在框内（候选正样本）
```

### 4.2 步骤二：中心采样（Center Sampling）

```
并非框内所有点都当正样本！

仅保留 GT 框 **中心附近** 的点:
  中心 (cx, cy)，半径 r = center_sampling_radius × stride
  默认 center_sampling_radius = 1.5

即: 点须在框内，且落在以 (cx,cy) 为圆心、r 为半径的圆内
     （实现常用矩形近似：到中心距离 < 某阈值）

目的: 避免靠近边界的点参与训练 → 减少低质量预测
```

**与 CenterNet 对比**：

```
CenterNet:  仅中心峰（+高斯软标签）
FCOS:       中心 **邻域内多个点** 为正（仍比「整框」小得多）
```

### 4.3 步骤三：FPN 层级 — 尺度限制

```
不同层只负责不同 **尺寸范围** 的 GT（按 max(l*,r*,t*,b*) 或面积）:

  P3 (s=8):   0 ~ 64   px
  P4:         64 ~ 128
  P5:         128 ~ 256
  P6:         256 ~ 512
  P7:         512 ~ ∞

若 GT 在某层超出该层负责的尺度范围 → 该层上此 GT 不产生正样本

→ 解决 FPN **每层都要预测所有物体** 导致的训练冲突
→ 小物体主要在 P3，大物体在 P6/P7
```

### 4.4 步骤四：多 GT 冲突

```
若一点同时落在多个 GT 框内（重叠）:
  分配给 **面积更小** 的 GT

理由: 小物体更应由细粒度层负责；大框套小框时优先检小目标
```

### 4.5 负样本

```
不满足上述正样本条件的点 → 负样本（cls 全 0，不算 bbox/centerness loss）
```

---

## 5. 损失函数

### 5.1 总损失

```
L = L_cls + L_reg + L_centerness

（实现中常对 L_reg 乘系数 λ_reg）
```

### 5.2 L_cls — Focal Loss

```
与 RetinaNet 相同，在所有参与训练的点上:
  正样本: 对应类 y=1
  负样本: 各类 y=0

Focal 抑制大量简单背景点
```

### 5.3 L_reg — IoU Loss（及演进）

```
论文/主流实现:  **IoU Loss** 或 **GIoU Loss**（仅正样本）

  L_reg = 1 - IoU(pred_box, gt_box)

比 Smooth L1 on (l,t,r,b) 更对齐评估指标

正样本的 pred_box 由 (x,y) + 预测的 l,t,r,b 解码得到
```

### 5.4 L_centerness — Binary Cross Entropy

```
仅正样本:
  目标 centerness*（由 GT 框与 (x,y) 几何算出）
  预测 centerness 用 sigmoid

L_ctr = BCE(pred, centerness*)
```

### 5.5 归一化

```
常按 **正样本数量** 或 sum(centerness*) 归一化 reg loss，
使不同图像目标数变化时梯度稳定
```

---

## 6. 训练与推理

### 6.1 训练超参（COCO，R50-FPN 1×）

| 超参 | 值 |
|------|-----|
| 优化器 | SGD，momentum 0.9，wd 1e-4 |
| LR | 0.01（8 GPU × 2 img） |
| Schedule | 1× / 2× COCO 标准 |
| 输入 | 短边 800，长边 ≤1333 |
| Focal | γ=2, α=0.25 |
| center_sampling_radius | 1.5 |

### 6.2 推理流程

```
1. 五层 FPN 各出 cls / ltrb / centerness
2. 每层解码所有位置的框与 score = cls × centerness
3. 合并五层候选（跨层可检同一物体，靠 NMS 去重）
4. score > 0.05
5. 每类 NMS（IoU 0.5）
6. top-100 per image
```

---

## 7. Anchor 是 FCOS 的特例（论文观点）

```
在位置 (x,y) 设一个 anchor 中心与宽高固定，
回归目标可重参数化为 l,t,r,b

→ anchor 只是 **对 (x,y) 的离散采样 + 固定先验框**
→ FCOS 直接在每个网格点预测，省去 anchor 设计

实验: FCOS 无需调 anchor scale/ratio 仍达 SOTA 级
```

---

## 8. 消融与关键结论

### 8.1 各组件

| 去掉 | 影响 |
|------|------|
| FPN 多尺度限制 | 训练冲突，AP 明显下降 |
| center sampling | 边界低质量框增多，AP 降 |
| centerness | 大量 duplicate，AP 大幅下降 |
| IoU loss 改 L1 | 定位 AP75 降 |

### 8.2 vs RetinaNet（同 R101-FPN）

```
FCOS 41.5  vs  RetinaNet 39.1  （论文同期对比）
→ anchor-free + centerness + IoU loss 可打赢 anchor 版
```

---

## 9. 后续改进（读 FCOS 后的延伸）

| 方法 | 相对 FCOS 改什么 |
|------|------------------|
| **ATSS** | 不用手工 scale range，**自适应** 按 anchor 中心距离与 IoU 选正负样本 |
| **GFL** | 将 cls 分数与 IoU 质量统一为 Generalized Focal Loss |
| **VFNet** | IoU-aware cls score |
| **YOLOX** | decoupled head + anchor-free，思想接近 FCOS |

---

## 10. 精读备忘：易混淆点

### 10.1 FCOS 不是「框内每个点都是正样本」

```
必须同时满足:
  ① 在框内  ② 中心采样邻域  ③ 该层尺度范围  ④ 多 GT 时选小框
```

### 10.2 centerness 不是 objectness 通道

```
objectness:  有没有物体（一类二分类）
centerness:  点离 **该框中心** 有多近（几何量）
最终仍要 × 每类 cls
```

### 10.3 仍要 NMS

```
 unlike CenterNet 热图峰
 FCOS 密集预测 → 同物多峰 → 必须 IoU-NMS
```

### 10.4 P3 与 P7 各管什么

```
P3: 小目标（stride 8，特征图大）
P7: 大目标（stride 128，感受野大）
勿与「大特征图检大物体」混淆
```

### 10.5 与 [CenterNet.md](./CenterNet.md)

```
CenterNet:  一个点代表一个物体（峰）
FCOS:       多个点监督同一个物体（中心区域），靠 centerness+NMS 选一个
```

---

## 11. 单阶段检测脉络

```
RetinaNet (2017)   anchor + Focal          → [RetinaNet.md](./RetinaNet.md)
CenterNet (2019)   中心热图                → [CenterNet.md](./CenterNet.md)
FCOS (2019)        每点 ltrb + centerness  ← 本文
ATSS (2020)        自适应样本选择
EfficientDet       anchor 路线             → [EfficientDet.md](./EfficientDet.md)
DETR               无点密集预测            → [DETR/DETR.md](./DETR/DETR.md)
```

---

## 12. 参考资料

- 原论文：[FCOS](https://arxiv.org/abs/1904.01355)（ICCV 2019）
- 前置：[RetinaNet.md](./RetinaNet.md)、[FPN](https://arxiv.org/abs/1612.03144)
- 对比：[CenterNet.md](./CenterNet.md)
- 后续：[ATSS](https://arxiv.org/abs/1912.02424)、[GFL](https://arxiv.org/abs/2006.04388)
- 实现：[tianzhi0549/FCOS](https://github.com/tianzhi0549/FCOS)

---

*文档版本：初稿 | 对应论文 ICCV 2019 FCOS*
