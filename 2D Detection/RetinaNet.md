# RetinaNet

> 本文档用于整理 RetinaNet（Focal Loss for Dense Object Detection）论文精读笔记。  
> 重点：**Focal Loss 与类别不平衡**、**FPN + 双子网检测头**，以及相对 SSD / 两阶段检测的**匹配策略与训练处理方式**。

> 相关：[SSD.md](./SSD.md)（Hard Negative Mining）· [RCNN/FasterRCNN.md](./RCNN/FasterRCNN.md)（FPN、Anchor）

---

## 1. 概览

### 1.1 基本信息

| 项目 | 内容 |
|------|------|
| 论文 | Focal Loss for Dense Object Detection |
| 作者/机构 | Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotar Dollár（FAIR） |
| 发表 | ICCV 2017 |
| 任务 | **密集预测单阶段检测**；论文核心贡献为 **Focal Loss**，检测器命名为 **RetinaNet** |
| 代码 | [facebookresearch/Detectron](https://github.com/facebookresearch/Detectron)、Detectron2 `RetinaNet` |

### 1.2 核心思想（一句话）

**在 ResNet-FPN 上对 P3–P7 每层密集预测分类与框回归，用 Focal Loss 对易分负样本（背景）降权，使单阶段 detector 在 COCO 上首次全面超过两阶段 Mask R-CNN，而无需 Hard Negative Mining。**

| 痛点 | RetinaNet 解法 |
|------|----------------|
| 单阶段 AP 长期低于 Faster/Mask R-CNN | **FPN** 强化多尺度语义 + 高分辨率 |
| 10⁴~10⁵ anchor 中背景占绝对多数 | **Focal Loss** 自动抑制 easy negative |
| SSD 靠 Hard Negative Mining 近似平衡 | **端到端平滑重加权**，无需 mining |

### 1.3 整体流水线

```
Input: 图像（短边 800，长边 ≤1333 常见）
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Backbone: ResNet（C3, C4, C5）                               │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  FPN: P3, P4, P5, P6, P7（自顶向下 + 横向连接 + 额外降采样）   │
└──────────────────────────────────────────────────────────────┘
    │
    ├────────────────────────────┬─────────────────────────────┐
    ▼                            ▼                             │
┌─────────────────┐    ┌─────────────────┐                  │
│  Classification │    │  Box Regression │   同结构、不同权重  │
│  Subnet (×4 conv)│    │  Subnet (×4 conv)│                  │
│  → A×K sigmoid  │    │  → A×4  offsets   │                  │
└─────────────────┘    └─────────────────┘                  │
    │                            │                             │
    └────────────────────────────┴─────────────────────────────┘
                                 │
                                 ▼
        训练: Focal Loss + Smooth L1（仅正样本回归）
        推理: 阈值 → 类内 NMS → 检测结果
```

### 1.4 COCO 精度参考（test-dev，bbox）

| 模型 | Backbone | 训练尺度 | AP | AP50 | AP75 | APs |
|------|----------|----------|-----|------|------|-----|
| Faster R-CNN FPN ResNet-101 | R101-FPN | 800 | 36.2 | 59.1 | 39.0 | 18.2 |
| Mask R-CNN ResNet-101-FPN | R101-FPN | 800 | 38.2 | 60.3 | 41.7 | 20.1 |
| SSD513 | ResNet-101 | 513 | 31.2 | 50.4 | 33.3 | 10.2 |
| **RetinaNet-50** | ResNet-50-FPN | 600/800 | 32.5 / 34.4 | — | — | — |
| **RetinaNet-101** | ResNet-101-FPN | 800 | **39.1** | **59.1** | **41.7** | **21.8** |

- **RetinaNet-101-800**：首个在 COCO 上 **bbox AP 超过** Mask R-CNN 的单阶段方法（2017）
- 训练时间比两阶段更短，推理速度有竞争力

---

## 2. 动机：类别不平衡（Class Imbalance）

### 2.1 密集检测中的极端不平衡

```
一张图 RetinaNet anchor 总数（量级）:

  P3:  ~80×80×9  ≈ 57k  （stride 8）
  P4~P7 递减
  合计: 约 10⁵ 量级

正样本（匹配 GT）:  通常 几个 ~ 几十个
负样本（背景）:      占 99.9%+
```

**标准交叉熵的问题**：

```
Easy negative:  背景 anchor，p(class)≈0 → CE 很小
Hard positive:  难检物体，CE 较大

但 easy negative 数量极大 → 总 loss 被背景主导
→ 有效梯度几乎来自「海量简单背景」
→ 模型倾向于预测全背景，正样本学不充分
```

### 2.2 现有对策与不足

| 方法 | 做法 | 局限 |
|------|------|------|
| **Hard Negative Mining**（SSD、Faster R-CNN OHEM） | 只保留 loss 最高的负样本 | 非光滑；需调 neg:pos；与 batch 耦合 |
| **Bootstrap** | 高分负样本重训 | 不稳定 |
| **Focal Loss** | 对所有样本连续降权 easy 例 | **RetinaNet 选用** |

### 2.3 Focal Loss 的直觉

```
CE:     关注所有样本，easy negative 数量碾压

Focal:  当模型对某样本已很自信（p_t→1）时，乘因子 (1-p_t)^γ → 0
        → easy 样本 loss 被大幅压低
        → 梯度集中在 hard positive / hard negative
```

---

## 3. Focal Loss 详解（论文第一核心）

### 3.1 二分类 CE 回顾

```
p ∈ [0,1]  模型对 y=1 的估计概率
y ∈ {0,1}  真标签

CE(p, y) = -[ y·log(p) + (1-y)·log(1-p) ]

定义 p_t（对真实类的预测概率）:
  p_t = p      if y=1
  p_t = 1-p    if y=0

则 CE = -log(p_t)
```

### 3.2 Focal Loss 定义

```
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

γ ≥ 0  focusing parameter（论文默认 γ=2）
α_t ∈ [0,1]  平衡因子（正类常用 α=0.25，负类 1-α）
```

**调制因子 (1 - p_t)^γ 的行为**：

| 样本类型 | p_t | (1-p_t)^γ (γ=2) | 效果 |
|----------|-----|-----------------|------|
| Easy positive | 0.9 | 0.01 | loss ×0.01 |
| Easy negative | 0.05 (y=0→p_t=0.95) | 0.0025 | 几乎不计 |
| Hard example | 0.3~0.7 | 0.09~0.49 | **保留主要梯度** |

```
γ=0  →  FL = 标准 CE（加 α 权重）
γ=1  →  中等聚焦
γ=2  →  论文默认，easy 样本权重骤降
γ=5  →  过强，难样本过少时训练不稳
```

### 3.3 多分类扩展（RetinaNet 实现）

RetinaNet 对 **K 个类别用 K 个独立 sigmoid**（非 softmax 含 background）：

```
对每个类别 c，视二分类问题:
  y=1 若 anchor 属于类 c
  y=0 否则（含 background 与其他类）

FL_c(p) = -α_c (1-p)^γ log(p)   正
        - (1-α_c) p^γ log(1-p)   负（对该类）

总分类损失 = Σ_c FL_c，在匹配到的 anchor 上计算
```

**为何不用 (K+1) softmax + background**：

```
softmax: 类间竞争，background 占一类
sigmoid: 每类独立，background = 所有类概率都低
         与 focal「压制 easy 背景」更契合
         且避免「背景类」与物体类数量极不平衡在 softmax 内的二次问题
```

### 3.4 α 与 γ 的作用分工

```
α:  全局平衡正负 **数量**（α=0.25 降低正类权重，因正样本极少）
γ:  平衡难易样本 **质量**（压低 easy，保留 hard）

二者正交，通常固定 α=0.25, γ=2，COCO 上鲁棒
```

### 3.5 与 Hard Negative Mining 的对比

```
Mining:  离散选 top-k 负样本，其余 loss=0
Focal:   连续权重，所有样本都参与，easy 权重≈0

优点:
  · 无需 neg:pos 比例超参
  · 梯度更平滑，实现更简单（Detectron2 标准配置）
  · AP 通常优于同等 backbone 的 SSD mining
```

---

## 4. 网络结构：RetinaNet = FPN + 双子网

### 4.1 Backbone + FPN

```
ResNet 输出:
  C3: stride  8,  1/8  分辨率
  C4: stride 16,  1/16
  C5: stride 32,  1/32

FPN 构建:
  P5 = 1×1 conv(C5)
  P4 = Upsample(P5) + 1×1(C4)
  P3 = Upsample(P4) + 1×1(C3)
  P6 = Conv stride2(C5)      ← 额外层，大物体
  P7 = ReLU(P6) + Conv stride2

输出通道统一 256（典型）
```

**与 Mask R-CNN 中 FPN 一致**，但检测头换成 **密集 anchor 子网**，无 RPN/RoI。

```
特征层级    stride    负责目标尺度
P3          8         小
P4          16        中小
P5          32        中
P6          64        大
P7          128       很大
```

### 4.2 两个「子网」（Subnetwork）

分类与回归 **结构相同、参数不共享**：

```
输入:  P_l  (H_l × W_l × 256)

共享 trunk（各自独立 4 层）:
  3×3 Conv, 256, ReLU  ×4

分类头:
  3×3 Conv → H_l × W_l × (A × K)
  A=9 anchors, K=80 (COCO)
  激活: sigmoid（每类独立）

回归头:
  3×3 Conv → H_l × W_l × (A × 4)
  仅正样本算 loss；类无关回归（4 维/anchor）
```

**参数共享 across levels**：同一子网权重应用于 P3–P7 所有层（不同特征图尺寸，卷积核共享）。

```
好处: 参数量可控；各尺度检测头行为一致
```

### 4.3 与 SSD 检测头的差异

| | SSD | RetinaNet |
|--|-----|-----------|
| 特征 | 手工多层 conv4_3…conv11 | **FPN 融合**语义+分辨率 |
| 分类 | Softmax (C+1) | **Sigmoid × K** + Focal |
| 负样本 | Hard mining | **Focal 降权** |
| 回归 | 类无关 | 类无关（同） |
| head | 每层独立 predictor | **共享权重子网** |

---

## 5. Anchor 设计（关键细节）

### 5.1 每层 9 个 Anchor

```
每个 FPN 层 P_l（base stride 2^l，l=3..7）:

  3 个 aspect ratio:  {0.5, 1, 2}
  3 个 scale factor:  {2^0, 2^(1/3), 2^(2/3)}  相对该层 base scale

  A = 3 × 3 = 9 anchors / cell
```

**每层一个「 octave」内的三尺度**，跨层覆盖 2^3 … 2^7 像素量级（相对原图，再乘输入尺度）。

### 5.2 Anchor 中心

```
与 SSD/Faster R-CNN 相同:
  cx = (j + 0.5) / W_l
  cy = (i + 0.5) / H_l
  归一化到 [0,1] 或像素坐标实现
```

### 5.3 匹配策略（比 SSD 更严）

对每个 anchor，计算与所有 GT 的 **最大 IoU**：

| max IoU | 标签 |
|---------|------|
| ≥ **0.5** | **正样本**，类别 = 对应 GT 类 |
| < **0.4** | **负样本**（背景，所有类 sigmoid 目标为 0） |
| ∈ [**0.4**, 0.5) | **忽略**（不参与 loss） |

**额外规则**（与 Faster R-CNN 类似）：

```
每个 GT 至少匹配一个 anchor：将与其 IoU 最大的 anchor 标为正（即使 <0.5 也强制？）
论文实现: 保证每个 GT 有 best anchor；通常 best IoU 会 ≥0.5
```

**与 SSD 对比**：

| | SSD | RetinaNet |
|--|-----|-----------|
| 正条件 | IoU≥0.5（可多对一） | IoU≥0.5 |
| 中间带 | 无，其余为负 | **0.4~0.5 忽略** |
| 负样本 | 全背景 + mining | IoU<0.4 + focal 自动降权 |

```
0.4~0.5 ignore 带 → 减轻「勉强算负样本」的噪声
```

### 5.4 边框编码

与 Faster R-CNN / SSD 相同（相对 anchor 中心宽高）：

```
t_x = (g_x - a_x) / a_w
t_y = (g_y - a_y) / a_h
t_w = log(g_w / a_w)
t_h = log(g_h / a_h)

损失: Smooth L1，仅正样本 anchor
```

---

## 6. 总损失与训练

### 6.1 损失函数

```
L = (1 / N_pos) · L_cls + (1 / N_pos) · L_box

L_cls = Σ FL(p_t)   在所有参与训练的 anchor 上（正+负），
                    focal 已对 easy neg 降权
L_box = Σ smooth_L1(t - t*)   仅正样本

N_pos = 正样本 anchor 数量（用于归一化，稳定 loss 尺度）
```

**注意**：分类 loss **不需要** 像 SSD 那样只选 3:1 负样本；focal 已隐式完成「难例聚焦」。

### 6.2 训练超参（COCO 典型）

| 超参 | 值 |
|------|-----|
| Backbone | ResNet-50 / 101 + FPN |
| 优化器 | SGD，momentum 0.9，wd 1e-4 |
| LR | 0.01（8 GPU × 2 img），0.1× at 60k, 80k iter |
| 迭代 | 90k |
| 输入 | 短边 600（R50）或 800（R101） |
| γ, α | 2, 0.25 |
| Anchor IoU | pos 0.5 / neg 0.4 / ignore 中间 |
| NMS | 0.5 |
| Score thr | 0.05 |

### 6.3 推理流程

```
1. 整图 → FPN → 各层 cls + box 预测
2. 解码所有 anchor → 候选框（每类 sigmoid 得分）
3. 得分 < 0.05 丢弃
4. 每类 Top-1000 预筛选（可选）
5. 类内 NMS，IoU=0.5
6. 每图最多 100 检测（COCO 标准）
```

---

## 7. 消融实验与关键结论

### 7.1 Focal Loss（γ, α）

| 配置 | COCO AP (R50) |
|------|---------------|
| CE + mining | ~31 |
| CE 无 mining | 很低 |
| **FL γ=2, α=0.25** | **~32.5+** |
| γ=0（=加权 CE） | 明显低于 γ=2 |
| γ=5 | 略降或不稳 |

### 7.2 架构组件

| 去掉/替换 | 影响 |
|-----------|------|
| 无 FPN（仅 C5） | AP 大幅下降，小目标 APs 尤其差 |
| 无 focal（普通 CE） | AP 降 ~10 点量级 |
| 共享 head vs 每层独立 | 共享略优且参数少 |

### 7.3 单阶段 vs 两阶段（论文论点）

```
RetinaNet-101-FPN AP 39.1  vs  Mask R-CNN R101 AP 38.2（同代 benchmark）

说明: 在 COCO 上，**类别不平衡处理** 曾是单阶段主要瓶颈
      FPN 解决多尺度，Focal 解决不平衡
      而非「单阶段结构本质不行」
```

---

## 8. 历史地位与后续影响

### 8.1 贡献

1. **Focal Loss** — 成为 one-stage / 密集预测标准损失（变体：GFL、VFNet、VarifocalLoss）；
2. **RetinaNet 架构** — FPN + 解耦 cls/reg 子网，Detectron2 内置；
3. **证明单阶段可达 SOTA** — 推动后续 YOLO、FCOS、ATSS 等发展；
4. 与 **SSD mining**、**Cascade 两阶段** 形成检测训练的三条主线。

### 8.2 局限

| 局限 | 说明 |
|------|------|
| 计算量 | P3 层 anchor 极多，推理比 YOLO 慢 |
| Anchor 手工 | 后续 ATSS、FCOS 走向 anchor-free |
| 多类 sigmoid | 类间不互斥，罕见「多类高分」需后处理 |
| 超大目标 | 依赖 P6/P7，极罕见 |

### 8.3 后续演进

```
RetinaNet (2017)   FPN + Focal + anchor
    ↓
FCOS (2019)        anchor-free + centerness
ATSS (2020)        自适应样本选择
GFL / VFL (2021)   广义 focal，quality-aware cls
YOLOv8 等          仍可见 focal 思想变体
```

---

## 9. 精读备忘：易混淆点

### 9.1 论文标题是 Focal Loss，模型叫 RetinaNet

```
Focal Loss:  通用分类损失，可用于其他密集任务
RetinaNet:   具体网络 = ResNet-FPN + 双子网 + Focal
```

### 9.2 p_t 在正负样本上的含义

```
正样本 (y=1):  p_t = p（预测为该类的概率）
负样本 (y=0):  p_t = 1-p（预测「非该类」的把握）

easy negative:  p≈0 → p_t≈1 → (1-p_t)^γ≈0
```

### 9.3 RetinaNet 没有单独 background 通道

```
背景 = 所有类 sigmoid 都接近 0
正样本 = 对应类 sigmoid → 1
与 SSD 的 background softmax 类不同
```

### 9.4 Focal 不等于「只训练难例」

```
所有 anchor 仍参与前向；easy 样本 loss 权重极小但非零
与 mining 的「硬截断选 subset」不同
```

### 9.5 回归为何类无关

```
与 SSD 相同: 4 维/anchor，不预测 K×4
分类已区分类别，回归只负责几何精修
```

### 9.6 与 SSD.md 中 mining 的衔接

```
SSD:      离散选 Top 难负例，neg:pos=3:1
RetinaNet: 连续 (1-p_t)^γ 降权 easy 负例

读 SSD 时见 11.5 节 → 本文即其改进路线
```

---

## 10. 单阶段检测脉络（RetinaNet 位置）

```
YOLO v1 (2016)     单尺度 grid
SSD (2016)         多尺度 + mining          → [SSD.md](./SSD.md)
RetinaNet (2017)   FPN + Focal              ← 本文
FCOS / ATSS        anchor-free / 自适应匹配
DETR (2020)        无 anchor、Transformer   → [DETR/DETR.md](./DETR/DETR.md)
```

**与两阶段关系**：

```
Faster/Mask R-CNN:  proposal + RoI，样本少但精
RetinaNet:          密集 anchor + Focal，样本多但加权

COCO 上二者 AP 打平后，路线融合（如 EfficientDet、YOLO+FPN）
```

---

## 11. 参考资料

- 原论文：[Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)（ICCV 2017）
- 前置：[SSD.md](./SSD.md)、[FPN](https://arxiv.org/abs/1612.03144)（CVPR 2017）
- 两阶段对比：[MaskRCNN.md](./RCNN/MaskRCNN.md)
- 实现：[Detectron2 RetinaNet](https://github.com/facebookresearch/detectron2)

---

*文档版本：初稿 | 对应论文 ICCV 2017 RetinaNet / Focal Loss*
