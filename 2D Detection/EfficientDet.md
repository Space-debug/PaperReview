# EfficientDet

> 本文档用于整理 EfficientDet（Scalable and Efficient Object Detection）论文精读笔记。  
> 重点：**BiFPN 加权双向融合**、**EfficientNet backbone**、**复合系数 φ 统一缩放**，以及相对 RetinaNet 的**效率–精度权衡**。

> 相关：[RetinaNet.md](./RetinaNet.md)（FPN、Focal Loss、anchor 头）· [SSD.md](./SSD.md)

---

## 1. 概览

### 1.1 基本信息

| 项目 | 内容 |
|------|------|
| 论文 | EfficientDet: Scalable and Efficient Object Detection |
| 作者/机构 | Mingxing Tan, Ruoming Pang, Quoc V. Le（Google Research） |
| 发表 | CVPR 2020 |
| 任务 | **可扩展、高效率** 单阶段目标检测（COCO） |
| 代码 | [google/automl](https://github.com/google/automl/tree/master/efficientdet) |

### 1.2 核心思想（一句话）

**用 EfficientNet 作 backbone、BiFPN 作可加权双向特征融合 neck，配合 RetinaNet 式 Focal Loss 检测头，并通过复合系数 φ 同时缩放网络深度/宽度/分辨率与 BiFPN 重复次数，在更少参数量与 FLOPs 下达到更高 AP。**

| 模块 | 作用 |
|------|------|
| **EfficientNet** | 复合缩放 backbone，高参数效率 |
| **BiFPN** | 学习权重的双向多尺度融合，替代普通 FPN/PAN |
| **共享检测头** | 类 RetinaNet 子网 + Focal Loss |
| **φ 复合缩放** | D0~D7 一条曲线扫精度–效率前沿 |

### 1.3 整体流水线

```
Input: 分辨率随 φ 增大（512 → 1536+）
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Backbone: EfficientNet-B0 ~ B7（MBConv + Swish）            │
│  输出多尺度特征 {P3, P4, P5, P6, P7}                          │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Neck: BiFPN × N（重复 N 次双向加权融合块，N 随 φ 增加）       │
└──────────────────────────────────────────────────────────────┘
    │
    ├────────────────────────────┬─────────────────────────────┐
    ▼                            ▼                             │
┌─────────────────┐    ┌─────────────────┐                  │
│  Class Subnet    │    │  Box Subnet      │  结构同 RetinaNet  │
│  Focal Loss      │    │  Smooth L1       │                  │
└─────────────────┘    └─────────────────┘                  │
    │                            │                             │
    └────────────────────────────┴─────────────────────────────┘
                                 ▼
                    NMS → 检测结果
```

### 1.4 COCO 精度与效率参考（test-dev，bbox）

| 模型 | φ | 输入 | AP | 参数量 | FLOPs |
|------|---|------|-----|--------|-------|
| RetinaNet-50 | — | 800 | 34.4 | 29M | 239B |
| RetinaNet-101 | — | 800 | 39.1 | 60M | 380B |
| **EfficientDet-D0** | 0 | 512 | **33.8** | **3.9M** | **2.5B** |
| **EfficientDet-D3** | 3 | 896 | **45.6** | 12M | 45B |
| **EfficientDet-D5** | 5 | 1280 | **49.2** | 21M | 130B |
| **EfficientDet-D7** | 7 | 1536 | **52.2** | 52M | 325B |
| EfficientDet-D7x | 7+ | 1536+ | 55.1 | — | — |

- **D0**：用约 1/60 FLOPs 接近 RetinaNet-50 AP
- **D7**：2020 前后 COCO SOTA 梯队，参数量仍低于许多两阶段大模型

---

## 2. 动机：为何要 EfficientDet

### 2.1 检测器的三维权衡

```
精度 (AP)  ←→  参数量 / FLOPs  ←→  延迟 (latency)

问题（2020 前）:
  · 追求 AP → ResNet-101/152 + 大 FPN → 数百 GFLOPs
  · 移动端 / 边缘需要小模型，AP 掉很多
  · backbone、neck、head、输入分辨率 各自缩放，组合爆炸
```

### 2.2 对 FPN 类 neck 的批评

| Neck | 问题 |
|------|------|
| **FPN** | 仅 top-down；**等权 sum** 融合，未区分特征重要性 |
| **PANet** | 加 bottom-up，仍等权；路径固定 |
| **NAS-FPN** | 结构搜索贵、不规则，难部署 |

**BiFPN 目标**：在 **固定、规则、可重复** 的结构下，用 **可学习权重** 做双向融合，并 **堆叠多次** 加深 fusion。

### 2.3 与 RetinaNet 的关系

```
EfficientDet ≈ EfficientNet + BiFPN + RetinaNet-style Head + Compound Scaling

继承 RetinaNet:
  ✓ P3–P7 多尺度 anchor
  ✓ Focal Loss + Smooth L1
  ✓ 分类/回归分离子网

替换/增强:
  ✗ ResNet-FPN  → EfficientNet + BiFPN
  ✗ 手工调各部件尺度 → 统一 φ 复合缩放
```

---

## 3. Backbone：EfficientNet 简述

EfficientDet 的 backbone 来自 [EfficientNet](https://arxiv.org/abs/1905.11946)（ICML 2019），检测论文中不重复所有细节，但需理解接口。

### 3.1 MBConv 与复合缩放

```
基本块 MBConv:
  1×1 expand → Depthwise 3×3 → SE（可选）→ 1×1 project
  激活: Swish (x·sigmoid(x))

EfficientNet 用系数 φ_b 同时缩放:
  · depth（层数）  α^φ
  · width（通道）  β^φ
  · resolution     γ^φ
  约束 α·β²·γ² ≈ 2（FLOPs 约翻倍）
```

### 3.2 检测用多尺度输出

```
EfficientNet-Bφ 截取 stage 特征:
  P3: stride 8
  P4: stride 16
  P5: stride 32
  再经额外 conv 得到 P6, P7（大目标层，同 RetinaNet）

通道数随 φ 增大（width multiplier）
```

---

## 4. BiFPN：论文第一核心（结构 + 加权融合）

### 4.1 对普通 FPN 的三条改进

```
1. 删除「仅单输入、无特征融合价值」的节点 → 简化图
2. 同一尺度层加 skip：原始 backbone 特征直连到输出 → 保留细节
3. 在 top-down + bottom-up 之上，用 **可学习权重** 做快速归一化融合
4. 将 BiFPN 块 **重复堆叠** 多次（depth 方向复合缩放）
```

### 4.2 BiFPN 单层拓扑（示意）

```
        P5_in ────────────────┐
           │    top-down      │
           ▼                  │
        P4_td ←── fuse(P5_up, P4_in)
           │                  │
           ▼                  │
        P3_out ←── fuse(P4_up, P3_in)     ← 最高分辨率输出
           │                  │
           │    bottom-up     │
           ▼                  │
        P4_out ←── fuse(P3_out_down, P4_td, P4_in)
           │                  │
           ▼                  │
        P5_out ←── fuse(P4_out_down, P5_in, P5_td)
           │                  │
           ▼                  │
        P6, P7 ←── 由 P5_out 降采样得到（stride 更大）
```

**双向含义**：

```
Top-down:  语义强的深层 → 上采样 → 与浅层融合（大物体语义指导）
Bottom-up: 浅层定位信息 → 下采样 → 与深层再融合（精确定位回传）
```

### 4.3 加权融合（Fast Normalized Fusion）

传统 FPN/PAN 多为 **等权相加**：

```
O = I1 + I2 + I3
```

BiFPN 引入 **可学习标量权重** w_i（每层、每 fusion 节点独立）：

```
O = ( Σ_i  ReLU(w_i) · I_i ) / ( Σ_i  ReLU(w_i) + ε )

I_i:  参与融合的输入特征（已 resize 到同一分辨率）
w_i:  可学习，ReLU 保证非负
ε:    1e-4，数值稳定
```

**为何 ReLU(w_i) 而非 softmax**：

```
softmax:  权重和强制为 1，训练慢、对 GPU 不友好
ReLU + 归一化:  和可变化，实验 AP 更好、更快（论文 ablation）
```

**三路融合例子**（P4_out）：

```
P4_out = Fuse( P4_in,  Upsample(P3_out),  Downsample(P5_path) )
       = (w1·P4_in + w2·Up(P3_out) + w3·Down(P5) + ε) / (w1+w2+w3+ε)
```

### 4.4 特征变换（fusion 前后）

每个节点融合前后通常接：

```
1×1 Conv + BN + Swish（统一通道到 W_bifpn，如 64/88/112…随 φ 增）
Depthwise 3×3 + BN + Swish（轻量空间混合）
```

### 4.5 重复堆叠（BiFPN depth）

```
第 1 个 BiFPN block:  输入来自 backbone P3–P7
第 2~N 个 block:      输入 = 上一 block 输出的 P3_out…P7_out

N = 3 + φ  （EfficientDet 规则，φ=0 → 3 层重复）

→ φ 越大，特征融合越深，参数量与 FLOPs 上升
```

### 4.6 BiFPN vs FPN vs PANet

| | FPN | PANet | BiFPN |
|--|-----|-------|-------|
| 方向 | 仅 top-down | +bottom-up | **双向** |
| 融合权重 | 1:1 sum | 1:1 sum | **学习 w_i** |
| 重复 | 1 次 | 1 次 | **N 次堆叠** |
| skip | 横向 1×1 | 同左 | **+ 原始输入 skip** |
| 参数效率 | 中 | 中 | **高（DW conv）** |

---

## 5. 检测头与损失（继承 RetinaNet）

### 5.1 Head 结构

与 RetinaNet 相同范式：

```
对每个 BiFPN 输出层 P_l:
  Class subnet: 4× (Conv3×3 + BN + Swish) → Conv3×3 → A×K sigmoid
  Box subnet:   4× (Conv3×3 + BN + Swish) → Conv3×3 → A×4

A = 9 anchors / cell（3 scale × 3 aspect ratio，同 RetinaNet）
子网权重 **跨层共享**
head 重复层数随 φ:  L_head = 3 + φ
```

### 5.2 损失函数

```
L = L_cls + L_box

L_cls:  Focal Loss，γ=2, α=0.25，K 类 sigmoid（同 RetinaNet）
L_box:  Smooth L1，仅正样本 anchor

Anchor 匹配（同 RetinaNet 常见设置）:
  正: max IoU ≥ 0.5
  负: max IoU < 0.4
  忽略: 0.4 ~ 0.5
```

### 5.3 边框编码

```
与 Faster R-CNN / RetinaNet / SSD 一致:
  t_x = (g_x - a_x) / a_w,  t_y = (g_y - a_y) / a_h
  t_w = log(g_w / a_w),     t_h = log(g_h / a_h)
```

---

## 6. 复合系数 φ：统一缩放（论文第二核心）

### 6.1 为何要 compound scaling

```
手工调参:
  ResNet50 vs 101、FPN channel 256 vs 512、输入 600 vs 800…
  → 组合空间巨大，难以找最优效率点

EfficientDet:
  一个整数 φ ∈ {0,1,…,7} 驱动整条网络
```

### 6.2 φ 控制的维度（概念表）

| 随 φ 增大而放大 | 规则（论文） |
|----------------|-------------|
| Backbone | EfficientNet-Bφ（更深更宽） |
| BiFPN width | W = 64 + φ×8（约） |
| BiFPN depth | N = 3 + φ 次重复 |
| Head depth | 3 + φ 层 conv |
| 输入分辨率 | 512 + φ×128（约，到 1536） |
| Anchor / 训练 | 随分辨率调整（实现中有对应表） |

**D0 vs D7 直观**：

```
D0:  B0 + 3×BiFPN + 512²  →  3.9M params, AP 33.8
D7:  B6 + 10×BiFPN + 1536² → 52M params, AP 52.2
```

### 6.3 缩放曲线意义

```
在 COCO 上画出 AP–FLOPs 曲线:
  EfficientDet 系列点 **压在** RetinaNet、YOLO、两阶段 的 Pareto 前沿上

→ 同样 AP 更少算力；同样算力更高 AP
```

---

## 7. 训练与推理

### 7.1 训练设置（COCO 典型）

| 超参 | 值 |
|------|-----|
| 优化器 | **AdamW**（检测里较少见，EfficientNet 系常用） |
| 学习率 | 0.08·batch/64，cosine decay |
| Weight decay | 1e-4 |
| Epoch | 300（大模型） |
| 增强 | **AutoAugment**、随机缩放裁剪 |
| EMA | 可选，推理用 EMA 权重 |
| 框架 | 原论文 TensorFlow，开源含 PyTorch 移植 |

### 7.2 推理流程

```
与 RetinaNet 相同:
  1. resize 到 φ 对应分辨率
  2. EfficientNet → BiFPN×N → head
  3. 解码 anchor → score 过滤（0.05）
  4. 类内 NMS 0.5
  5. top-100 per image
```

### 7.3 延迟优化（工程）

```
· BiFPN 用 depthwise separable conv 降 FLOPs
· 导出 TensorFlow / TFLite / ONNX 用于边缘
· D0~D2 面向实时；D5~D7 面向精度榜
```

---

## 8. 消融实验与关键结论

### 8.1 BiFPN 组件

| 配置 | AP 变化 |
|------|---------|
| FPN baseline | 基准 |
| + bottom-up (PAN) | 小幅 + |
| + 加权 fusion | 明显 + |
| + 重复 BiFPN | 继续 + |
| **完整 BiFPN** | 最佳 |

### 8.2 加权融合方式

| 融合 | 结果 |
|------|------|
| Sum（等权） | 较低 |
| Softmax 权重 | 慢、AP 略低 |
| **Fast normalized fusion (ReLU w)** | **最好** |

### 8.3 Backbone 替换

```
ResNet-50 + BiFPN  →  AP 提升
EfficientNet-B0 + BiFPN →  更少参数达更高 AP

→ BiFPN 与 EfficientNet **协同**，非简单叠加
```

### 8.4 Compound scaling

```
只放大 backbone 不放大 BiFPN/分辨率 → 收益饱和
统一 φ 缩放 → AP–FLOPs 曲线最优
```

---

## 9. 历史地位与局限

### 9.1 贡献

1. **BiFPN** — 加权双向融合成为高效 neck 代表（后续 YOLOv8 等 neck 设计受其影响）；
2. **EfficientDet 缩放律** — φ 统一 backbone/neck/head/分辨率；
3. **效率标杆** — 推动社区关注 **AP per FLOP / per param**；
4. 验证 **单阶段 + Focal** 路线可登顶 COCO（D7）。

### 9.2 局限

| 局限 | 说明 |
|------|------|
| TensorFlow 起源 | 早期生态不如 PyTorch 顺手（后有移植） |
| Anchor-based | 仍依赖 9 anchor/层；后续 YOLOX/FCOS 去 anchor |
| 大 φ 训练贵 | D6/D7 需大分辨率 + 长 schedule |
| NAS 系延伸 | EfficientDet-D7x 等需更强算力与调参 |

### 9.3 后续演进

```
EfficientDet (2020)     EfficientNet + BiFPN + φ
    ↓
YOLOv8/v9 等            改进 neck（C2f、PAFPN）与 scaling
    ↓
RT-DETR、YOLO-World     效率 + 端到端 / 开放词汇
```

---

## 10. 精读备忘：易混淆点

### 10.1 BiFPN 不是「又一个 FPN 名字」

```
FPN:   单向 + 等权，通常 1 遍
BiFPN: 双向 + 学权 + 堆叠 N 遍 + 加权公式分母归一化
```

### 10.2 Fusion 权重 w_i 是 per-edge 的

```
不是全局一个 w，而是每个融合节点有自己的 {w1,w2,w3}
不同尺度、不同方向（td/bu）各自学习
```

### 10.3 EfficientDet 没有改检测 loss

```
分类仍 Focal、回归仍 Smooth L1
提升主要来自 **特征质量（BiFPN）+ backbone 效率 + 缩放策略**
```

### 10.4 φ 与 EfficientNet 的 φ_b 关系

```
EfficientNet 论文: φ_b 只缩放分类 backbone
EfficientDet:      φ 缩放 **整检测器**（含 BiFPN、head、分辨率）

符号不要混: EfficientDet-D3 的「3」是检测复合系数
```

### 10.5 D0 的 AP 33.8 并不低

```
3.9M 参数、2.5B FLOPs 下接近 RetinaNet-50（29M、239B）
→ 读 EfficientDet 必看 **效率坐标**，不单看 AP 绝对值
```

### 10.6 与 RetinaNet.md 阅读顺序

```
RetinaNet:  学会 FPN 检测头 + Focal + anchor 匹配
EfficientDet:  在同样 head 上换 EfficientNet+BiFPN，并系统缩放
```

---

## 11. 单阶段检测脉络（EfficientDet 位置）

```
SSD (2016)          多尺度 + mining
RetinaNet (2017)    FPN + Focal          → [RetinaNet.md](./RetinaNet.md)
EfficientDet (2020) EfficientNet + BiFPN + φ  ← 本文
YOLOv5/v8           另一套 backbone/neck 工程化
DETR (2020)         无 anchor             → [DETR/DETR.md](./DETR/DETR.md)
```

---

## 12. 参考资料

- 原论文：[EfficientDet](https://arxiv.org/abs/1911.09070)（CVPR 2020）
- Backbone：[EfficientNet](https://arxiv.org/abs/1905.11946)（ICML 2019）
- 前置：[RetinaNet.md](./RetinaNet.md)、[SSD.md](./SSD.md)
- Neck 相关：[FPN](https://arxiv.org/abs/1612.03144)、[PANet](https://arxiv.org/abs/1803.08189)
- 实现：[google/automl/efficientdet](https://github.com/google/automl/tree/master/efficientdet)

---

*文档版本：初稿 | 对应论文 CVPR 2020 EfficientDet*
