# CenterNet（Objects as Points）

> 本文档用于整理 **Objects as Points**（常被称作 CenterNet）论文精读笔记。  
> 重点：**中心点热图表示**、**Gaussian 目标与 Modified Focal Loss**、**wh/reg 分支解码**，以及相对 anchor-based / FCOS 的**处理方式差异**。

> 相关：[RetinaNet.md](./RetinaNet.md)（anchor + Focal）· [CornerNet](https://arxiv.org/abs/1808.01271)（角点热图，损失同源）

> **同名辨析**：ICCV 2019 另有 Duan 等人 [CenterNet: Keypoint Triplets](https://arxiv.org/abs/1904.06850)（中心点 + 角点对），与本文 **Zhou et al. CVPR 2019** 路线不同；下文默认指 **Objects as Points**。

---

## 1. 概览

### 1.1 基本信息

| 项目 | 内容 |
|------|------|
| 论文 | Objects as Points |
| 作者/机构 | Xingyi Zhou, Dequan Wang, Philipp Krähenbühl（UT Austin） |
| 发表 | CVPR 2019 |
| 任务 | **Anchor-free** 目标检测；同一框架可扩展 **3D 检测、姿态估计、跟踪** |
| 代码 | [xingyizhou/CenterNet](https://github.com/xingyizhou/CenterNet) |

### 1.2 核心思想（一句话）

**把每个物体表示为 bbox 中心点 (x, y) 及少量属性（宽高、亚像素偏移），用关键点估计网络输出 per-class 热图，在热图峰值处读取属性得到框，无需 anchor、无需 IoU-NMS。**

| 对比 | CenterNet |
|------|-----------|
| RetinaNet / SSD | 10⁴+ **anchor**，IoU 匹配 |
| FCOS | 框内 **所有点** 为正样本 |
| CornerNet | **两个角点** 热图 + 配对 |
| **CenterNet** | **一个中心点** 热图 + wh/reg |

### 1.3 整体流水线

```
Input: H×W×3（如 512×512）
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Backbone + Upsample（DLA-34 / Hourglass / ResNet-DCN）     │
│  输出 stride=4 的特征图（分辨率 H/4 × W/4）                   │
└──────────────────────────────────────────────────────────────┘
    │
    ├──────────────────┬──────────────────┬──────────────────┐
    ▼                  ▼                  ▼                  │
┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  heatmap │    │  wh      │    │  reg     │  （可选偏移）   │
│  C×h×w   │    │  2×h×w   │    │  2×h×w   │              │
│  每类峰值 │    │  宽高     │    │  亚像素   │              │
└──────────┘    └──────────┘    └──────────┘              │
    │                  │                  │                  │
    └──────────────────┴──────────────────┘                  │
                           ▼
        训练: Gaussian 热图 + Modified Focal + L1(wh,reg@中心)
        推理: 热图 3×3 maxpool 取峰 → 读 wh/reg → 解码 bbox
                           （无 IoU-NMS）
```

### 1.4 COCO 精度参考（test-dev，bbox）

| 模型 | Backbone | 输入 | AP | FPS (1080Ti) |
|------|----------|------|-----|--------------|
| RetinaNet-101 | R101-FPN | 800 | 39.1 | ~10 |
| CornerNet | Hourglass-104 | 511 | 42.1 | ~3 |
| **CenterNet** | **DLA-34** | 512 | **42.1** | **28** |
| **CenterNet** | ResNet-101-DCN | 512 | **45.1**（多尺度） | ~7 |

- **DLA-34**：AP 与 CornerNet 同级，速度 **~10×** 更快
- 参数效率与简洁推理是其卖点

---

## 2. 动机：物体即点

### 2.1 Anchor-based 的代价

```
RetinaNet 等:
  · 手工设计 anchor 尺度/比例
  · IoU 匹配、正负样本极不平衡（靠 Focal 缓解）
  · 推理 IoU-NMS，超参敏感，且对密集场景易误抑制
```

### 2.2 中心点表示的优势

```
物体 → 单点 (cx, cy) + 属性 {w, h, class, ...}

优点:
  1. 输出空间小: 每类一张 h×w 热图，峰值即目标
  2. 无 anchor、无 pairwise 角点组合（相对 CornerNet）
  3. 热图峰之间的 NMS 用 3×3 local max 即可，**无需 box IoU-NMS**
  4. 同一表示可接 3D 尺寸、深度、关键点、跟踪偏移 → **统一框架**
```

### 2.3 与 FCOS 的「中心」区别

| | FCOS | CenterNet |
|--|------|-----------|
| 正样本区域 | GT 框内 **所有** 特征点 | 仅 **中心点**（热图高斯峰） |
| 分类 | 每点预测类 + centerness | **热图 per-class** |
| 定位 | l,t,r,b 四边距 | **wh** 或中心+宽高 |
| NMS | 通常仍需 | **默认不用 IoU-NMS** |

---

## 3. 网络结构详解

### 3.1 Backbone 与输出分辨率

常用 backbone 与输出 stride：

| Backbone | 特点 | 输出 stride |
|----------|------|-------------|
| **DLA-34** | 默认，速度/精度均衡 | **4** |
| Hourglass-104 | 精度高，慢 | 4 |
| ResNet-101 + **DCN** | 可变形卷积，AP 最高 | 4 |

```
输入 512×512 → 特征图 128×128（stride=4）

三次 upsample（反卷积或 interpolate+conv）接在 backbone 后
→ 保证小物体在特征图上有足够像素
```

### 3.2 三个输出头（Head）

所有 head 共享 backbone 特征，**独立 3×3 conv** 输出：

#### （1）Heatmap 头 — 分类 + 是否存在

```
输出:  C × h × w     C = 类别数（VOC 20，COCO 80）
激活:  sigmoid（每类独立，无 background 通道）
含义:  每个位置是「类 c 物体中心」的置信度
```

#### （2）wh 头 — 宽高（类无关）

```
输出:  2 × h × w     (w, h) 单位: 原图像素（或 stride 缩放后一致）
监督:  仅在 **GT 中心位置** 算 L1 loss
推理:  只在热图峰值处读取 wh
```

#### （3）reg 头 — 亚像素偏移（可选但默认开启）

```
输出:  2 × h × w     (δx, δy)
原因:  GT 中心落在特征图上往往是分数坐标，下采样取整有误差
监督:  L1，仅在中心位置
      δx = gt_cx - floor(gt_cx),  δy 同理

精修后中心:
  cx = (x_peak + δx) × stride
  cy = (y_peak + δy) × stride
```

### 3.3 从中心点到 bbox 解码

```
已知峰值网格坐标 (x, y)、reg、wh、stride=4:

  cx = (x + δx) × stride
  cy = (y + δy) × stride
  w, h = wh(x, y)   （从 wh 图该位置读取）

  x1 = cx - w/2,  y1 = cy - h/2
  x2 = cx + w/2,  y2 = cy + h/2
```

**无 anchor、无 RoI、无 box IoU-NMS** —— 这是相对两阶段检测最大的推理差异。

---

## 4. 训练：热图目标与损失（关键细节）

### 4.1 GT 热图如何绘制（Gaussian）

对每个 GT 框：

```
1. 计算 bbox 中心 (cx, cy)，映射到特征图坐标（÷ stride）
2. 根据目标大小计算半径 r（见 4.2）
3. 在类 c 对应通道上画 2D 高斯:
      Y_c(x,y) = exp( -((x-cx)²+(y-cy)²) / (2σ²) )
   峰值在中心为 1，向外衰减
4. 同类多物体重叠:  取 element-wise **max**（避免互相覆盖削弱峰值）
```

**为何用高斯而非单像素 one-hot**：

```
单像素:  训练极难正样本，且相邻同类实例中心可能挤在一起
高斯:    提供软正样本区域；半径控制「一个峰占多大」→ 隐式处理拥挤
```

### 4.2 高斯半径 r 的计算（CornerNet 同源）

目标：半径内高斯与 bbox 的 IoU 满足阈值（如 **0.7**），且尽量小以免相邻物体热图融合。

```
对 truncated bbox（仅中心附近）求最小 r
使得 Gaussian_bbox IoU ≥ min_overlap（默认 0.7）

物体越大 → r 越大；物体越小 → r 越小
```

**拥挤场景**：两个同类中心近 → 各自半径受限 → 热图尽量 **双峰** 而非连成一片。

### 4.3 Modified Focal Loss（热图分类）

继承 CornerNet，对 **所有像素** 计算，压制 easy background：

```
对每个像素，设 y 为 GT 热图值 ∈ [0,1]，p 为预测 sigmoid:

若 y = 1（中心邻域峰值，实现中常取 y==1 或 y>0.99）:
  L = - (1-p)^α · log(p)          α=2

若 y < 1:
  L = - (1-y)^β · p^α · log(1-p)   β=4

总热图 loss = sum over all pixels / num_objects
（或按正样本数归一化，实现略有差异）
```

**与 RetinaNet Focal 区别**：

```
RetinaNet:  稀疏 anchor，sigmoid per class，γ=2, α=0.25
CenterNet:  密集网格，**GT 是连续高斯** y∈[0,1]，β=4 强调 near-miss 负样本
```

### 4.4 wh 与 reg 的 L1 损失

```
L_wh = (1/N) Σ_{objects} |wh_pred(c) - wh_gt|_1     仅在中心格点
L_reg = (1/N) Σ_{objects} |reg_pred(c) - reg_gt|_1

N = 图像中物体个数
wh/reg 图其余位置 **不参与 loss**（无监督）
```

### 4.5 总损失

```
L = L_heatmap + λ_wh · L_wh + λ_reg · L_reg

默认 λ_wh = 0.1, λ_reg = 1（reg 对定位敏感）
```

---

## 5. 推理流程（逐步）

```
输入图像 → resize（保持 aspect，如 512）→ normalize
────────────────────────────────────────────────────────────

1. 前向
   heatmap[C,h,w], wh[2,h,w], reg[2,h,w]

2. 热图峰值提取（替代 IoU-NMS）
   对每个类 c:
     hm' = MaxPool3×3(hm) == hm   （保留局部极大）
     取 score > 0.1 的峰
     或 Top-K（如 100）峰 per image

3. 对每个峰 (x, y, class=c, score=s):
     δ = reg[:, y, x]
     w,h = wh[:, y, x]
     解码 bbox（见 3.3）

4. 输出 {box, class, score}
   **不做** 类间 IoU-NMS（可选加，通常不需要）

────────────────────────────────────────────────────────────
DLA-34 @ 512: ~28 FPS；峰数 K 控制速度–recall
```

**3×3 maxpool NMS 示意**：

```
热图某通道:
  0.1  0.2  0.1
  0.2  **0.9**  0.3   → 仅保留 0.9 峰，抑制邻域 0.2/0.3
  0.1  0.4  0.2
```

---

## 6. 多任务扩展（同一「点」框架）

论文强调 CenterNet 是 **统一表示**，换 head 即可：

| 任务 | 额外输出 | 说明 |
|------|----------|------|
| **2D 检测** | wh, reg | 上文 |
| **3D 检测** | dim, depth, rot | KITTI 车辆：中心点 + 3D 尺寸/深度/朝向 |
| **人体姿态** | K 个 keypoint 热图 | 每人中心 + 关节热图（Small HRNet 等后续可接） |
| **跟踪** | 位移 offset | 跨帧中心关联 |

```
检测与姿态共享 backbone，仅 head 不同
→ 「Objects as Points」标题的由来
```

---

## 7. 与相关方法对比

### 7.1 CenterNet vs CornerNet

| | CornerNet | CenterNet |
|--|-----------|-----------|
| 表示 | 左上 + 右下 **角点** | **中心点** |
| 配对 | embedding 或启发式 | **不需要** |
| 速度 | 慢（Hourglass） | **DLA 快很多** |
| AP | ~42 | ~42（相当） |

### 7.2 CenterNet vs YOLO / RetinaNet

| | YOLOv3 / RetinaNet | CenterNet |
|--|-------------------|-----------|
| 先验 | anchor / default box | **无** |
| 正样本 | 多 anchor 或全框内点 | **热图峰** |
| 后处理 | IoU-NMS | **heatmap peak NMS** |
| 小目标 | FPN 多层 | 单 stride4 + 高分辨率 head |

### 7.3 局限性

```
· 大物体: 中心点特征仍有效，但 wh 回归范围大、误差敏感
· 极度拥挤: 热图峰可能重叠，依赖高斯半径与 peak NMS
· 单尺度输出（原版）: 极大/极小目标不如 FPN 多尺度（后续工作加 multi-scale CenterNet）
· 宽高各 1 个全局回归: 非常规宽高比物体略弱
```

---

## 8. 消融与实现要点

### 8.1 各组件贡献（定性）

| 去掉 | 影响 |
|------|------|
| reg 偏移 | AP 降，中心量化误差明显 |
| Gaussian 改 one-hot | 训练难收敛，AP 降 |
| wh loss | 框尺寸错乱 |
| 3×3 peak NMS | 重复检测增多 |

### 8.2 训练超参（COCO，DLA-34）

| 超参 | 值 |
|------|-----|
| 优化器 | Adam |
| LR | 1.25e-4（batch 32） |
| Epoch | 140 |
| 输入 | 512×512（随机 scale 0.6~1.3） |
| 数据增强 | 随机 flip、crop、color jitter |
| stride | 4 |

---

## 9. 历史地位与后续

### 9.1 贡献

1. **Anchor-free 极简范式** — 影响 FCOS、ATSS、YOLOX 等「中心」思想；
2. **热图 + 属性** — 与关键点/分割统一，利于 multi-task；
3. **无 IoU-NMS** — 简化推理 pipeline（工业界仍常加 NMS 保底）；
4. 开源 CenterNet 生态（pose、tracking、3D 扩展多）。

### 9.2 后续演进

```
CenterNet (2019)     中心热图 + wh/reg
    ↓
CenterNet2 (2021)    更强调 real-time、改进 backbone
FCOS / ATSS          密集点监督 + 改进匹配
RT-DETR / YOLOv8     端到端或无 NMS 的另一条路
```

---

## 10. 精读备忘：易混淆点

### 10.1 两篇 CenterNet

```
Objects as Points (Zhou, CVPR 2019)  →  本文
Keypoint Triplets (Duan, ICCV 2019)  →  中心+角点三元组，别混
```

### 10.2 「无 NMS」的含义

```
无 **bbox IoU-NMS**
有 **heatmap 3×3 local max** 抑制相邻峰 → 仍是 NMS 思想，在热图空间完成
```

### 10.3 wh 图为何全图预测但只中心监督

```
全图 forward 方便实现；loss mask 仅在 GT 中心
推理也 **只读峰值处** wh/reg，其余位置无意义
```

### 10.4 背景类

```
热图 **无 background 通道**；背景 = 所有类热图接近 0
与 RetinaNet sigmoid 多类类似
```

### 10.5 stride=4 的含义

```
特征图 128×128 对应 512 输入 → 中心定位误差 ±2 像素级
reg 头用于 sub-pixel 补偿
```

### 10.6 与 [RetinaNet.md](./RetinaNet.md) 阅读关系

```
RetinaNet:  anchor 密集 + Focal on sparse matched anchors
CenterNet:  无 anchor，Focal on dense heatmap + 峰即目标
```

---

## 11. 单阶段检测脉络

```
SSD / RetinaNet     anchor + Focal
CornerNet (2018)    角点热图
CenterNet (2019)    中心热图 + wh           ← 本文
FCOS / ATSS (2019+)  每点预测边距
EfficientDet (2020) BiFPN + anchor         → [EfficientDet.md](./EfficientDet.md)
DETR (2020)         query 集合预测         → [DETR/DETR.md](./DETR/DETR.md)
```

---

## 12. 参考资料

- 原论文：[Objects as Points](https://arxiv.org/abs/1904.07850)（CVPR 2019）
- 相关：[CornerNet](https://arxiv.org/abs/1808.01271)、[FCOS](https://arxiv.org/abs/1904.01355)
- 对比：[RetinaNet.md](./RetinaNet.md)、[SSD.md](./SSD.md)
- 实现：[xingyizhou/CenterNet](https://github.com/xingyizhou/CenterNet)

---

*文档版本：初稿 | 对应论文 CVPR 2019 Objects as Points（CenterNet）*
