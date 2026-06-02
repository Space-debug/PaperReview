# YOLO11

> 本文档用于深入学习 YOLO11：**相对 YOLOv8 的架构升级**、**C3k2 / C2PSA 注意力模块**、**轻量化 Detect 分类头**，以及训练/推理与扩展任务。  
> 检测头、TAL、DFL、Loss 等与 v8 相同的细节见 [YOLOv8 笔记](./YOLOv8.md)。  
> 代码：[ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) · 官方文档：[YOLO11](https://docs.ultralytics.com/models/yolo11/)

---

## 1. 基线：YOLO11 概览

### 1.1 基本信息

| 项目 | 内容 |
|------|------|
| 作者/机构 | Ultralytics（Glenn Jocher, Jing Qiu 等） |
| 发布时间 | **2024 年 9 月**（YV24 大会发布） |
| 任务 | detect / seg / pose / obb / cls（与 v8 相同五任务） |
| 代码 | `pip install ultralytics`，`YOLO("yolo11n.pt")` |
| 论文 | **无正式论文**，工程迭代 |
| 相对 v8 | **C3k2 替代 C2f + C2PSA 注意力 + DWConv 分类头** |
| 后续版本 | YOLO26（2025，NMS-free end2end，见 [YOLOv8 §20](./YOLOv8.md)） |

### 1.2 整体架构（相对 v8 的变化标注）

> **详细差异见 §1.5~§1.7（升级动机与四大升级点）、§3.4（yaml 逐行对照）、§8（完整差异总表）。**

```
Input (B, 3, 640, 640)
    │
    ▼
    │                                              ↑ 新增 P5 注意力
    ▼
Neck: PAN-FPN（C3k2 + Upsample + Concat）★ 模块全换 C3k2
    │
    ▼
Head: Detect（解耦头，Anchor-free）
    │   Reg 分支 cv2: 不变（Conv×2 + 1×1）
    │   Cls 分支 cv3: ★ DWConv 轻量化（legacy=False）
    ▼
Output（3 尺度 P3/P4/P5）:
  与 v8 相同: (B, 64+nc, H, W)，reg_max=16，无 obj 分支
```

### 1.3 官方精度（COCO val2017，640）

| 模型 | mAP@0.5:0.95 | 参数量(M) | FLOPs(B) | CPU ONNX(ms) | T4 TRT10(ms) | vs YOLOv8 同档 mAP | vs v8 参数量 |
|------|--------------|-----------|----------|--------------|--------------|-------------------|--------------|
| YOLO11n | **39.5** | 2.6 | 6.5 | 56.1 | 1.5 | v8n 37.3 **+2.2** | v8n 3.2 **-19%** |
| YOLO11s | **47.0** | 9.4 | 21.5 | 90.0 | 2.5 | v8s 44.9 **+2.1** | v8s 11.2 **-16%** |
| YOLO11m | **51.5** | 20.1 | 68.0 | 183.2 | 4.7 | v8m 50.2 **+1.3** | v8m 25.9 **-22%** |
| YOLO11l | **53.4** | 25.3 | 86.9 | 238.6 | 6.2 | v8l 52.9 **+0.5** | v8l 43.7 **-42%** |
| YOLO11x | **54.7** | 56.9 | 194.9 | 462.8 | 11.3 | v8x 53.9 **+0.8** | v8x 68.2 **-16%** |

**核心卖点：** 更少参数、更高 mAP、GPU 推理速度基本持平或略快。官方强调 YOLO11m 比 YOLOv8m **参数少 22%、mAP 更高**。

### 1.4 相对 YOLOv8 的核心变革（速览）

| 维度 | YOLOv8 | YOLO11 | 好处 |
|------|--------|--------|------|
| Backbone 块 | C2f | **C3k2** | 单块 MACs ≈ C2f 的 30%，参数更少 |
| P5 末端 | SPPF 结束 | SPPF + **C2PSA** | P5 语义增强；n 档 1×PSABlock，l/x 档 2× |
| Neck 块 | C2f | **C3k2** | 与 backbone 统一 |
| Cls 分支 | Conv×2 + 1×1 | **DWConv + Conv** | cv3 参数约为 v8 的 **16%** |
| legacy 标志 | True | **False** | 由 C3k2 解析时自动触发 |
| depth 缩放(n) | 0.33 | **0.50** | 配合 yaml repeat 减少，总块数相近 |
| 标签分配 | TAL | TAL（不变） | 训练 API 完全兼容 |
| 框回归 | DFL + CIoU | DFL + CIoU（不变） | reg_max=16 不变 |
| Loss | v8DetectionLoss | v8DetectionLoss（不变） | 同一套 hyp |

### 1.5 升级动机：不是「换模块」，而是「重分配参数预算」

YOLO11 的设计目标：**在相近或更低 FLOPs 下提高 mAP**。实现路径不是单纯堆层，而是：

```
                    YOLOv8n 参数预算 ~3.16M
                    ─────────────────────────
  Backbone C2f×多次 repeat     ~45%   ← 大量参数耗在重复 Bottleneck
  Neck C2f                     ~30%
  Detect cv3 (legacy Conv)     ~16%   ← 分类头很重
  SPPF + 早期 Conv             ~9%

                    YOLO11n 参数预算 ~2.62M（-17%）
                    ─────────────────────────
  Backbone C3k2(e=0.25)        ~35%   ← 单块更轻，但 P3/P4 通道更宽
  Neck C3k2                    ~25%
  C2PSA（新增）                ~14%   ← 从 cv3 省下的预算投给注意力
  Detect cv3 (DWConv)          ~3%    ← 大幅缩减
  Detect cv2（不变）           ~15%
  SPPF + Conv                  ~8%
```

**关键洞察：** v11 **减** 了 cls 头和 C2f 重复堆叠，**增** 了 C2PSA 和 backbone 部分 stage 的通道宽度（如实测 v11n P3 backbone 128ch vs v8n 64ch），在「更宽但更浅的特征提取 + P5 全局注意力」之间做 trade-off。

### 1.6 四大升级点详解（相对 v8）

---

#### 升级点 ①：C2f → C3k2（全网骨干 + Neck）

**相同点：** 二者共用 **C2f 骨架**——`cv1 扩通道 → chunk 二分 → 一支旁路 + 一支过 Bottleneck 链 → concat → cv2 压回`。C3k2 **继承** C2f，forward 逻辑完全一致：

```python
# C2f / C3k2 共用 forward（block.py）
y = list(self.cv1(x).chunk(2, 1))      # [part1, part2]
y.extend(m(y[-1]) for m in self.m)     # 链式 Bottleneck / C3k
return self.cv2(torch.cat(y, 1))
```

**不同点（三处）：**

| # | 差异 | YOLOv8 | YOLO11 | 影响 |
|---|------|--------|--------|------|
| 1 | yaml **repeat** | 3/6/6/3（n 档） | **2/2/2/2** | v8 单 stage 堆更多 block |
| 2 | **depth_multiple** | n=**0.33** | n=**0.50** | 实际 block 数 = round(repeat×depth) |
| 3 | **expansion e** | 默认 **0.5** | 小模型 **0.25** | hidden 通道减半 |
| 4 | **Block 类型** | 固定 Bottleneck | n/s: Bottleneck；**m/l/x: C3k** | 大模型换 3×3 大核 |

**repeat 实际计算（yolo11n，depth=0.50）：**

| yaml 行 | 模块 | yaml repeat | 实际 n = max(round(r×d),1) |
|---------|------|-------------|---------------------------|
| backbone P2 | C3k2 | 2 | **1** |
| backbone P3 | C3k2 | 2 | **1** |
| backbone P4 | C3k2 | 2 | **1** |
| backbone P5 | C3k2 | 2 | **1** |
| neck ×4 | C3k2 | 2 | **1** |

**对比 yolov8n（depth=0.33）：**

| yaml 行 | 模块 | yaml repeat | 实际 n |
|---------|------|-------------|--------|
| backbone P2 | C2f | 3 | **1** |
| backbone P3 | C2f | **6** | **2** |
| backbone P4 | C2f | **6** | **2** |
| backbone P5 | C2f | 3 | **1** |

v8 **P3/P4 各 2 个 Bottleneck**，v11 各 **1 个**；但 v11 单块更轻（e=0.25），且 P3 backbone 输出 **128ch**（v8 仅 **64ch**）。

**Bottleneck vs C3k（m/l/x 自动启用）：** 公式推导见 **§2.2**；C3k 参数量约为单 Bottleneck 的 **1.1×**，换更大感受野，并非更轻。

**实测特征图通道（640 输入，forward hook）：**

| Stage | YOLOv8n 输出 | YOLO11n 输出 | 说明 |
|-------|-------------|-------------|------|
| P2 (160²) | 32 | **64** | v11 更宽 |
| P3 (80²) backbone | 64 | **128** | v11 2× 通道 |
| P4 (40²) backbone | 128 | **128** | 相同 |
| P5 (20²) SPPF 后 | 256 | 256 | 相同 |
| Neck → Detect P3 | 64 | 64 | 融合后对齐 |
| Neck → Detect P4 | 128 | 128 | 相同 |
| Neck → Detect P5 | 256 | 256 | 相同 |

Neck 输出尺度与 v8 **一致**，Detect 头输入通道相同，**可直接复用 TAL/DFL/后处理**；差异在「如何提取到这些特征」。

---

#### 升级点 ②：C2PSA（Backbone P5 末端新增）

**v8：** SPPF 输出直接进 Neck。  
**v11：** SPPF → **C2PSA** → Neck；PAN 下采样拼 P5 时用 **layer 10**（C2PSA 后），不是 layer 9（SPPF 后）。

**为何放 P5、不放 P3？**

| 因素 | P5 (20×20) | P3 (80×80) |
|------|-----------|-----------|
| 分辨率 N=H×W | 400 | 6400 |
| Self-Attn O(N²) | 160K | **41M**（不可接受） |
| 语义层级 | 高，适合全局上下文 | 低，更依赖局部卷积 |
| 目标类型 | 大目标、场景理解 | 小目标、边缘细节 |

**C2PSA forward 逐步（yolo11n：C=256, H=W=20）：**

```
输入 x                           (1, 256, 20, 20)

cv1: Conv 1×1 → 256→256         split → a(128), b(128)
                                  ↑ 半通道旁路，不经过注意力

b → PSABlock #1:                 （yolo11n 仅 1 块；l/x 有 #2）
      b + Attention(b)
      b + FFN(b)

cv2: concat [a, b] → 256         Conv 1×1 融合
输出                             (1, 256, 20, 20)   shape 不变
```

**与 PSA / C2fPSA 对比：**

| 模块 | 结构 | 在 YOLO11 中的位置 |
|------|------|-------------------|
| PSA | 单块 attn+ffn | 未直接使用 |
| C2fPSA | C2f 内嵌 PSA Bottleneck | 未使用 |
| **C2PSA** | split 旁路 + 堆叠 PSABlock | **backbone 末端 ×2** |

旁路 `a` 保留纯卷积特征，避免注意力「洗平」局部细节；`b` 支路做全局 mixing。

**代价：** +367K 参数（yolo11n 约 **14%**），+149M MACs（约 **2.3%** FLOPs）。用较小计算换 P5 语义，Neck 自 layer10 取特征时已含注意力增强。

---

#### 升级点 ③：Detect cls 分支 DWConv 化（legacy=False）

**触发机制（`nn/tasks.py` → `parse_model`）：**

```python
legacy = True                    # 默认：v3/v5/v8/v9 用旧 cv3
...
if m is C3k2:
    legacy = False               # ★ 遇到 C3k2 后，Detect 用新 cv3
...
if m in {Detect, ...}:
    m.legacy = legacy            # 写入 Detect 类变量
```

只要 yaml 含 C3k2（YOLO11 全部含），Detect 自动 **legacy=False**。Reg 分支 **cv2 完全不变**。

**同输入通道 (64,128,256) 下 cv2/cv3 参数量（nc=80）：**

| 分支 | legacy cv3 (v8) | DWConv cv3 (v11) | 变化 |
|------|----------------|-----------------|------|
| cv2 三尺度合计 | 381.9K | **381.9K** | **不变** |
| cv3 三尺度合计 | 515.8K | **83.0K** | **-84%** |
| Detect 头合计 | 897.7K | **464.9K** | **-48%** |

**单尺度 P3（in=64, c3=80）结构对比：**

```
v8 legacy cv3:
  Conv(64→80, 3×3) → Conv(80→80, 3×3) → Conv2d(80→80, 1×1)
  参数量 ≈ 110.5K

v11 DWConv cv3:
  [DWConv(64,3,g=64) → Conv(64→80,1×1)]
  → [DWConv(80,3,g=80) → Conv(80→80,1×1)]
  → Conv2d(80→80, 1×1)
  参数量 ≈ 19.9K
```

**为何只轻量化 cls、不轻量化 reg？**

| 分支 | 任务 | 需求 | v11 策略 |
|------|------|------|----------|
| cv2 reg | 四边 DFL 分布，亚像素定位 | 强空间建模 | **保持 Conv 3×3×2** |
| cv3 cls | 80 类语义，粗粒度 | 通道混合为主 | **DWConv 提空间 + 1×1 混通道** |

分类只需「这个 cell 像不像某类」；回归要「到四条边的距离分布」，对空间精度更敏感。

**c2/c3 通道公式（与 v8 相同，head.py）：**

```python
c2 = max(16, ch[i]//4, reg_max*4)     # reg 隐层，通常 64
c3 = max(ch[i], min(nc, 100))         # cls 隐层，随尺度增大
```

---

#### 升级点 ④：Compound Scaling 策略调整

| 型号 | v8 depth | v11 depth | v8 width | v11 width | v8 max_ch | v11 max_ch |
|------|----------|-----------|----------|-----------|-----------|------------|
| n | 0.33 | **0.50** | 0.25 | 0.25 | 1024 | 1024 |
| s | 0.33 | **0.50** | 0.50 | 0.50 | 1024 | 1024 |
| m | 0.67 | **0.50** | 0.75 | **1.00** | **768** | **512** |
| l | 1.00 | 1.00 | 1.00 | 1.00 | 512 | 512 |
| x | 1.00 | 1.00 | 1.25 | **1.50** | 512 | 512 |

**影响解读：**

- **n/s：** depth 从 0.33→0.50 但 yaml repeat 从 3~6 降到 2，**实际 Bottleneck 数相近或更少**；width 不变。
- **m：** width 0.75→1.0 但 max_ch 768→512，**宽通道被 cap 截断**，配合 C3k2 效率换精度。
- **x：** width 1.25→1.5，更大模型更宽；m/l/x 自动 **c3k=True**。

**m/l/x 专属：parse_model 自动 c3k=True**

```python
if m is C3k2 and scale in "mlx":
    args[3] = True    # C3k2 内部 Block 换 C3k
```

---

### 1.7 未改动的部分（与 v8 完全相同）

以下组件 YOLO11 **零修改**，迁移训练/推理时可直接参考 [YOLOv8 笔记](./YOLOv8.md)：

| 组件 | 说明 |
|------|------|
| TaskAlignedAssigner | α=0.5, β=6.0, topk=10 |
| v8DetectionLoss | box + cls + dfl，软标签 |
| DFL reg_max=16 | Integral 解码 |
| Anchor-free | 8400 预测点，grid 中心 |
| NMS | conf=cls_max，iou=0.7 |
| 增广 / hyp | mosaic、mixup、hyp.yaml 同一套 |
| Segment/Pose/OBB Head | 继承 v8，仅 backbone 换 C3k2 |

**对用户的影响：** `yolo detect train` 命令、data.yaml、超参、导出流程 **与 v8 一致**，仅 `model=yolo11*.pt` 和权重文件不同。

---

## 2. 核心模块详解

### 2.1 C3k2：C2f 的正式继任者

C3k2 **继承 C2f**，在 v8.2 中作为可选变体出现，YOLO11 **全面默认使用**。

```
Input (C1)
  │
  Conv 1×1 → 2×C_hidden        ← c = int(c2 × e)
  │
  Split → [part1 | part2]
  │
  part2 → Block₁ → Block₂ → ... → Block_n
  │
  Concat [part1, Block₁_out, ..., Block_n_out]    通道 = (2+n)×c
  │
  Conv 1×1 → C_out
```

**参数量粗算（单层 C3k2，忽略 BN）：**

```
设 c1, c2, hidden c = c2×e, n 个 Block

cv1:  c1 × 2c × 1²
每个 Bottleneck: 2 × (c×c×1² + c×c×3²) ≈ 20c²
cv2:  (2+n)c × c2 × 1²

e=0.25 vs e=0.5 → c 减半 → 所有 c² 项变为 1/4
```

**Block 类型（由 yaml args 决定）：**

| args | 内部 Block | 使用场景 |
|------|-----------|----------|
| `[c2, False, 0.25]` | 标准 **Bottleneck** + e=0.25 | n/s 小模型，hidden 通道压缩 |
| `[c2, True]` | **C3k**（3×3 核堆叠） | m/l/x 大模型，更大感受野 |
| `[c2, False]` + attn | Bottleneck + **PSABlock** | 可选注意力（部分 yaml） |

**源码（`nn/modules/block.py`）：**

```python
class C3k2(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, ...):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(Bottleneck(...), PSABlock(...)) if attn
            else C3k(self.c, self.c, 2, ...) if c3k
            else Bottleneck(self.c, self.c, ...) 
            for _ in range(n)
        )
```

**parse_model 自动规则（`nn/tasks.py`）：**

```python
if m is C3k2:
    legacy = False          # ★ 触发 Detect 新 cls 头
    if scale in "mlx":      # m/l/x 型号
        args[3] = True      # 自动 c3k=True
```

**C3k2 vs C2f 对比：**

| | C2f (v8) | C3k2 (v11) |
|---|----------|------------|
| 结构骨架 | 相同 C2f 分流 concat | 相同 |
| forward | `chunk → chain → cat` | **完全相同** |
| 默认 Block | Bottleneck k=(1,3) | Bottleneck 或 **C3k k=(3,3)** |
| 小模型 e | 0.5 | **0.25**（hidden 减半） |
| yaml repeat (n档) | 3/6/6/3 | **2/2/2/2** |
| 实际 block 数 (n档) | 1/2/2/1 per stage | **1/1/1/1** |
| parse_model 副作用 | legacy 保持 True | **legacy=False** |

### 2.1.1 源码继承关系

```python
class C2f(nn.Module):
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C3k2(C2f):          # 仅重写 __init__，替换 self.m 的 Block 类型
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, ...):
        super().__init__(c1, c2, n, shortcut, g, e)   # 建 cv1/cv2
        self.m = ModuleList(...)                       # 覆盖 Bottleneck 链
```

C3k2 **不重写 forward**，与 C2f 100% 相同的特征融合路径；差异仅在 `self.m` 里装什么 Block、以及 `e` 和 `n` 取值。

### 2.2 C3k vs Bottleneck：参数量与 MACs 公式推导

YOLO11 **n/s** 的 C3k2 内部用标准 **Bottleneck k=(1,3)**；**m/l/x** 自动换 **C3k**（内含 2× Bottleneck k=(3,3)）。二者并非「C3k 更省参数」——实测 **C3k 参数量约为单 Bottleneck 的 1.1×**，换取的是 **更大感受野**。

#### 2.2.1 约定：Ultralytics Conv 参数量

`Conv(c1, c2, k, s, g)` 使用 `bias=False`，参数量（不含 BN）：

```
Params_Conv = c1 × c2 × k² / g

MACs_Conv ≈ 2 × H × W × c1 × c2 × k² / g    （乘加各算 1）
```

BN 参数 `2×c2`（γ, β）相对 Conv 可忽略，下文**只计 Conv weight**。

#### 2.2.2 标准 Bottleneck 推导

**源码（block.py）：**

```python
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3,3), e=0.5):
        c_ = int(c2 * e)              # 内部 hidden
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
```

**C3k2 / C3 内部调用方式：** `Bottleneck(c, c, shortcut, g, k=..., e=1.0)`  
→ `c1=c2=c_`，`e=1.0` → 内部 hidden 仍为 **c**。

**参数量（c1=c2=c，groups=1）：**

```
Params = Params_cv1 + Params_cv2
       = c × c × k1² + c × c × k2²
       = c² × (k1² + k2²)
```

| 核组合 | k1 | k2 | 系数 | 公式 | c=64 实测 |
|--------|----|----|------|------|----------|
| **标准** k=(1,3) | 1 | 3 | 1+9=**10** | **10c²** | 41,216 |
| **C3k 内部** k=(3,3) | 3 | 3 | 9+9=**18** | **18c²** | 73,984 |
| 比值 (3,3)/(1,3) | | | 1.8 | | **1.79×** |

**MACs（特征图 H×W 不变）：**

```
MACs_Bottleneck ≈ 2HW × c² × (k1² + k2²)

k=(1,3):  20c²HW
k=(3,3):  36c²HW     ← 1.8× 于 (1,3)
```

**c=64, H=W=40 实测：** Bottleneck(1,3) MACs=66.4M；(3,3)=118.8M。

**结构示意：**

```
标准 Bottleneck k=(1,3)          C3k 内单个 Bottleneck k=(3,3)
─────────────────────────          ────────────────────────────────
  c ── Conv1×1 ── c                c ── Conv3×3 ── c
         │                                  │
         └── Conv3×3 ── c                   └── Conv3×3 ── c
              (+shortcut)                         (+shortcut)

  先 1×1 再 3×3，感受野 3×3          两层 3×3 堆叠，感受野 5×5 等效
```

#### 2.2.3 C3k 模块完整推导

**C3k 继承 C3**，在 C3 框架内把 `m` 中的 Bottleneck 换成 **k=(3,3)**，且默认 **n=2**：

```python
class C3k(C3):
    c_ = int(c2 * e)                                    # 默认 e=0.5
    cv1 = Conv(c1, c_, 1)                               # 分支 1
    cv2 = Conv(c1, c_, 1)                               # 分支 2（旁路）
    cv3 = Conv(2*c_, c2, 1)                             # 融合
    m   = Sequential(Bottleneck(c_, c_, k=(3,3), e=1.0) × n)
```

**C3k2 中的调用：** `C3k(self.c, self.c, n=2, shortcut, g)` → `c1=c2=c`，`e=0.5`，`n=2`。

记 **c_ = ⌊c × e⌋ = c/2**（YOLO11 中 c 为 8 的倍数）。

**各项参数量：**

| 组件 | 公式 | 以 c 表示 |
|------|------|----------|
| cv1 | c × c_ × 1² | **c²/2** |
| cv2 | c × c_ × 1² | **c²/2** |
| cv3 | 2c_ × c × 1² | **c²** |
| 每个 Bottleneck(c_,c_,k=(3,3),e=1.0) | 18 × c_² | **4.5c²** |
| n=2 个 Bottleneck | 2 × 4.5c² | **9c²** |
| **C3k 合计** | | **11c²** |

```
Params_C3k = c²/2 + c²/2 + c² + n × 18 × (c/2)²
           = 2c² + n × 4.5c²

n=2 时: Params_C3k = 11c²
Ratio vs Bottleneck(1,3) = 11/10 = 1.10
```

| c | Bottleneck(1,3) | C3k(n=2) | 比值 |
|---|-----------------|----------|------|
| 16 | 2,624 | 2,944 | 1.12 |
| 64 | 41,216 | 45,568 | 1.11 |
| 256 | 656,384 | 722,944 | 1.10 |

**MACs：** c=64, H=W=40 实测 C3k 73.7M vs Bottleneck(1,3) 66.4M → **1.11×**。

#### 2.2.4 为何 m/l/x 仍选 C3k？（+10% 参数换感受野）

| 指标 | Bottleneck k=(1,3) | C3k (2× k=(3,3) + C3 双分支) |
|------|-------------------|------------------------------|
| 参数量 | 10c² | **11c² (+10%)** |
| MACs | 20c²HW | **~22c²HW (+10%)** |
| 等效感受野 | ~3×3 | **~5×5** |
| C3 双分支 | 无 | **有**（cv1 支路 + cv2 旁路 concat） |
| 适用 | n/s 小模型 | m/l/x 大模型 |

```python
if m is C3k2 and scale in "mlx":
    args[3] = True    # Block 换 C3k(c,c,2)
```

#### 2.2.5 嵌入 C3k2 时的总参数（回顾）

```
hidden c = int(c2 × e)           # yolo11n 小模型 e=0.25

Params_C3k2 = c1×2c + (2+n)×c×c2 + Block
              └ cv1 ┘  └── cv2 ──┘   n 个 Block

Block = 10c²   (Bottleneck 1,3, n/s)
      = 11c²   (C3k, m/l/x)
```

**数值例（yolo11n，c1=128, c2=64, n=1, Bottleneck）：** c=16 → 合计 **≈9.7K**（§12 实测）。同 c 换 C3k 仅 Block +256（+10%）。

#### 2.2.6 小结

| 模块 | 参数量 | 在 YOLO11 中 |
|------|--------|-------------|
| Bottleneck(1,3) | **10c²** | n/s 的 C3k2 |
| Bottleneck(3,3) | **18c²** | C3k 内部单元 |
| C3k(n=2, e=0.5) | **11c²** | m/l/x 的 C3k2 |

**关键结论：** v11 相对 v8 的参数节省来自 **e=0.25、更少 repeat、DWConv cv3**，**不是** C3k 比 Bottleneck 更轻。

### 2.3 C2PSA：Backbone 末端注意力

YOLO11 在 SPPF 之后插入 **C2PSA**（仅 backbone P5，不在 neck 重复）。

**源码（block.py）：**

```python
class C2PSA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        self.c = int(c1 * e)                    # hidden = C/2
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = Sequential(*(PSABlock(self.c, ...) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)                           # 仅 b 走注意力
        return self.cv2(torch.cat((a, b), 1))
```

```
SPPF 输出 (B, C, 20, 20)
    │
    ▼
C2PSA ×2（yaml repeat=2；yolo11n 实际 n=1 即 1×PSABlock）
```

**yaml `- [-1, 2, C2PSA, [1024]]` 与 repeat 规则：**

```
yaml repeat=2 × depth_multiple → 插入 C2PSA 的 n 参数

yolo11n (depth=0.50): n = round(2×0.5) = 1  →  1×PSABlock
yolo11s (depth=0.50): n = 1
yolo11l (depth=1.00): n = round(2×1.0) = 2  →  2×PSABlock
yolo11x (depth=1.00): n = 2
```

```
    cv1: 1×1 → split [a | b]   （各 C/2 通道）
    b → PSABlock → PSABlock    （n=2）
    cv2: concat [a, b] → 1×1
    ▼
输出 → 送入 Neck 上采样
```

**与 C2fPSA 的区别：** C2PSA 是独立模块，专门堆叠 PSABlock；C2fPSA 是在 C2f 的 Bottleneck 链中嵌入 PSA。

### 2.4 Attention + PSABlock

YOLO11 使用的是 **卷积式多头自注意力**（非 ViT 的 Linear Q/K/V），输入输出均为 `(B,C,H,W)`，与检测特征图天然兼容。

**Attention 完整 forward（源码）：**

```python
def forward(self, x):                    # x: (B, C, H, W)
    B, C, H, W = x.shape
    N = H * W
    qkv = self.qkv(x)                    # Conv 1×1: C → (2·nh·kd + nh·hd)
    q, k, v = qkv.view(B, nh, 2*kd+hd, N).split([kd, kd, hd], dim=2)
    # q,k: (B, nh, kd, N)   v: (B, nh, hd, N)

    attn = (q.transpose(-2,-1) @ k) * scale    # (B, nh, N, N)
    attn = attn.softmax(dim=-1)
    out = (v @ attn.transpose(-2,-1))          # (B, nh, hd, N)
    out = out.view(B, C, H, W) + self.pe(v.view(B,C,H,W))   # + 深度可分离 PE
    return self.proj(out)                      # Conv 1×1
```

| 参数 | 公式 / yolo11n P5 | 含义 |
|------|------------------|------|
| num_heads | `max(c // 64, 1)` → 128//64=**2** | head 数 |
| head_dim | `c // num_heads` → **64** | 每 head 维度 |
| key_dim | `head_dim × attn_ratio` → **32** | Q/K 压缩，减 O(N²) 开销 |
| scale | `key_dim ** -0.5` | 缩放因子 |

**yolo11n C2PSA 内 PSABlock（c=128, H=W=20）：**

```
N = 400
nh = 2, kd = 32, hd = 64

Attention 主项 MACs:
  Q@K:  2 × 2 × 400 × 400 × 32  ≈ 20.5M
  Attn@V: 2 × 2 × 400 × 400 × 64 ≈ 41.0M
FFN:  Conv 128→256→128, 400 px  ≈ 52M
单层 PSABlock ≈ 130M，C2PSA 含 2 层 ≈ 260M + cv1/cv2
```

**PSABlock = Attention + FFN + 残差：**

```python
class PSABlock(nn.Module):
    def forward(self, x):
        x = x + self.attn(x)    # 残差
        x = x + self.ffn(x)     # FFN: Conv1×1 → Conv1×1
        return x
```

**计算量提示：** C2PSA 只在 **20×20** 的 P5 特征图上运行，分辨率低，注意力开销可控（不像在 80×80 P3 上全局 attn 那样爆炸）。

### 2.5 Detect 分类头：legacy=False 的 DWConv 结构

YOLO11 解析 yaml 时遇到 C3k2 会将 `legacy=False` 写入 Detect 类变量，**仅 cv3 变化**，cv2/cv4/DFL 不变。

**Detect 初始化通道逻辑（v8/v11 共用）：**

```python
c2 = max(16, ch[i]//4, reg_max*4)    # reg 隐层 → yolo11n 三尺度均为 64
c3 = max(ch[i], min(nc, 100))        # cls 隐层 → 64/128/256
```

**v8 legacy=True（旧 cv3）— 每层：**

```
Conv(c_in→c3, 3×3) → Conv(c3→c3, 3×3) → Conv2d(c3→nc, 1×1)
```

**v11 legacy=False（新 cv3）— 每层：**

```
Stage1: DWConv(c_in, 3×3, g=c_in) → Conv(c_in→c3, 1×1)
Stage2: DWConv(c3,  3×3, g=c3)    → Conv(c3→c3,  1×1)
Stage3: Conv2d(c3→nc, 1×1)
```

**DWConv 参数量：** `c × k²`（groups=c）；标准 Conv：`c_in × c_out × k²`

以 P3 为例 (c_in=64, c3=80)：
- legacy 第一个 Conv3×3: 64×80×9 = **46K**
- DWConv: 64×9 = **576** + Conv1×1: 64×80 = **5K**

**yolo11n Detect 头参数实测：**

| 尺度 | ch | cv2 | cv3 (legacy) | cv3 (DWConv) |
|------|-----|-----|-------------|--------------|
| P3 | 64 | 78.1K | 110.5K | **19.9K** |
| P4 | 128 | 115.0K | 156.6K | **25.7K** |
| P5 | 256 | 188.7K | 248.7K | **37.4K** |
| **合计** | | **381.9K** | **515.8K** | **83.0K** |

Reg 分支 `cv2` 仍为 `Conv 3×3 → Conv 3×3 → 1×1`，因为定位需要更强空间建模，不做轻量化。

**源码（`nn/modules/head.py`）：**

```python
self.cv3 = (
    nn.ModuleList(Conv×3 结构...) if self.legacy
    else nn.ModuleList(
        nn.Sequential(
            nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
            nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
            nn.Conv2d(c3, self.nc, 1),
        ) for x in ch
    )
)
```

| | legacy cv3 | DWConv cv3 |
|---|-----------|------------|
| 3×3 卷积类型 | 标准 Conv（c_in→c_out） | **DWConv**（groups=channels） |
| 参数量 | 较高 | **显著降低** |
| 表达能力 | 基线 | DWConv 提取空间 + 1×1 混合通道，足够分类 |

Reg 分支 `cv2` 仍为 `Conv 3×3 → Conv 3×3 → 1×1`，因为定位需要更强空间建模，不做轻量化。

### 2.6 其他模块（与 v8 相同）

Conv（BN+SiLU）、Bottleneck、SPPF、DFL、DWConv 定义均与 v8 一致，详见 [YOLOv8 §2](./YOLOv8.md)。

---

## 3. Backbone 与 Neck

### 3.1 Backbone 逐层（YOLO11n，640 输入）

```
层    操作                        输出尺寸           说明
 0    Conv 3×3 s=2                320×320×64
 1    Conv 3×3 s=2                160×160×128
 2    C3k2 ×2  [256,F,0.25]       160×160×64       e=0.25 窄 hidden
 3    Conv 3×3 s=2                 80×80×128       → P3 源 (layer 4 前)
 4    C3k2 ×2  [512,F,0.25]        80×80×128       → P3 源 (layer 4)
 5    Conv 3×3 s=2                 40×40×256       → P4 源 (layer 6 前)
 6    C3k2 ×2  [512,T]             40×40×256       → P4 源 (layer 6)，c3k=True 仅 m/l/x
 7    Conv 3×3 s=2                 20×20×512
 8    C3k2 ×2  [1024,T]             20×20×512
 9    SPPF                          20×20×512
10    C2PSA ×2                      20×20×512       ★ 新增，Neck Concat 用 layer 10
```

**与 v8 backbone 对比：**

| 位置 | YOLOv8n | YOLO11n |
|------|---------|---------|
| P2 | C2f ×3 | C3k2 ×2 (e=0.25) |
| P3 | C2f ×6 | C3k2 ×2 |
| P4 | C2f ×6 | C3k2 ×2 |
| P5 | C2f ×3 + SPPF | C3k2 ×2 + SPPF + **C2PSA** |
| 总 repeat | 更多 Bottleneck | 更少 repeat，C3k2 更高效 |

### 3.2 Neck（PAN-FPN）

```
FPN:
  layer10(P5) → Upsample → Concat layer6(P4) → C3k2×2 → Upsample → Concat layer4(P3) → C3k2×2 → N3(P3)

PAN:
  N3 → Conv↓ → Concat N4_feat → C3k2×2 → N4(P4)
  N4 → Conv↓ → Concat layer10 → C3k2×2 → N5(P5)

Detect: [N3, N4, N5]  →  stride 8 / 16 / 32
```

**关键索引变化（相对 v8）：**

| Concat 目标 | YOLOv8 | YOLO11 |
|-------------|--------|--------|
| Neck 上采样拼 P5 | layer **9** (SPPF) | layer **10** (C2PSA 后) |
| 其余 P4/P3 | layer 6 / 4 | 相同 |

Neck 中 P5 大尺度分支使用 `C3k2 [1024, True]`（m/l/x 为 C3k block），小尺度分支 `[256/512, False]`。

### 3.3 模型缩放（yaml scales）

```yaml
# cfg/models/11/yolo11.yaml
scales:
  n: [0.50, 0.25, 1024]   # depth, width, max_channels
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]
```

| 型号 | depth | width | max_ch | 层数 | 参数量 | GFLOPs |
|------|-------|-------|--------|------|--------|--------|
| n | 0.50 | 0.25 | 1024 | 181 | 2.62M | 6.6 |
| s | 0.50 | 0.50 | 1024 | 181 | 9.46M | 21.7 |
| m | 0.50 | 1.00 | 512 | 231 | 20.1M | 68.5 |
| l | 1.00 | 1.00 | 512 | 357 | 25.3M | 87.6 |
| x | 1.00 | 1.50 | 512 | 357 | 57.0M | 196.0 |

**与 v8 缩放对比：**

| 型号 | v8 depth | v11 depth | v8 width | v11 width |
|------|----------|-----------|----------|-----------|
| n | 0.33 | **0.50** | 0.25 | 0.25 |
| s | 0.33 | **0.50** | 0.50 | 0.50 |
| m | 0.67 | **0.50** | 0.75 | **1.00** |
| l | 1.00 | 1.00 | 1.00 | 1.00 |
| x | 1.00 | 1.00 | 1.25 | **1.50** |

YOLO11 通过 **C3k2 更高效的结构 + 注意力 + DWConv 头** 实现「depth 看似更大、参数却更少」。

### 3.4 Yaml 逐行对照（yolov8n vs yolo11n）

| 段 | 行 | YOLOv8n | YOLO11n | 差异说明 |
|----|-----|---------|---------|----------|
| BB | 0-1 | Conv×2 下采样 | 相同 | 无变化 |
| BB | 2 | C2f **[128]×3** | C3k2 **[256,F,0.25]×2** | 块类型+args；v11 P2 输出 **64ch** vs v8 **32ch** |
| BB | 3 | Conv 256 s=2 | Conv 256 s=2 | 相同 |
| BB | 4 | C2f **[256]×6** | C3k2 **[512,F,0.25]×2** | v8 repeat×depth=2；v11 n=1；v11 P3 **128ch** |
| BB | 5 | Conv 512 s=2 | 相同 | |
| BB | 6 | C2f **[512]×6** | C3k2 **[512,T]×2** | v11 P4 yaml 预置 c3k=True（m/l/x 生效） |
| BB | 7 | Conv 1024 s=2 | 相同 | |
| BB | 8 | C2f **[1024]×3** | C3k2 **[1024,T]×2** | |
| BB | 9 | SPPF | SPPF | 相同 |
| BB | **10** | — | **C2PSA [1024]×2** | **v11 独有** |
| Neck | up+concat P4 | cat **L6** | cat **L6** | 相同索引 |
| Neck | C2f/C3k2 | C2f [512]×3 | C3k2 [512,F]×2 | |
| Neck | up+concat P3 | cat **L4** | cat **L4** | 相同 |
| Neck | | C2f [256]×3 | C3k2 [256,F]×2 | → Detect P3 **64ch** |
| PAN | down+concat | cat **L12** | cat **L13** | 索引因多 C2PSA 层偏移 +1 |
| PAN | | C2f [512]×3 | C3k2 [512,F]×2 | → Detect P4 **128ch** |
| PAN | down+concat P5 | cat **L9** SPPF | cat **L10** C2PSA | **关键索引变化** |
| PAN | | C2f [1024]×3 | C3k2 [1024,T]×2 | → Detect P5 **256ch** |
| Head | Detect | L15,L18,L21 | L16,L19,L22 | 输入通道相同 |

**自定义 yaml 注意：** 若在 v11 backbone 增删层，**PAN 拼 P5 的 from 索引**必须指向 C2PSA 输出（非 SPPF），否则 Neck 拿不到注意力特征。

### 3.5 特征图逐层实测（640 输入，forward hook）

| Layer | YOLOv8n | YOLO11n |
|-------|---------|---------|
| L2 backbone | 32 @ 160² | **64** @ 160² |
| L4 P3 src | 64 @ 80² | **128** @ 80² |
| L6 P4 src | 128 @ 40² | 128 @ 40² |
| L8/L9 P5 | 256 @ 20² | 256 @ 20² |
| L10 | — | **C2PSA** 256 @ 20² |
| Neck→Det P3 | 64 @ 80² | 64 @ 80² |
| Neck→Det P4 | 128 @ 40² | 128 @ 40² |
| Neck→Det P5 | 256 @ 20² | 256 @ 20² |

v11 **backbone 中段更宽、Neck 融合后对齐**，Detect 头无需任何改动即可接相同 stride/ch。

---

## 4. Head：Detect（与 v8 共用逻辑 + cv3 差异）

YOLO11 检测头在 **TAL 分配、DFL 解码、Loss、NMS** 上与 v8 **字节级相同**；唯一结构差异是 §2.5 的 **cv3 DWConv 化**（legacy=False）。以下分「不变」与「变化」两部分说明。

### 4.1 完全不变的部分

| 组件 | 说明 |
|------|------|
| **cv2 reg 分支** | Conv3×3→Conv3×3→1×1，输出 4×reg_max=64 通道 |
| **DFL** | reg_max=16，Integral 解码四边距离 |
| **Anchor-free** | 每 cell 1 点，8400 预测点 |
| **bias_init** | box bias=2.0，cls bias=log(5/nc/stride²) |
| **训练 raw 输出** | (B, 144, H, W) = 64 reg + 80 cls |
| **推理 decode** | dist2bbox + sigmoid(cls) → (B, 84, 8400) |
| **NMS** | conf=max(cls)，iou=0.7 |

### 4.2 唯一变化：cv3 cls 分支

见 §2.5 参数表。训练时 cls logits 仍走 BCE + TAL 软标签，**Loss 公式不变**；变的是提取 cls 特征的网络结构更轻。

### 4.3 输出格式

```
训练 raw: (B, nc + reg_max×4, H, W) = (B, 144, H, W)   COCO nc=80
推理 decode: (B, 4+nc, 8400) = (B, 84, 8400)
  [0:4]  xywh（letterbox 尺度）
  [4:84] cls sigmoid
```

### 4.4 Anchor-free + DFL

- 每 cell 一个参考点（grid 中心 + 0.5 偏移）
- 四边距离用 **16 档 DFL 分布**回归（与 v8 相同）
- 无 objectness 分支，置信度 = **max(cls)**

### 4.5 解耦头数据流（v11）

```
P3(64ch,80²) / P4(128ch,40²) / P5(256ch,20²)
    │
    ├─ cv2 [不变] → (B, 64, H, W)   bbox DFL logits
    │
    └─ cv3 [DWConv] → (B, 80, H, W)   cls logits
         ↓ permute + concat
    训练: (B, 144, N)  →  TAL + v8DetectionLoss
    推理: DFL decode + sigmoid → NMS
```

### 4.6 Segment / Pose / OBB 的 Head

扩展任务 Head（Segment/Pose/OBB）**继承 v8 实现**，仅在 yaml 中 backbone+neck 换 C3k2；例如 Segment 仍用 Proto+cv4 coeff，Pose 仍用 cv4 关键点。详见 [YOLOv8 §13~§16](./YOLOv8.md)。

---

## 5. 标签分配与损失（继承 v8）

| 组件 | 类/函数 | 说明 |
|------|---------|------|
| 标签分配 | TaskAlignedAssigner | α=0.5, β=6.0, topk=10 |
| 检测 Loss | v8DetectionLoss | box + cls + dfl |
| 框 Loss | BboxLoss + CIoU + DFLoss | 与 v8 相同 |
| Cls Loss | BCE × 软标签 | 负样本也参与 |

**训练命令与 v8 完全兼容，仅换权重名：**

```bash
yolo detect train data=coco.yaml model=yolo11n.pt epochs=100 imgsz=640
yolo detect val model=yolo11s.pt data=coco.yaml
yolo detect predict model=yolo11m.pt source=images/ conf=0.25 iou=0.7
```

**从 v8 迁移：** 同一套 `data.yaml`、增广、超参可直接使用；`yolov8n.pt` 权重**不能**直接加载到 yolo11.yaml（结构不同），需重新训练或使用 COCO 预训练 yolo11*.pt。

---

## 6. 推理全流程

与 YOLOv8 相同：Letterbox → 前向 → DFL 解码 → conf 过滤 → NMS(iou=0.7) → 坐标还原。

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model("bus.jpg", conf=0.25, iou=0.7)
results[0].show()
```

**导出：**

```bash
yolo export model=yolo11s.pt format=onnx simplify=True
yolo export model=yolo11s.pt format=engine half=True device=0
```

**IMX500 边缘部署：** 官方支持 YOLO11n（与 YOLOv8n 并列），通过 `"C2PSA" in model.__str__()` 识别 YOLO11 配置。

---

## 7. 扩展任务概览

YOLO11 五任务 yaml 均在 `cfg/models/11/`，Backbone+Neck **与 detect 共享**，仅 Head 不同。

| 任务 | 权重示例 | Head | 额外输出 | 详解 |
|------|----------|------|----------|------|
| detect | yolo11n.pt | Detect | bbox+cls | 本文 §4 |
| segment | yolo11n-seg.pt | Segment | +mask coeff+proto | [YOLOv8 §13](./YOLOv8.md) |
| pose | yolo11n-pose.pt | Pose（非 Pose26） | +17 keypoints | [YOLOv8 §16](./YOLOv8.md) |
| obb | yolo11n-obb.pt | OBB | xywhr | [YOLOv8 §14](./YOLOv8.md) |
| classify | yolo11n-cls.pt | Classify | 类概率 | [YOLOv8 §19](./YOLOv8.md) |
| **open-vocab** | **yoloe-11s.pt** | **YOLOEDetect** | 文本/视觉 prompt | **§14** |

### 7.1 各任务官方精度摘要

**Segment（COCO，640）：**

| 模型 | mAP_box | mAP_mask | 参数量(M) |
|------|---------|----------|-----------|
| YOLO11n-seg | 38.9 | 32.0 | 2.9 |
| YOLO11s-seg | 46.6 | 37.8 | 10.1 |
| YOLO11m-seg | 51.5 | 41.5 | 22.4 |

**Pose（COCO，640）：**

| 模型 | mAP_pose | mAP_pose@0.5 | 参数量(M) |
|------|----------|--------------|-----------|
| YOLO11n-pose | 50.0 | 81.0 | 2.9 |
| YOLO11s-pose | 58.9 | 86.3 | 9.9 |
| YOLO11m-pose | 64.9 | 89.4 | 20.9 |

**OBB（DOTAv1，1024）：**

| 模型 | mAP_test@0.5 | 参数量(M) |
|------|--------------|-----------|
| YOLO11n-obb | 78.4 | 2.7 |
| YOLO11s-obb | 79.5 | 9.7 |
| YOLO11m-obb | 80.9 | 20.9 |

**Classify（ImageNet，224）：**

| 模型 | Top-1 Acc | 参数量(M) |
|------|-----------|-----------|
| YOLO11n-cls | 70.0% | 2.8 |
| YOLO11s-cls | 75.4% | 6.7 |
| YOLO11m-cls | 77.3% | 11.6 |

**注意：** YOLO11-pose 使用标准 **Pose + v8PoseLoss**，**不含** YOLO26 的 Pose26/RLE（见 [YOLOv8 §20](./YOLOv8.md)）。

---

## 8. YOLO11 vs YOLOv8 差异总表（详细版）

### 8.1 架构与模块

| 维度 | YOLOv8 | YOLO11 | 差异级别 |
|------|--------|--------|----------|
| Backbone 块 | C2f | **C3k2** | ★★★ 核心 |
| Backbone P5 末端 | SPPF | SPPF + **C2PSA** | ★★★ 新增 |
| Neck 块 | C2f | **C3k2** | ★★★ 核心 |
| Reg 头 cv2 | Conv×2 | Conv×2 | 无变化 |
| Cls 头 cv3 | legacy Conv×2 | **DWConv** | ★★ 重要 |
| Detect.legacy | True | **False** | 由 C3k2 触发 |
| 层数 (n) | 23 | **24** | +1（C2PSA） |
| 参数量 (n) | 3.16M | **2.62M** | -17% |
| FLOPs (n) | 8.9G | **6.6G** | -26% |

### 8.2 训练与损失（无差异）

| 维度 | YOLOv8 | YOLO11 |
|------|--------|--------|
| 标签分配 | TaskAlignedAssigner topk=10 | 相同 |
| Loss 类 | v8DetectionLoss | 相同 |
| box loss | CIoU + DFL | 相同 |
| cls loss | BCE × 软标签 | 相同 |
| 增广 | Mosaic, Mixup, HSV... | 相同 |
| hyp.yaml | box/cls/dfl/mosaic... | 相同 |
| 预训练 | yolov8*.pt | **yolo11*.pt**（不可交叉 load） |
| 命令 | `yolo detect train model=...` | **完全相同** |

### 8.3 推理与后处理（无差异）

| 维度 | YOLOv8 | YOLO11 |
|------|--------|--------|
| Letterbox | 相同 | 相同 |
| 预测点数 | 8400 | 8400 |
| conf 阈值 | max(cls) | 相同 |
| NMS iou | 0.7 默认 | 相同 |
| 输出 Results | boxes, masks, kpts... | 相同 API |

### 8.4 导出与部署

| 维度 | YOLOv8 | YOLO11 |
|------|--------|--------|
| ONNX 输出 | (1, 84, 8400) detect | 相同 |
| TensorRT / FP16 | 支持 | 支持 |
| IMX500 | YOLOv8n | **YOLO11n**（靠 `"C2PSA" in str(model)` 识别） |
| 识别 yaml 版本 | 无 C2PSA | 含 **C2PSA** → YOLO11 |

### 8.5 精度与速度（COCO 640）

| 模型 | v8 mAP | v11 mAP | Δ mAP | v8 参数 | v11 参数 | v8 TRT ms | v11 TRT ms |
|------|--------|---------|-------|---------|----------|-----------|------------|
| n | 37.3 | **39.5** | +2.2 | 3.2M | **2.6M** | ~1.5 | ~1.5 |
| s | 44.9 | **47.0** | +2.1 | 11.2M | **9.4M** | ~2.5 | ~2.5 |
| m | 50.2 | **51.5** | +1.3 | 25.9M | **20.1M** | ~4.7 | ~4.7 |
| l | 52.9 | **53.4** | +0.5 | 43.7M | **25.3M** | ~6.2 | ~6.2 |
| x | 53.9 | **54.7** | +0.8 | 68.2M | **56.9M** | ~11.3 | ~11.3 |

**规律：** mAP 全面提升，参数量显著下降（l 档 -42%），GPU TRT 延迟基本持平。

### 8.6 何时选 v8 / v11

| 场景 | 推荐 | 原因 |
|------|------|------|
| 全新项目 | **YOLO11** | 更高 mAP、更少参数 |
| 已有 v8 微调权重 | 继续 **v8** 或重训 v11 | 权重不兼容 |
| 教程/社区资料 | 两者均可 | v8 资料更多；v11 官方主推 |
| 边缘 ultra-light | yolo11n | 2.6M vs 3.2M |
| 开放词汇 | **YOLOE-11** | 见 §14 |
| NMS-free 部署 | YOLO26 | 见 §9 |

### 8.7 速查对照（原 §8 精简版）

| 模块 | YOLOv8 | YOLO11 |
|------|--------|--------|
| Backbone 块 | C2f | **C3k2** |
| P5 注意力 | 无 | **C2PSA**（n 档 1 block，l/x 档 2 block） |
| Neck 块 | C2f | **C3k2** |
| Cls 头 | Conv×2 | **DWConv×2 + Conv 1×1** |
| Reg 头 | Conv×2 | Conv×2（不变） |
| legacy | True | **False** |
| TAL / DFL / Loss | ✓ | ✓（相同） |
| end2end | 否 | 否（YOLO26 才有） |

---

## 9. YOLO11 vs YOLO26（简要）

| 维度 | YOLO11 | YOLO26 |
|------|--------|--------|
| 定位 | 2024 生产主力 | 2025 下一代 |
| 检测范式 | TAL + NMS | **NMS-free end2end** |
| DFL reg_max | 16 | **1**（简化） |
| Pose | Pose + OKS | **Pose26 + RLE** |
| SPPF | 标准 | 变体（k=5,3, True） |
| CPU 速度 | 基线 | **更快**（如 11n 56ms → 26n 39ms CPU ONNX） |
| mAP (n) | 39.5 | **40.9** |

若需 NMS-free 部署或 RLE 姿态，选 YOLO26；若生态成熟、教程多、生产验证充分，YOLO11 仍是稳妥选择。

---

## 10. 精度优化方向

- **数据**：与 v8 相同，清洗标注、增大 imgsz、类别平衡
- **训练**：COCO 预训练 `yolo11*.pt` 微调；`close_mosaic` 最后 10 epoch 关 Mosaic
- **模型选型**：n/s 边缘；m 精度速度平衡；l/x 最高精度
- **结构定制**：改 yaml 中 C3k2 repeat、换 C3k、调整 C2PSA 层数（需重新训练验证）
- **导出**：GPU 生产用 TensorRT FP16；CPU 用 ONNX；Jetson 可 INT8 校准（见 [YOLOv8 §17](./YOLOv8.md)）

---

## 11. 相关论文与参考

| 资源 | 关联 |
|------|------|
| [Ultralytics YOLO11 文档](https://docs.ultralytics.com/models/yolo11/) | 官方 benchmark |
| [YOLOv8 本地笔记](./YOLOv8.md) | TAL / DFL / Loss / Segment / Pose / 部署 |
| [YOLOv5 本地笔记](./YOLOv5.md) | 更早演进对比 |
| PSA / Attention 思想 | 卷积式自注意力于 P5 特征 | §2.3, §13 |
| YOLOE / MobileCLIP | 开放词汇对比学习 | §14 |
| YOLO26 文档 | 后续 NMS-free 版本 | §9 |

---

## 12. C3k2 逐行 Forward 数值推演

以 **YOLO11n Backbone 第 2 层** 为例：输入来自 layer1 Conv 输出 `(1, 128, 160, 160)`，yaml 为 `C3k2 [256, False, 0.25]`。

### 12.1 通道缩放（parse_model）

```
yaml args c2=256, width=0.25, max_channels=1024
实际 c2 = make_divisible(min(256, 1024) × 0.25, 8) = 64

yaml repeats=2, depth=0.50
实际 n = max(round(2 × 0.50), 1) = 1    → 1 个 Bottleneck

e=0.25 → hidden c = int(64 × 0.25) = 16
```

**实例化：** `C3k2(c1=128, c2=64, n=1, c3k=False, e=0.25)`

### 12.2 Forward 逐步（H=W=160）

```
Step 0  输入 x                    (1, 128, 160, 160)

Step 1  cv1 = Conv 1×1            (1, 128, 160, 160) → (1, 32, 160, 160)
        输出通道 = 2×c = 2×16 = 32

Step 2  chunk(2) 分流
        part1 = a                 (1, 16, 160, 160)   ← 直连旁路
        part2 = b                 (1, 16, 160, 160)   ← 进入 Bottleneck 链

Step 3  Bottleneck(part2)         (1, 16, 160, 160) → (1, 16, 160, 160)
        Conv 1×1 (16→16) + Conv 3×3 (16→16) + 残差

Step 4  concat [part1, part2, Bottleneck_out]
        y = [a, b, b']            (1, 48, 160, 160)
        通道数 = (2+n)×c = (2+1)×16 = 48

Step 5  cv2 = Conv 1×1            (1, 48, 160, 160) → (1, 64, 160, 160)
```

### 12.3 与同级 C2f 对比（同输入 128→64）

若 v8 使用 `C2f(128, 64, n=1, e=0.5)`：

| 步骤 | C2f (e=0.5) | C3k2 (e=0.25) |
|------|-------------|---------------|
| hidden c | 32 | **16** |
| cv1 输出通道 | 64 | **32** |
| concat 通道 | (2+1)×32 = **96** | (2+1)×16 = **48** |
| cv2 输入 | 96 → 64 | 48 → 64 |

**C3k2 省在哪里：** hidden 通道减半（e=0.25），concat 后 cv2 的 1×1 卷积输入通道从 96 降到 48，所有中间 3×3 也在 16 通道上运算。

### 12.4 实测 MACs / 参数量（thop，160×160）

| 模块 | MACs | 参数量 | 相对 C2f |
|------|------|--------|----------|
| C2f(128→64, n=1, e=0.5) | 858.5M | 33.2K | 100% |
| C3k2(128→64, n=1, e=0.25) | **254.8M** | **9.7K** | **29.7% MACs** |

同一输入分辨率下，单个 C3k2 计算量约为 C2f 的 **30%**，参数量约为 **29%**。

### 12.5 m/l/x 的 C3k 分支

当 `scale in "mlx"` 时 parse_model 自动 `c3k=True`，C3k2 内部 Block 从 **Bottleneck(1,3)** 换为 **C3k(n=2)**。

```
参数量: 10c² → 11c²  (+10%，见 §2.2.3)
MACs:   ~20c²HW → ~22c²HW
收益:   感受野 3×3 → 5×5 等效 + C3 双分支
```

大模型用略重的 Block 换语义；n/s 小模型保持 Bottleneck 省参数。

---

## 13. 模块级 FLOPs / 参数量对比（v8 vs v11）

以下在本地 ultralytics 包上实测（640 输入，yolo11n / yolov8n 配置）。

### 13.1 全模型

| 模型 | 参数量 | 官方 FLOPs | 说明 |
|------|--------|-----------|------|
| YOLOv8n | **3.157M** | 8.9G | C2f + legacy cv3 |
| YOLO11n | **2.624M** | 6.6G | C3k2 + C2PSA + DWConv cv3 |
| 差值 | **-17%** | **-26%** | 参数与计算双降 |

### 13.2 Detect 分类头 cv3（3 尺度合计，nc=80）

以 yolo11n 检测头通道 `(64, 128, 256)` 为例（P3/P4/P5）：

| cv3 结构 | MACs | 参数量 | 相对 legacy |
|----------|------|--------|-------------|
| legacy=True（v8 风格 Conv×2） | 1.059G | 515.8K | 100% |
| legacy=False（v11 DWConv） | **0.188G** | **83.0K** | **17.8% MACs，16.1% 参数** |

**结论：** v11 的分类头计算量约为 v8 的 **1/5**，这是 YOLO11 整体 FLOPs 下降的主要来源之一。

**DWConv cv3 单层（P3, 64ch, 80×80）结构：**

```
DWConv(64→64, 3×3, groups=64)  →  Conv(64→64, 1×1)
  →  DWConv(64→64, 3×3)       →  Conv(64→64, 1×1)
  →  Conv2d(64→80, 1×1)
```

Reg 分支 cv2 未轻量化，因定位任务对空间建模要求更高。

### 13.3 C2PSA 开销（P5, 256ch, 20×20, n=2）

| 指标 | 数值 |
|------|------|
| MACs | **148.8M** |
| 参数量 | 367.4K |
| 占 yolo11n 总 FLOPs | ~148.8M / 6600M ≈ **2.3%** |
| 占 yolo11n 总参数 | 367K / 2624K ≈ **14.0%** |

C2PSA 参数量占比高于 FLOPs 占比（Attention 在 20×20 上 QK^T 为 O(N²)，N=400 可控）。

**Attention 手工估算（单层 PSABlock，C=128, H=W=20）：**

```
N = H×W = 400
num_heads = 128//64 = 2,  head_dim = 64,  key_dim = 32

QKV Conv1×1:     2 × 128 × (2×2×32 + 64) × 400 ≈ 13.1M
Q @ K:           2 × 2 × 400 × 400 × 32 ≈ 20.5M   ← O(N²) 主项
Attn @ V:        2 × 2 × 400 × 400 × 64 ≈ 41.0M
PE + proj:       约 5~10M
FFN 1×1×2:       2 × 128 × 256 × 400 × 2 ≈ 52.4M

单层 PSABlock 合计 ≈ 130M MACs（与 thop 148M 同量级）
C2PSA n=2 → 约 260M，加上 cv1/cv2 split ≈ 149M（实测）
```

### 13.4 参数量「预算」分配（yolo11n 2.62M）

| 模块类别 | 大致占比 | 相对 v8 |
|----------|----------|---------|
| C3k2（backbone+neck） | ~55% | 单块更轻，层数结构不同 |
| C2PSA | ~14% | **v8 无此项** |
| Detect cv2+cv3 | ~25% | cv3 大幅缩减 |
| SPPF + Conv 下采样 | ~6% | 相近 |

### 13.5 汇总：v11 如何做到「少参数、高精度」

```
1. C3k2(e=0.25)     →  backbone/neck 单块 MACs ≈ C2f 的 30%
2. 更少 repeat      →  yaml ×2 + depth0.5 → 实际 n=1（v8n C2f×3~×6）
3. DWConv cv3       →  分类头 MACs ≈ legacy 的 18%
4. C2PSA            →  +14% 参数，+2% FLOPs，换 P5 语义增强
────────────────────────────────────────────────────────
净效果：总参数 -17%，FLOPs -26%，mAP +2.2（n 档）
```

---

## 14. YOLOE-11 开放词汇检测

YOLOE 是 Ultralytics 的**开放词汇**检测/分割方案，YOLOE-11 复用 YOLO11 的 **C3k2 + C2PSA** backbone，将 Detect 替换为 **YOLOEDetect**（文本/视觉 prompt 对比学习头）。

> 官方 yaml：`cfg/models/11/yoloe-11.yaml`  
> API：`from ultralytics import YOLOE`

### 14.1 与 YOLO11 / YOLO-World 的关系

| 模型 | Backbone | Head | 文本编码 | 特点 |
|------|----------|------|----------|------|
| YOLO11 | C3k2+C2PSA | Detect | 无 | 闭集 80 类 |
| YOLOv8-World | C2f | WorldDetect | CLIP | 早期开放词汇 |
| **YOLOE-11** | C3k2+C2PSA | **YOLOEDetect** | **MobileCLIP** | 文本+视觉双 prompt、可 prompt-free |

### 14.2 YOLOEDetect 结构

```
P3/P4/P5 特征
    │
    ├─ cv2（继承 Detect）→ bbox DFL raw
    │
    ├─ cv3（同 v11 DWConv 风格）→ 视觉嵌入 (B, embed, H, W)   embed=512
    │
    └─ cv4 = BNContrastiveHead(embed)
           │
           scores = einsum("bchw,bkc->bkhw", norm(cv3(x)), text_pe)
                  = 视觉特征 × 文本嵌入 相似度图
```

**对比 v8 Detect cls 分支：**

| | Detect | YOLOEDetect |
|---|--------|-------------|
| cv3 输出 | nc=80 logits | **embed=512** 视觉特征 |
| 分类方式 | 独立 sigmoid | **与文本嵌入对比**（Contrastive） |
| 类别数 | 训练固定 nc | 推理时由 **text_pe 行数** 决定 |
| 额外模块 | 无 | cv4 + reprta + savpe |

**BNContrastiveHead 核心：**

```python
x = BatchNorm2d(x)                    # 图像嵌入归一化
w = normalize(text_pe, dim=-1)        # 文本嵌入 L2 归一化
scores = einsum("bchw,bkc->bkhw", x, w) * exp(logit_scale) + bias
# 输出 (B, num_text_classes, H, W)
```

### 14.3 文本编码与 set_classes

```python
from ultralytics import YOLOE

model = YOLOE("yoloe-11s.pt")

# 方式一：运行时指定任意类别（MobileCLIP 编码）
model.set_classes(["person", "dog", "backpack"])
results = model("bus.jpg")

# 方式二：手动传入 embedding
emb = model.get_text_pe(["cat", "remote"])
model.set_classes(["cat", "remote"], emb)
```

**内部流程（`get_text_pe`）：**

```
类名 list
  → MobileCLIP tokenize + encode_text   # text_model 默认 mobileclip:blt
  → reprta = Residual(SwiGLUFFN)        # 文本嵌入精炼
  → L2 normalize
  → text_pe (1, K, 512)               K = 类别数
```

### 14.4 视觉 Prompt（Visual Prompt）

除文本外，YOLOE 支持**视觉 prompt**——在参考图上框选示例，提取视觉嵌入：

```python
visual_prompts = {
    "bboxes": [[x1, y1, x2, y2], ...],   # 参考框
    "cls": ["target_object", ...],
}
results = model.predict("query.jpg", visual_prompts=visual_prompts, refer_image="ref.jpg")
```

**SAVPE（Spatial-Aware Visual Prompt Embedding）** 将参考框区域特征映射到与 text_pe 同维的嵌入空间，适合文本难以描述的目标。

### 14.5 Prompt-Free 模式（fuse + LRPC）

训练完成后可将文本嵌入**融合进卷积权重**，推理时无需 CLIP：

```python
# 内部：head.fuse(txt_feats) 将 text_pe 乘入 cv3 最后一层 Conv
# 得到 prompt-free 模型，等价于固定类别 Detect
model.set_vocab(vocab, names)
```

**LRPCHead（Loc-Reg-Prompt-Cls）** 在 fuse 后做轻量 prompt-free 推理，过滤低分 anchor。

### 14.6 训练

```bash
# 检测
yolo train model=yoloe-11s.yaml data=Objects365.yaml epochs=100

# 分割
yolo segment train model=yoloe-11s-seg.yaml data=coco.yaml epochs=100
```

**YOLOETrainer 要点：**

| 项 | 说明 |
|----|------|
| nc 含义 | 单图最大 text sample 数（非真实类别数），硬编码 max 80 |
| overlap_mask | 强制 False |
| compile | 不支持 `compile=True` |
| 损失 | 在 v8DetectionLoss 基础上增加对比学习项 |

### 14.7 参数量对比

| 模型 | 参数量 | 相对 yolo11n |
|------|--------|--------------|
| yolo11n | 2.62M | 基线 |
| yoloe-11n | **5.01M** | **+91%（+2.38M）** |

增量主要来自：cv3 输出 512 维嵌入（非 80 类）、cv4 对比头、reprta、SAVPE。Backbone 与 yolo11 **完全相同**。

### 14.8 适用场景

| 场景 | 推荐 |
|------|------|
| 固定 80 类 COCO | yolo11n.pt（更快更小） |
| 自定义类别名、零样本 | yoloe-11s.pt + set_classes |
| 罕见/难描述目标 | 视觉 prompt |
| 部署无 CLIP 依赖 | fuse prompt-free |

---

## 15. v8 → v11 权重迁移可行性

### 15.1 结论先行

| 迁移方式 | 可行性 | 说明 |
|----------|--------|------|
| yolov8n.pt → yolo11n.yaml 全量 load | **不可行** | 结构不同，key/shape 大量不匹配 |
| 同任务 yolo11 预训练 → 微调 | **推荐** | 官方 COCO 预训练 |
| 部分层手动拷贝 | **理论可行，收益低** | 仅极少数早期 Conv 同 shape |
| yolo11 → yoloe-11 backbone | **可行** | backbone yaml 相同，head 不同 |

### 15.2 State Dict 对比（实测 yolov8n vs yolo11n）

```
v8  keys: 467
v11 keys: 651
同名 key: 182
shape 完全一致: 137（多为 BN running 统计等同名不同层）
4D 卷积 weight 精确匹配: 10
```

**不匹配原因：**

1. **模块类型不同**：C2f vs C3k2，层索引与命名不对应  
2. **通道数不同**：同 stage 输出通道因 yaml width/repeat 变化  
3. **v11 独有 C2PSA**：v8 无对应 key  
4. **cv3 结构不同**：legacy Conv vs DWConv，权重 shape 完全不同  

### 15.3 若强行 load 会发生什么

```python
from ultralytics.nn.tasks import DetectionModel
v11 = DetectionModel("yolo11n.yaml", ch=3, nc=80)
v11.load("yolov8n.pt")   # strict=False 默认
# → 仅 shape 匹配的层加载（约 early Conv），其余随机初始化
# → mAP 接近随机，不如从头训或 yolo11 预训练
```

Ultralytics `load()` 使用 `intersect_dicts` 按 shape 过滤，**不会报错但 silently 跳过大部分层**。

### 15.4 推荐迁移路径

```
场景 A：COCO 检测微调
  yolo11s.pt（官方预训练）→ 自己的 data.yaml → train

场景 B：从 v8 升级
  无法直迁权重 → 用 yolo11 预训练重新微调（通常 50~100 epoch 即可追平）

场景 C：YOLOE 开放词汇
  yoloe-11s.pt → set_classes / 继续 train on Objects365

场景 D：知识蒸馏（进阶）
  v8 教师 → v11 学生，需自写蒸馏脚本（ultralytics 未内置）
```

### 15.5 结构对齐参考（Backbone 早期 Conv）

唯一较稳定可迁移的是 **最前几层 stride=2 的 Conv**（64、128 通道），但 gains 极小；**不建议**为此做 partial load，直接使用 yolo11 预训练权重更省时。

---

## 附录

### A. 常用命令

```bash
# 检测
yolo detect train data=coco.yaml model=yolo11n.pt epochs=100 imgsz=640
yolo detect val model=yolo11s.pt data=coco.yaml
yolo detect predict model=yolo11m.pt source=images/ conf=0.25 iou=0.7

# 扩展任务
yolo segment train data=coco-seg.yaml model=yolo11n-seg.pt epochs=100
yolo pose train data=coco-pose.yaml model=yolo11n-pose.pt epochs=100
yolo obb train data=dota8.yaml model=yolo11n-obb.pt epochs=100
yolo classify train data=imagenet model=yolo11n-cls.pt epochs=100

# 导出
yolo export model=yolo11s.pt format=onnx simplify=True
yolo export model=yolo11s.pt format=engine device=0 half=True

# YOLOE 开放词汇
yolo train model=yoloe-11s.yaml data=Objects365.yaml epochs=100
python -c "from ultralytics import YOLOE; m=YOLOE('yoloe-11s.pt'); m.set_classes(['person','car']); m('bus.jpg')"
```

### B. 关键源文件

| 路径 | 内容 |
|------|------|
| `cfg/models/11/yolo11.yaml` | 检测模型结构 |
| `cfg/models/11/yolo11-{seg,pose,obb,cls}.yaml` | 扩展任务 yaml |
| `nn/modules/block.py` | **C3k2**, **C3k**, **C2PSA**, **PSABlock**, **Attention**, **BNContrastiveHead**, **SAVPE** |
| `cfg/models/11/yoloe-11.yaml` | YOLOE 开放词汇检测 |
| `cfg/models/11/yoloe-11-seg.yaml` | YOLOE 开放词汇分割 |
| `nn/modules/head.py` | Detect, **YOLOEDetect**, YOLOESegment |
| `models/yolo/yoloe/` | YOLOETrainer, YOLOEDetectValidator |
| `models/yolo/model.py` | **YOLOE** 类（set_classes / get_text_pe） |
| `nn/tasks.py` | parse_model（legacy=False, c3k 自动切换） |
| `utils/loss.py` | v8DetectionLoss（与 v8 共用） |
| `utils/tal.py` | TaskAlignedAssigner |

### C. 文档覆盖清单

**已覆盖：**

- [x] YOLO11 概览与官方精度
- [x] **v8→v11 升级动机与四大升级点详解（§1.5~§1.7）**
- [x] **yaml 逐行对照 + 特征图实测（§3.4~§3.5）**
- [x] **v8/v11 详细差异总表（§8）**
- [x] **C3k vs Bottleneck 参数量/MACs 公式推导（§2.2）**
- [x] C3k2 / C3k 结构及 m/l/x 自动 c3k 规则
- [x] C2PSA / PSABlock / Attention 原理
- [x] Detect legacy=False DWConv 分类头
- [x] Backbone / Neck 逐层与 v8 索引差异
- [x] yaml scales 与 v8 对比
- [x] 扩展任务精度摘要
- [x] v8 / v26 对照
- [x] 训练推理命令
- [x] **C3k2 逐行 forward 数值推演（§12）**
- [x] **模块级 FLOPs / 参数量对比（§13）**
- [x] **YOLOE-11 开放词汇（§14）**
- [x] **v8→v11 权重迁移分析（§15）**

**可继续深入：**

- [ ] C2PSA 与 v8 无注意力版的 ablation 思路
- [ ] YOLOE prompt-free fuse 逐步源码
- [ ] yolo11 自定义 yaml 改 C2PSA repeat 实验模板

### D. 参考资料

- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- [YOLOv8 本地笔记](./YOLOv8.md)
- [YOLOv5 本地笔记](./YOLOv5.md)
