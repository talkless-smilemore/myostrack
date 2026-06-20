# uav_wsp.py 文件结构与函数调用关系

> 本文档解释 `lib/models/layers/uav_wsp.py` 的完整结构：每个函数/类做什么、
> 它们之间的调用关系、以及整个数据流是怎么串起来的。

---

## 一、文件总览（按代码顺序）

| # | 名称 | 类型 | 一句话 |
|---|------|------|--------|
| 1 | `compute_spectral_weights()` | 顶层函数 | 计算谱衰减系数 `d_i = (σ_i/σ₁)^β` |
| 2 | `_detail_saliency_from_svd()` | 顶层函数 | 计算门控初始化的"细节显著性"偏置 |
| 3 | `WSP_DEFAULT_PRIOR_CONFIG` | 顶层常量 | 默认的层配置策略（5组，10个block） |
| 4 | `WeightedSpectralProjection` | **类（核心）** | 单个 WSP 适配器模块，替换一个 nn.Linear |
| 5 | `collect_wsp_regularisation()` | 顶层函数 | 遍历模型，收集所有WSP模块的正则损失 |
| 6 | `inject_wsp_into_backbone()` | 顶层函数 | 把 WSP 模块注入 ViT backbone |
| 7 | `_get_linear()` | 顶层函数 | 工具函数：在模块中按名字找 Linear 层 |
| 8 | `_count_frozen()` | 顶层函数 | 工具函数：统计哪些 block 被冻结 |

---

## 二、类的内部结构（WeightedSpectralProjection）

| # | 方法 | 类型 | 一句话 | 在哪调用 |
|---|------|------|--------|---------|
| A | `_global_progress` | 类变量 | 所有WSP共享的训练进度 [0,1] | `set_progress()` |
| B | `set_progress(progress)` | **类方法** | 设置训练进度 → 控制正则化调度 | OSTrackActor |
| C | `__init__(...)` | 构造方法 | SVD + 初始化全部参数 | `from_linear()` |
| D | `from_linear(linear, ...)` | **工厂方法** | 从 nn.Linear 创建 WSP 模块 | `inject_wsp_into_backbone()` |
| E | `forward(x)` | **前向传播** | 加权投影 + 门控 + 残差 | OSTrack backbone |
| F | `_gini_coefficient(g)` | 静态方法 | 计算 Gini 系数 | `regularisation_loss()` |
| G | `_schedule_lam()` | 内部方法 | 正则化权重随 epoch 调度 | `regularisation_loss()` |
| H | `regularisation_loss()` | **正则化** | 熵 + Group-Lasso + Gini | `collect_wsp_regularisation()` |
| I | `merge_to_weight()` | **推理合并** | 将适配器合并到权重中 | 推理脚本 |

---

## 三、完整调用链

### 3.1 模型构建阶段（1 次调用）

```
ostrack.py::build_ostrack()
  │
  ├── 创建 ViT backbone（vit_base_patch16_224_ce）
  ├── 创建 box head
  └── 读取 cfg.TRAIN.UAV_WSP.ENABLE 判断是否启用 WSP
        │
        └── True →
              inject_wsp_into_backbone(backbone, ...)        ← 函数 6
                │
                ├── 遍历 layer_configs 的每个组（5 组）
                │     │
                │     ├── 遍历 groups["blocks"]（如 [7,1]）
                │     │     │
                │     │     └── 遍历 targets（如 ["attn.qkv", ...]）
                │     │           │
                │     │           └── _get_linear(block, "attn.qkv")    ← 函数 7
                │     │                 │
                │     │                 └── 返回 (linear, parent, attr_name)
                │     │                       │
                │     │                       └── WeightedSpectralProjection.from_linear()  ← 类方法 D
                │     │                             │
                │     │                             └── WeightedSpectralProjection.__init__()  ← 构造 C
                │     │                                   │
                │     │                                   ├── 保存超参数（第 1 步）
                │     │                                   ├── 注册冻结 weight/bias（第 2 步）
                │     │                                   ├── rank<=0 保护（第 3 步）
                │     │                                   ├── SVD 分解（第 4 步）
                │     │                                   │     └── torch.linalg.svd(W₀) → U, S, Vh
                │     │                                   ├── compute_spectral_weights()      ← 函数 1
                │     │                                   │     └── d_i = (σ_i/σ₁)^β  → 谱权重 D
                │     │                                   ├── _detail_saliency_from_svd()    ← 函数 2
                │     │                                   │     └── gate_bias = detail_i 的归一化
                │     │                                   ├── 初始化 A（kaiming）/ B（全零）（第 7 步）
                │     │                                   └── 初始化 gate_logit = gate_bias（第 8 步）
                │     │
                │     └── 替换：setattr(parent, attr_name, WSP模块)
                │
                └── _count_frozen(backbone, configs)              ← 函数 8
                      └── 返回冻结 block 数量
```

**注入结果统计（V2 默认配置）**：
- 10/12 个 block 被注入适配器
- 2 个 block 冻结（B0, B11）
- 共 28 个 WSP 模块（每个目标 Linear 层一个）

### 3.2 训练阶段（每个 epoch 调用）

```
ltr_trainer.py::train_epoch()
  │
  └── ltr_trainer.py::cycle_dataset()
        │
        └── 对于每个 batch:
              │
              ├── data['epoch'] = self.epoch     ← 当前 epoch 号
              │
              └── actor(data)
                    │
                    └── ostrack_actor.py::__call__()
                          │
                          └── forward_pass(data)
                                │
                                ├── WeightedSpectralProjection.set_progress(epoch/EPOCH)
                                │     └── 设置类变量 _global_progress              ← 类方法 B
                                │           （所有 WSP 实例共享）
                                │
                                ├── self.net(template, search, ...)
                                │     │
                                │     └── backbone 内部调用每个 WSP 模块的 forward()
                                │           │
                                │           └── WeightedSpectralProjection.forward(x)  ← 实例方法 E
                                │                 │
                                │                 ├── out = x @ W₀ᵀ + b          [冻结路径]
                                │                 ├── xr = x - (x@Vk)·D·Vkᵀ       [加权输入投影]
                                │                 ├── z = xr @ Aᵇ                 [低秩编码]
                                │                 ├── u = z @ Bᵇ                  [目标解码]
                                │                 ├── ur = u - (u@Uk)·D·Ukᵀ       [加权输出投影]
                                │                 ├── g = sigmoid(gate_logit)      [门控计算]
                                │                 ├── ug = ur * g                  [门控应用]
                                │                 └── return out + scale*ug        [残差合并]
                                │
                                └── compute_losses(out_dict, data)
                                      │
                                      ├── task_loss = GIoU + L1 + Focal
                                      │
                                      ├── temporal_loss = smoothL1(pred, pred_next)  [时间连续性]
                                      │
                                      └── reg_loss = collect_wsp_regularisation(model)  ← 函数 5
                                            │
                                            └── 遍历模型所有子模块:
                                                  │
                                                  └── 对于每个 WeightedSpectralProjection:
                                                        │
                                                        ├── WeightedSpectralProjection._schedule_lam()  ← 实例方法 G
                                                        │     └── 根据 _global_progress 返回 (λe, λg, λf)
                                                        │
                                                        └── WeightedSpectralProjection.regularisation_loss()  ← 实例方法 H
                                                              │
                                                              ├── 项1: λe·H(g)    [熵正则化，门控二值化]
                                                              ├── 项2: λg·mean(||B_i||₂)  [Group-Lasso]
                                                              └── 项3: -λf·G(g)  [Gini聚焦，V2关闭]
                                      │
                                      └── loss = task_loss + temporal_loss + reg_loss
```

### 3.3 推理阶段（1 次调用 + N 次前向）

```
推理脚本加载 checkpoint
  │
  └── 模型准备好后，每个 WSP 模块调用 merge_to_weight()    ← 实例方法 I
        │
        ├── BA = B @ A                          [d_out, d_in]
        ├── BA = BA - (BA@Vk)·D·Vkᵀ            [右边 + 加权投影]
        ├── BA = BA - Uk·D·(Ukᵀ@BA)            [左边 + 加权投影]
        ├── delta = (α/r)·sigmoid(g)·BA         [门控缩放]
        ├── W_merged = W₀ + delta               [合并到冻结权重]
        └── 之后 forward() = 普通 Linear 前向     [零开销推理]
```

---

## 四、数据流一图流

```
┌─────────────────────────────────────────────────────────────────┐
│             模型构建（build_ostrack，1 次）                       │
│                                                                 │
│  ViT backbone  ──→  inject_wsp_into_backbone()                 │
│    (12 blocks)        │                                         │
│                       ├── 遍历 5 个配置组                        │
│                       ├── 遍历 block 索引 [7,1] / [6,9,3] / ... │
│                       ├── 遍历 target ["attn.qkv", ...]         │
│                       └── _get_linear → from_linear → __init__  │
│                             │                         │         │
│                             │                    SVD → D → gate_bias
│                             ↓                         ↓         │
│                        nn.Linear 被替换为  A, B, gate_logit     │
│                        WeightedSpectralProjection               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│           训练（每个 epoch × 每个 batch）                         │
│                                                                 │
│  <epoch> ──→ set_progress(epoch/EPOCH)                         │
│                  │                                               │
│                  ↓  (所有 WSP 实例共享)                          │
│  _global_progress [0,1]  ──→  _schedule_lam() → (λe, λg, λf)   │
│                                                                 │
│  <input x> ──→ forward(x)                                      │
│                   │                                              │
│                   ├── out = x @ W₀ᵀ + b      [冻结路径]        │
│                   ├── xr = P_R(w)·x          [输入投影]         │
│                   ├── z, u = xr@Aᵀ, z@Bᵀ     [低秩适配器]      │
│                   ├── ur = P_L(w)·u          [输出投影]         │
│                   ├── ug = sigmoid(g)·ur     [★门控★]          │
│                   └── return out + α/r·ug    [残差]             │
│                                                                 │
│  <out_dict> ──→ compute_losses()                               │
│                   ├── task_loss = GIoU + L1 + Focal             │
│                   ├── temporal_loss = smoothL1(pred, pred_next) │
│                   ├── reg_loss = Σ regularisation_loss()        │
│                   │                 ├── λe·H(gate)              │
│                   │                 ├── λg·mean(||B_i||₂)       │
│                   │                 └── -λf·Gini(gate)          │
│                   └── loss = task + temporal + reg              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│           推理（1 次 merge + N 次前向）                           │
│                                                                 │
│  merge_to_weight():                                             │
│    W_merged = W₀ + (α/r)·sigmoid(g)·(P_L·B·A·P_R)              │
│                                                                 │
│  之后 forward(x) = x @ W_mergedᵀ + b  ← 普通 Linear 速度       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 五、关键参数在哪里定义 / 被谁使用

| 参数 | 定义位置 | 使用位置 | 作用 |
|------|---------|---------|------|
| `rank` | YAML: 每个配置组 | `__init__` → `lora_A/B` 的维度 | 低秩瓶颈大小 |
| `top_k` | YAML: 每个配置组 | `__init__` → SVD 截断数 | 保护多少奇异方向 |
| `alpha` | YAML: 每个配置组 | `forward`, `merge_to_weight` | 适配器输出缩放 |
| `spectral_beta` | YAML: 每个配置组 | `compute_spectral_weights()` | 谱衰减指数 |
| `ENTROPY_LAM_MAX` | YAML: 全局 | `_schedule_lam()` → `regularisation_loss()` | 熵正则化峰值 |
| `WARMUP_RATIO` | YAML: 全局 | `_schedule_lam()` | 正则化延迟比例 |
| `ANNEAL_RATIO` | YAML: 全局 | `_schedule_lam()` | 正则化爬坡比例 |
| `GROUP_LASSO_LAM_MAX` | YAML: 全局 | `_schedule_lam()` → `regularisation_loss()` | Group-Lasso 峰值 |
| `FOCUS_LAM_MAX` | YAML: 全局 | `_schedule_lam()` → `regularisation_loss()` | Gini 聚焦峰值 |
| `targets` | YAML: 每个配置组 | `inject_wsp_into_backbone()` → `_get_linear()` | 注入哪些层 |
| `blocks` | YAML: 每个配置组 | `inject_wsp_into_backbone()` | 注入哪些 block |

---

## 六、快速对照：想改某处该改什么

| 想改什么 | 改哪里 | 备注 |
|---------|--------|------|
| 在哪些 block 加 WSP | YAML 的 `LAYER_CONFIGS` 或 `WSP_DEFAULT_PRIOR_CONFIG` | |
| 增加/减少适配器的 rank | YAML 对应组的 `rank` 值 | rank↑ = 能力↑ = 参数量↑ |
| 调整小目标保护程度 | `spectral_beta` | β↓ = 更多适应自由 |
| 让门控更快二值化 | 增大 `ENTROPY_LAM_MAX` | V2=1e-5 |
| 重新打开 Gini 聚焦 | 设置 `FOCUS_LAM_MAX > 0` | 从 1e-6 开始试 |
| 想换一种初始化门控 | 改 `_detail_saliency_from_svd()` | 第 99~125 行 |
| 想改正则化调度策略 | 改 `_schedule_lam()` | 第 456~469 行 |
| 想改前向逻辑 | 改 `forward()` | 第 400~433 行 |
