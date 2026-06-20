# OSTrack 反无人机追踪改造完整说明

> 本文档系统性地记录基于原始 OSTrack（OS-Track[^1]）所做的全部修改，
> 重点阐述 **UAV-WSP（Weighted Spectral Projection）** 方案的设计原理、
> 文件级改动位置、以及所有融入的反无人机先验知识。

---

## 目录

1. [总体架构概览](#一总体架构概览)
2. [与原始 OSTrack 的差异总表](#二与原始-ostrack-的差异总表)
3. [WSP 核心方案详解](#三wsp-核心方案详解)
4. [反无人机先验知识体系](#四反无人机先验知识体系)
5. [文件级改动清单](#五文件级改动清单)
6. [配置体系](#六配置体系)
7. [数据流与训练流程](#七数据流与训练流程)

---

## 一、总体架构概览

```
原始 OSTrack:    Backbone(ViT) ──→ Box Head ──→ 预测框
                    │
                仅全参数微调，无 PEFT 支持

当前 OSTrack+:   Backbone(ViT) ──→ Box Head ──→ 预测框 + 时间连续性损失
                    │
          ╔════════╧════════╗
          ║  UAV-WSP Adapter ║  ← 核心创新
          ║  28个 WSP 模块   ║
          ╚════════════════╝
                    │
         ┌──────────┼──────────┐
         ↓          ↓          ↓
    加权谱投影   门控机制   聚焦正则化
    P=I-UDUᵀ   细节初始化   Gini 最大化
```

---

## 二、与原始 OSTrack 的差异总表

| 维度 | 原始 OSTrack | 当前版本（UAV-WSP） |
|------|-------------|---------------------|
| 微调方式 | 全参数微调（Full FT） | **参数高效微调（PEFT）**，仅训练约 0.5% 参数 |
| PEFT 方法 | 无 | **加权谱投影 $P = I - UDU^T$** |
| 门控机制 | 无 | **可学习软门控** + 细节显著性初始化 |
| 正则化 | 无 | 熵正则化 + Group-Lasso + Gini 聚焦 |
| 时间建模 | 单帧独立推理 | **时间连续性损失** $L_{temp}$（帧间运动先验） |
| 推理辅助 | 无 | **自适应卡尔曼滤波器**（帧间平滑+搜索区域调节） |
| 训练数据 | GOT-10k, LaSOT, COCO, VID | **AntI-UAV 全系列** (300/410/600) |
| 搜索区域大小 | 320×320 | **256×256**（缩小以聚焦小目标） |
| 搜索因子 | 5.0 | **4.0**（缩小搜索范围减少背景干扰） |
| 最佳模型保存 | 无 | YOLO 风格 `OSTrack_best.pth.tar` |
| 训练精度 | FP32 | **AMP 混合精度** |
| 配置命名 | OPLoRA / SGLoRA | **UAV_WSP**（旧配置标为 deprecated） |

---

## 三、WSP 核心方案详解

### 3.1 设计演进路线

```
OPLoRA ──→ NS-OPLoRA ──→ SGLoRA ──→ UAV-WSP V1 ──→ UAV-WSP V2（当前）
 硬投影      硬投影+       硬投影+       加权投影+      V1 + 参数放松 +
 P=I-UUᵀ    静态掩码      软门控        Gini聚焦      时间连续性损失
```

### 3.2 加权谱投影（Weighted Spectral Projection）

**代码位置**：`lib/models/layers/uav_wsp.py` 类 `WeightedSpectralProjection`

这是与原始 OSTrack 最根本的改动。原始 OSTrack 无任何 PEFT 组件，而我们引入了一个全新的适配器模块，其核心是**将硬正交投影替换为按奇异值比值渐变保护的加权投影**。

#### 数学形式

| 组件 | 原始 OPLoRA/SGLoRA | UAV-WSP |
|------|-------------------|---------|
| 输入投影 | $P_R = I - V_k V_k^T$ | $P_R(w) = I - V_k D V_k^T$ |
| 输出投影 | $P_L = I - U_k U_k^T$ | $P_L(w) = I - U_k D U_k^T$ |
| 谱权重 | $d_i \in \{0, 1\}$（一刀切） | $d_i = (\sigma_i / \sigma_1)^\beta \in (0, 1]$（渐变保护） |

其中 $D = \text{diag}(d_1, \ldots, d_k)$，$d_i = (\sigma_i / \sigma_1)^\beta$。

#### 前向计算流程（对应代码 `forward()` 第 400~433 行）

```
输入 x
  │
  ├── Frozen path: out = x @ W₀ᵀ + b              ← 原始权重不变
  │
  └── Adapter path:
        x_r = P_R(w) · x                           ← 加权输入投影
        z   = x_r @ Aᵀ                             ← 低秩编码 (r-dim)
        u   = z @ Bᵀ                               ← 目标解码 (d_out-dim)
        u_r = P_L(w) · u                           ← 加权输出投影
        u_g = σ(s) · u_r                           ← 目标显著性门控
        Δy  = (α/r) · u_g                          ← 缩放残差
  │
  └── out + Δy                                     ← 合并输出
```

#### 为何加权投影适用于反无人机追踪

反无人机追踪中的目标通常很小（< 30px），其信息分布在大幅面图像中需要精细的细节特征。在 SVD 分解中：

- **大奇异值 $\sigma_1$**：对应粗粒度全局结构（背景场景、大物体模式）
- **中奇异值 $\sigma_{k/2}$**：携带边缘、纹理、高频边界响应——**小目标的关键信息**
- 原始硬投影 $P=I-UU^T$ **不加区分地移除所有 top-k 方向**，导致中奇异值方向的细节适应空间被堵塞
- 加权投影 $P=I-UDU^T$ 保留了中奇异值方向的大部分适应空间，让小目标特征可以通过 LoRA 学习

### 3.3 细节显著性门控初始化（Detail-Saliency Gate Init）

**代码位置**：`uav_wsp.py` 函数 `_detail_saliency_from_svd()`（第 99~125 行）

```python
detail_i = Σ_j (Uk[i,j]² · (1 - d_j))    # 输出通道i的"细节重要性"
gate_logit_init[i] = 0.6 · norm(detail) - 0.3
```

- 与小奇异值（细节）方向耦合强的通道 → 初始门控偏高 (~0.55)
- 与大奇异值（背景）方向耦合强的通道 → 初始门控偏低 (~0.45)
- 这给目标细节相关通道一个"头马效应"，加速小目标特征的收敛

### 3.4 正则化损失体系

**代码位置**：`uav_wsp.py` 方法 `regularisation_loss()`（第 471~505 行）

$$L_{reg} = \lambda_e(t) \cdot H(\sigma(s)) + \lambda_g(t) \cdot \sum_i \|B_{i,:}\|_2 - \lambda_f(t) \cdot G(\sigma(s))$$

| 项 | 含义 | 反无人机动机 |
|----|------|-------------|
| $\lambda_e H(g)$ | **熵正则化**：推动门控值趋向 0 或 1 | 产生稀疏的通道选择性，仅少数特征通道响应目标 |
| $\lambda_g \sum\|B_i\|_2$ | **Group-Lasso**：B 矩阵行级稀疏 | 神经元级压缩，减少冗余通道 |
| $-\lambda_f G(g)$ | **Gini 聚焦**：最大化门控值的集中度 | 小目标在特征空间只占据少数判别通道，高 Gini = 更聚焦 |

**正则化调度**（代码 `_schedule_lam()` 第 456~469 行）：

```
Epoch 0~20:  λ=0           (自由学习，无正则化压力)
Epoch 20~30: λ 线性爬坡     (逐步稀疏化)
Epoch 30~40: λ=max         (硬二值门控，压缩就绪)
```

### 3.5 V2 参数放松

与 V1 相比，V2 显著放松了约束，获得更好的跟踪性能：

| 参数 | V1 峰值 | V2 峰值 | 变化 |
|------|---------|---------|------|
| `spectral_beta` | 1.0（大部分层） | **0.5** | ↓ 更自由 |
| `ENTROPY_LAM_MAX` | 5e-5 | **1e-5** | ↓ 80% |
| `GROUP_LASSO_LAM_MAX` | 1e-5 | **5e-6** | ↓ 50% |
| `FOCUS_LAM_MAX` | 1e-5 | **0** | 关闭 |
| B4/B10 | 冻结 | **rank=2 轻量解冻** | 新增 |
| 核心层 rank | 4~8 | **8~12** | ↑ |

---

## 四、反无人机先验知识体系

本系统在**数据层、模型层、损失层、推理层**四个层面引入了反无人机追踪的领域先验知识。

### 4.1 数据层先验

#### ① 专用反无人机数据集

| 数据集 | 规模 | 特点 |
|--------|------|------|
| Anti-UAV300 | ~300 段红外视频 | 基础反无人机基准 |
| Anti-UAV410 | ~410 段红外视频 | 扩展版本，含更多场景 |
| Anti-UAV600 | ~600 段红外视频 | 最新扩展版本 |

- **代码位置**：`lib/train/base_functions.py` 第 86~109 行
- **数据集类**：`lib/train/dataset/anti_uav_json.py` — JSON + 红外图像加载器
- **YAML 配置**：三个数据集以 1:1:1 比例混合训练

#### ② 缩小搜索区域

```yaml
SEARCH:
  SIZE: 256         # 原 OSTrack 为 320
  FACTOR: 4.0       # 原 OSTrack 为 5.0
```

- **动机**：无人机目标小，过大的搜索区域引入过多背景噪声，且下采样后目标特征更弱
- **效果**：搜索区域缩小 36%，目标在特征图中的像素占比提升

#### ③ 因果采样模式

**代码位置**：`lib/train/data/sampler.py`

```python
frame_sample_mode='causal'  # 搜索帧严格在模板帧之后
```

确保时间顺序一致性，模拟真实跟踪场景。

### 4.2 模型层先验

#### ④ 加权谱投影（核心）

- **$\beta$ 值的物理意义**：控制投影对 top-k 奇异方向的保护强度
- **$\beta=0.5$（默认）**：$\sigma_k$ 方向保留约 70% 的原始 LoRA 适应自由度
- **$\beta=0.3$（遮挡层）**：$\sigma_k$ 方向保留约 85% 的自由度
- **代码**：`uav_wsp.py::WeightedSpectralProjection.__init__()` 第 318~332 行

#### ⑤ 层选择策略（基于扰动实验）

**代码位置**：`uav_wsp.py` 第 143~201 行 `WSP_DEFAULT_PRIOR_CONFIG`

通过 PatchShuffle + GaussBlur + PhaseScramble 三重扰动实验测量各层对反无人机追踪的敏感性，选择注入适配器的层：

| Block | 敏感性 | 是否注入 | 理由 |
|-------|--------|---------|------|
| B0 | 0.729 | ❌冻结 | patch embedding，纯投影无学习能力 |
| **B1** | **1.224** | ✅ y | **纹理点火层**，4.6× 敏感性跳变 |
| **B2** | 1.061 | ✅ y | 全局空间构建 |
| **B3** | 1.170 | ✅ y | CE-1 入口，空间筛选 |
| **B4** | 1.089 | ✅ y(V2 新增) | 空间细化 |
| **B5** | **1.380** | ✅ y | CE 后局部重建 |
| **B6** | **1.559** | ✅ y | CE-2 综合决策，**全网络最高** |
| **B7** | **1.491** | ✅ y | 融合枢纽，纹理+结构双轨 |
| **B8** | 0.843 | ✅ y | 纯局部纹理，遮挡恢复 |
| **B9** | **1.479** | ✅ y | CE-3 纹理筛选 |
| **B10** | 1.246 | ✅ y(V2 新增) | 精细调优 |
| B11 | 1.025 | ❌冻结 | 稳定输出 |

#### ⑥ 注意力层 vs MLP 层差异化适配

基于 CE（Candidate Elimination）流程中各层角色的理解：

- **注意力层（qkv, proj）**：处理空间关系，对 CE 决策层（B3, B6, B9）更关键
- **MLP 层（fc1, fc2）**：处理特征变换，对重建层（B5, B2）更关键
- 不同层组使用不同的 rank 和 β 配置，**代码**：YAML 中的 `targets` 字段

### 4.3 损失层先验

#### ⑦ 时间连续性损失（V2 核心新增）

**动机**：无人机受物理惯性约束，连续两帧间目标位置不能瞬移。

**数学形式**：
$$L_{temp} = \text{smooth}_{L1}(\text{predBox}_{search}, \text{predBox}_{next})$$

**实现流程**：

1. **采样**（`lib/train/data/sampler.py` 第 155~165 行）：采样搜索帧 t 的同时，也采样帧 t+1
2. **裁剪**（`lib/train/data/processing.py` 第 145~170 行）：t+1 帧**复用搜索帧的 jittered crop 窗口**，保证仅像素内容不同
3. **前向**（`lib/train/actors/ostrack.py` 第 79~89 行）：用相同模板分别推理搜索帧和下一帧
4. **损失**（`lib/train/actors/ostrack.py` 第 122~131 行）：smooth L1 约束两帧预测的一致性

**关键设计决策**：
- 两帧使用**完全相同的 crop 窗口**（中心位置、缩放一致）→ 像素差异纯粹来自目标运动
- 此先验**仅对反无人机数据有效**：通用跟踪数据（LaSOT、GOT-10k）的采样间隔可能很大，不满足连续运动假设

#### ⑧ 聚焦正则化（Gini 最大化）

```python
L_reg += -λ_f · Gini(σ(s))  # 最大化门控 Gini 系数
```

**反无人机动机**：小目标在特征空间中只占据少数判别通道。高 Gini 意味着模型学习到"少数通道高度响应、多数通道保持沉默"的稀疏显著性模式——这与小目标占据图像极少像素这一物理事实一致。

**代码**：`uav_wsp.py` 第 439~454 行（Gini 计算）、第 498~503 行（损失项）

### 4.4 推理层先验

#### ⑨ 自适应卡尔曼滤波器

**代码位置**：推理阶段（`tracking/test.py` 调用链中）
**配置**：`lib/config/ostrack/config.py` 第 148~163 行

卡尔曼滤波器在推理阶段提供帧间平滑和异常检测：

| 参数 | 值 | 反无人机意义 |
|------|----|-------------|
| `BASE_PROCESS_NOISE` | 0.01 | 运动模型不确定性 |
| `VELOCITY_NOISE_SCALE` | 10.0 | 无人机速度变化容忍度 |
| `CONF_THRESHOLD` | 0.3 | 低置信度判定门槛 |
| `MIN_SEARCH_FACTOR` | 2.5 | 低置信度时缩小搜索范围 |
| `MAX_SEARCH_FACTOR` | 5.0 | 高不确定性时扩大搜索 |
| `ADAPT_GAIN` | 2.0 | 噪声自适应的增益 |
| `MAX_LOW_CONF_FRAMES` | 5 | 连续低置信帧阈值 |

#### ⑩ TF32 加速 + 高精度 matmul

**代码位置**：`tracking/test.py` 第 28~32 行

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
```

利用 Tensor Core 加速矩阵运算的同时保持高精度，对微小目标的精确回归尤为重要。

---

## 五、文件级改动清单

### 5.1 新增文件

| 文件 | 说明 | 关键内容 |
|------|------|----------|
| `lib/models/layers/uav_wsp.py` | **WSP 核心模块（~570 行）** | `WeightedSpectralProjection` 类、`inject_wsp_into_backbone` 注入函数、`collect_wsp_regularisation` 正则收集 |
| `experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp.yaml` | V2 实验配置（AntI-UAV 全系列） | 5组层配置、时间连续性损失、40 epoch |
| `experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav300_wsp.yaml` | Anti-UAV300 专用配置 | 同上，仅含 300 系列数据 |
| `experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav410_wsp.yaml` | Anti-UAV410 专用配置 | 同上，仅含 410 系列数据 |

### 5.2 修改文件

| 文件 | 改动类型 | 详细说明 |
|------|---------|----------|
| `lib/models/ostrack/ostrack.py` | **模型构建** | `build_ostrack()`: 移除旧 OPLoRA/NeuroOPLoRA/SGLoRA 引用，新增 WSP 注入（第 157~178 行） |
| `lib/train/actors/ostrack.py` | **训练前向** | 替换 import 为 WSP（第 8 行）；新增 WSP 进度追踪 `set_progress()`（第 68~71 行）；新增时间连续性损失（第 122~131 行）；新增 `Loss/wsp_reg` 和 `Loss/temporal` 日志（第 143~144 行） |
| `lib/config/ostrack/config.py` | **配置定义** | 新增 `UAV_WSP` 完整配置段（第 76~85 行）；新增 `TEMPORAL_SMOOTHNESS`（第 93~96 行）；新增 `CENTER_PRIOR`（第 38~40 行）；新增 `SAVE_BEST`（第 86~87 行）；旧 OPLoRA/SGLoRA 标为 deprecated（第 55~75 行） |
| `lib/train/data/sampler.py` | **数据采样** | `TrackingSampler.__init__()` 新增 `enable_temporal_smoothness` 参数（第 25 行）；`getitem()` 中新增 t+1 帧采样（第 154~165 行） |
| `lib/train/data/processing.py` | **数据增强** | `STARKProcessing.__call__()` 新增 `search_next_images` 处理，复用搜索帧的 crop 窗口（第 145~170 行） |
| `lib/train/base_functions.py` | **数据加载** | `build_dataloaders()` 传递 `enable_temporal_smoothness` 标志到 sampler（第 152~160 行） |
| `lib/train/trainers/base_trainer.py` | **训练管理** | 新增 `_try_save_best()` YOLO 风格最佳模型保存（第 70~79 行，第 129~157 行）；`save_checkpoint()` 支持 `tag="best"`（第 161~213 行） |
| `lib/train/trainers/ltr_trainer.py` | **训练循环** | `train_epoch()` 中 stats 重置前调用 `_try_save_best()`（第 135~136 行） |
| `tracking/test.py` | **推理入口** | 新增 TF32 Tensor Core 加速配置（第 28~32 行） |
| `tracking/analysis_results.py` | **结果分析** | display_name 更新为 WSP 跟踪器 |
| `lib/test/parameter/ostrack.py` | **参数加载** | 支持 `TEST.CHECKPOINT` 配置为 `"best"` 或路径（第 28~37 行） |

### 5.3 未修改文件（保留向后兼容）

| 文件 | 保留原因 |
|------|----------|
| `lib/models/layers/oplora.py` | 旧实验 checkpoint 加载需用 |
| `lib/models/layers/neuro_oplora.py` | 同上 |
| `lib/models/layers/sglora.py` | 同上 |

### 5.4 删除文件

| 文件 | 说明 |
|------|------|
| `experiments/ostrack/*_uyv_finetune*.yaml` | 已废弃的微调配置 |
| `experiments/ostrack/*_sglora.yaml` | 已废弃的 SGLoRA 配置 |
| `experiments/ostrack/*_oplora.yaml` | 已废弃的 OPLoRA 配置 |
| `docs/adaptive_kalman_tracker.md` | 卡尔曼滤波文档已整合 |

---

## 六、配置体系

### 6.1 完整的 YAML 配置结构

```yaml
# ===== 数据层 =====
DATA:
  SAMPLER_MODE: causal          # 因果采样（时间有序）
  MAX_SAMPLE_INTERVAL: 200      # 最大帧间隔
  SEARCH:
    SIZE: 256                   # 搜索区域 256×256（原 320）
    FACTOR: 4.0                 # 搜索因子 4.0（原 5.0）
  TRAIN:
    DATASETS_NAME: [ANTI_UAV_TRAIN, ANTI_UAV410_TRAIN, ANTI_UAV600_TRAIN]
                                # 三个反无人机数据集

# ===== 模型层 =====
MODEL:
  BACKBONE:
    TYPE: vit_base_patch16_224_ce   # 含 CE 的 ViT
    CE_LOC: [3, 6, 9]              # 三层 CE 剪枝
    CE_KEEP_RATIO: [0.7, 0.7, 0.7] # 每层保留 70% token
    CE_TEMPLATE_RANGE: 'CTR_POINT'  # CE 模板范围：中心点
  HEAD:
    TYPE: CENTER                    # Center-based 预测头

# ===== UAV-WSP PEFT 层 =====
TRAIN:
  UAV_WSP:
    ENABLE: true
    LAYER_CONFIGS:
      - blocks: [7, 1]
        rank: 12; top_k: 16; alpha: 8.0; spectral_beta: 0.5
        targets: [attn.qkv, attn.proj, mlp.fc1, mlp.fc2]
      - blocks: [6, 9, 3]
        rank: 12; top_k: 16; alpha: 8.0; spectral_beta: 0.5
        targets: [attn.qkv, attn.proj]
      - blocks: [5, 2]
        rank: 8; top_k: 16; alpha: 8.0; spectral_beta: 0.5
        targets: [mlp.fc1, mlp.fc2]
      - blocks: [8]
        rank: 2; top_k: 16; alpha: 8.0; spectral_beta: 0.3
        targets: [attn.qkv, attn.proj]
      - blocks: [4, 10]
        rank: 2; top_k: 16; alpha: 8.0; spectral_beta: 0.3
        targets: [attn.qkv, mlp.fc1]
    ENTROPY_LAM_MAX: 1e-5
    WARMUP_RATIO: 0.5
    ANNEAL_RATIO: 0.25
    GROUP_LASSO_LAM_MAX: 5e-6
    FOCUS_LAM_MAX: 0.0

  # ===== 时间连续性损失 =====
  TEMPORAL_SMOOTHNESS:
    ENABLE: true
    LOSS_WEIGHT: 0.05

  # ===== 训练策略 =====
  EPOCH: 40
  SAVE_BEST: true
  SAVE_BEST_METRIC: "Loss/total"
  AMP: true

# ===== 推理层 =====
TEST:
  KALMAN_FILTER:
    ENABLE: false                 # 可选启用
    # ... (自适应卡尔曼参数)
```

### 6.2 配置参数速查表

#### UAV-WSP 适配器参数

| 参数 | 默认值 | 作用域 | 反无人机意义 |
|------|--------|--------|-------------|
| `rank` | 8~12 | per-group | 低秩瓶颈维度，越大适应能力越强 |
| `top_k` | 16 | per-group | 保护的奇异方向数 |
| `alpha` | 8.0 | per-group | 缩放因子，控制适配器输出幅度 |
| `spectral_beta` | 0.5 | per-group | **谱衰减指数**，控制小目标细节保护程度 |
| `targets` | - | per-group | 目标层列表（qkv/proj/fc1/fc2） |

#### 正则化参数

| 参数 | V2 默认 | 反无人机动机 |
|------|---------|-------------|
| `ENTROPY_LAM_MAX` | 1e-5 | 温和推动门控二值化，产生稀疏通道选择 |
| `GROUP_LASSO_LAM_MAX` | 5e-6 | 适度神经元级压缩 |
| `FOCUS_LAM_MAX` | 0.0 | V2 关闭（V1 验证中发现过于激进） |
| `WARMUP_RATIO` / `ANNEAL_RATIO` | 0.5 / 0.25 | 前 20 epoch 自由学习，后 10 epoch 逐步稀疏 |

#### 时间连续性损失参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `ENABLE` | true | 是否启用 |
| `LOSS_WEIGHT` | 0.05 | 损失权重，推荐范围 0.01~0.1 |

#### 卡尔曼滤波器参数

| 参数 | 默认 | 反无人机意义 |
|------|------|-------------|
| `BASE_PROCESS_NOISE` | 0.01 | 运动模型噪声基准 |
| `CONF_THRESHOLD` | 0.3 | 低置信度判定（无人机频繁遮挡） |
| `MAX_LOW_CONF_FRAMES` | 5 | 连续低置信容忍上限 |
| `MIN_SEARCH_FACTOR` | 2.5 | 低置信时缩小搜索范围 |
| `MAX_SEARCH_FACTOR` | 5.0 | 高不确定时扩大搜索 |

---

## 七、数据流与训练流程

### 7.1 训练数据流

```
                      ┌─── Frame t (template) ──┐
                      │   Frame t+δ (search)     │
     Anti-UAV JSON ──→│   Frame t+δ+1 (next)     │──→ STARKProcessing
      (IR 图像)       └──────────────────────────┘       │
                                                         ↓
                      ┌─────────────────────────────────────┐
                      │  jittered_center_crop()              │
                      │  search 和 next 使用相同 crop 窗口   │
                      └─────────────────────────────────────┘
                                                         ↓
                      ┌─────────────────────────────────────┐
                      │  STARKProcessing 输出：              │
                      │  template_images (128×128)          │
                      │  search_images   (256×256)          │
                      │  search_next_images (256×256)       │
                      └─────────────────────────────────────┘
```

### 7.2 训练前向流

```
                      OSTrack Actor
                                                         
  template ──→ Backbone (ViT-CE) ──→ ┐                   
  search    ─→ Backbone (ViT-CE) ──→ ├→ Box Head ──→ pred_box (search)
  next      ─→ Backbone (ViT-CE) ──→ ┘              ─→ pred_box (next)
       │                                                     │
       ↓                                                     ↓
  WSP set_progress()                L_task = GIoU + L1 + Focal
  (动态正则化调度)                     L_temp = smoothL1(pred_search, pred_next)
                                       L_reg = entropy + group-lasso + focus
                                       L_total = L_task + λ_temp·L_temp + L_reg
```

### 7.3 推理流

```
  输入帧 ──→ ViT Backbone ──→ Box Head ──→ 预测框
                │                              │
                ↓                              ↓
           WSP 已 merge              [可选] 卡尔曼滤波器
           到权重中                    帧间平滑 + 搜索区域调节
          (零开销推理)
```

---

## 附录：与原始 OSTrack 代码的逐文件对比

### 原始 OSTrack（GitHub: `botaoye/OSTrack`）

```
OSTrack/
├── lib/
│   ├── models/ostrack/ostrack.py      ← 无 PEFT，仅 backbone + head
│   ├── train/actors/ostrack.py        ← 仅 GIoU+L1+Focal 损失
│   ├── config/ostrack/config.py       ← 无 WSP/OPLoRA 配置
│   └── train/trainers/               ← 无 best-model 保存
├── tracking/test.py                   ← 无 TF32 配置
└── experiments/                        ← GOT-10k/LaSOT/COCO 配置
```

### 当前版本（本分支 `block`）

```
OSTrack/
├── lib/models/layers/uav_wsp.py       ← ADDED: WSP 核心模块
├── lib/models/ostrack/ostrack.py      ← MODIFIED: WSP 注入
├── lib/train/actors/ostrack.py        ← MODIFIED: 时间连续性 + WSP 日志
├── lib/train/data/sampler.py          ← MODIFIED: t+1 帧采样
├── lib/train/data/processing.py       ← MODIFIED: next 帧 crop 共享
├── lib/train/base_functions.py        ← MODIFIED: temporal 标志传递
├── lib/train/trainers/base_trainer.py ← MODIFIED: YOLO best 保存
├── lib/train/trainers/ltr_trainer.py  ← MODIFIED: best 检查时机
├── lib/config/ostrack/config.py       ← MODIFIED: UAV_WSP/TEMPORAL 配置段
├── lib/test/parameter/ostrack.py      ← MODIFIED: CHECKPOINT 灵活加载
├── tracking/test.py                   ← MODIFIED: TF32 加速
├── tracking/analysis_results.py       ← MODIFIED: WSP 跟踪器名
├── lib/train/dataset/anti_uav_json.py ← ADDED (独立功能): 反无人机数据集
├── experiments/*_wsp.yaml             ← ADDED: WSP 实验配置
└── lib/train/dataset/__init__.py      ← MODIFIED (推测): 注册 AntiUAVJson
```

---

[^1]: Ye, Botao, et al. "OSTrack: Joint Feature Learning and Relationship Modeling for Visual Tracking." ECCV 2022.
