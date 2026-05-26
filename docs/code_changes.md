# 代码改动说明 — 反无人机跟踪三重先验增强

本文档说明为实现以下三项改进所做的所有代码变更：

1. **静态空间先验** — 欧几里得距离中心先验（Fan et al. 2026）注入搜索 token
2. **动态空间先验** — 自适应卡尔曼滤波（innovation-based Q 调整）
3. **自适应搜索因子** — 基于估计速度动态扩大/缩小搜索区域

---

## 一、文件变更总览

| 文件 | 操作 | 说明 |
|------|------|------|
| `lib/test/tracker/kalman_filter.py` | **新建** | 自适应卡尔曼滤波器实现 |
| `lib/models/layers/rpe.py` | 修改 | 新增 `generate_center_distance_prior()` |
| `lib/models/ostrack/base_backbone.py` | 修改 | `finetune_track()` 初始化中心偏置；`forward_features()` 注入 |
| `lib/models/ostrack/vit_ce.py` | 修改 | `forward_features()` 注入中心偏置 |
| `lib/test/tracker/ostrack.py` | 修改 | 集成卡尔曼 + 自适应搜索因子 + 丢失回退 |
| `lib/config/ostrack/config.py` | 修改 | 新增 `CENTER_PRIOR` 和 `KALMAN_FILTER` 配置块 |

---

## 二、detailed change log

### 2.1 `lib/test/tracker/kalman_filter.py` — 新建

**类: `AdaptiveKalmanFilter`**

六维恒定速度（CV）卡尔曼滤波器，核心特性：

| 特性 | 说明 |
|------|------|
| 状态向量 | `[cx, cy, w, h, vx, vy]` |
| 观测向量 | `[cx, cy, w, h]` |
| 自适应 Q | 基于 innovation Mahalanobis 距离自动放大/衰减过程噪声 |
| 置信度门控 | `update_with_confidence()` 仅在 score > 阈值时更新 |

**公开 API:**

```python
kf = AdaptiveKalmanFilter(dt=1.0, ...)
kf.init(cx, cy, w, h)           # 第一帧初始化
pred = kf.predict()               # 预测下一帧 [cx,cy,w,h]
filtered = kf.update(z_cx, z_cy, z_w, z_h)  # 用观测更新
vx, vy = kf.get_velocity()       # 获取估计速度
speed = kf.estimated_speed        # 速度大小（像素/帧）
```

**自适应 Q 机制:**

```
Mahalanobis 距离 > ADAPT_THRESHOLD  → Q *= ADAPT_GAIN（最大 MAX_Q_SCALE 倍）
Mahalanobis 距离 ≤ ADAPT_THRESHOLD  → Q *= DECAY_RATE（最小恢复至基准值）
```

### 2.2 `lib/models/layers/rpe.py` — 修改

**新增函数: `generate_center_distance_prior(feat_sz_s, num_bins)`**

```python
def generate_center_distance_prior(feat_sz_s: int, num_bins: int = 16):
    """
    为特征图上每个搜索 token 计算到中心的欧几里得距离，
    并离散化为 num_bins 个 bin 索引。
    
    Args:
        feat_sz_s: 搜索特征图尺寸，如 256/16=16
        num_bins:  离散化 bin 数
    
    Returns:
        LongTensor (feat_sz_s²,) — 每个 token 的 bin 索引 [0, num_bins-1]
    """
```

生成的 bin 索引用于可学习 `nn.Embedding` 查询中心距离嵌入向量。

### 2.3 `lib/models/ostrack/base_backbone.py` — 修改

**改动 A: 在 `finetune_track()` 末尾新增中心偏置初始化**

位置：`self.add_sep_seg` 代码块之后、`self.return_inter` 之前。

```python
# Euclidean distance centre prior (Fan et al. 2026)
cp_cfg = getattr(cfg.MODEL, "CENTER_PRIOR", None)
if cp_cfg is not None and getattr(cp_cfg, "ENABLE", False):
    feat_sz_s = search_size[0] // new_patch_size
    num_bins = getattr(cp_cfg, "NUM_BINS", 16)
    self.register_buffer("center_dist_idx",
        generate_center_distance_prior(feat_sz_s, num_bins), persistent=True)
    self.center_dist_embed = nn.Embedding(num_bins, self.embed_dim)
    trunc_normal_(self.center_dist_embed.weight, std=0.02)
else:
    self.center_dist_embed = None
    self.center_dist_idx = None
```

**改动 B: 在 `forward_features()` 中注入中心嵌入**

位置：`x += self.pos_embed_x` 之后、`if self.add_sep_seg:` 之前。

```python
# inject centre-distance embedding into search tokens
if self.center_dist_embed is not None:
    centre_emb = self.center_dist_embed(self.center_dist_idx.to(x.device))
    centre_emb = centre_emb.unsqueeze(0).expand(B, -1, -1)
    x = x + centre_emb
```

### 2.4 `lib/models/ostrack/vit_ce.py` — 修改

**改动: `forward_features()` 中注入中心嵌入**

与 `base_backbone.py` 完全相同的注入逻辑，添加在同一位置（`x += self.pos_embed_x` 之后）。

### 2.5 `lib/test/tracker/ostrack.py` — 修改

**改动 A: 新增导入**

```python
from lib.test.tracker.kalman_filter import AdaptiveKalmanFilter
```

**改动 B: `__init__()` 中新增卡尔曼初始化**

从 `cfg.TEST.KALMAN_FILTER` 读取配置，创建 `AdaptiveKalmanFilter` 实例。当 `ENABLE=False` 时 `self.kf = None`，行为与原始代码完全一致。

新增属性:
- `self.kf_enable` — 是否启用卡尔曼
- `self.kf` — `AdaptiveKalmanFilter` 实例或 None
- `self.kf_conf_threshold` — 置信度门控阈值
- `self.kf_min_sf / max_sf` — 自适应搜索因子范围
- `self.kf_speed_ratio` — 速度-搜索因子比率
- `self.kf_max_low_conf` — 触发重检测的连续低置信帧数

**改动 C: `initialize()` 中新增卡尔曼初始化**

```python
if self.kf is not None:
    bbox = info['init_bbox']
    cx = bbox[0] + bbox[2] / 2.0
    cy = bbox[1] + bbox[3] / 2.0
    self.kf.init(cx, cy, bbox[2], bbox[3])
```

**改动 D: `track()` 方法重构**

核心流程变更：

```
原始:  self.state → sample_target → 模型前向 → map_box_back → self.state
                                                         
新版:  kf.predict() → 计算自适应 sf → sample_target(kf_pred, sf)
       → 模型前向 → 反算图像坐标                                 
       → [confidence > threshold] ? kf.update() + 滤波输出                                    
         : 跳过更新 + 预测维持                                    
       → [连续低置信 > max_low_conf] ? _kf_recovery_attempt()
```

当 `self.kf is None` 时，`track()` 的行为完全等价于原始代码。

**改动 E: 新增辅助方法**

- `_adaptive_search_factor(est_speed, target_w, target_h)` — 根据速度计算自适应搜索因子
- `_kf_recovery_attempt(image, H, W, prev_score_map)` — 扩大搜索区域重新检测

### 2.6 `lib/config/ostrack/config.py` — 修改

**新增 `cfg.MODEL.CENTER_PRIOR` 配置块**

```python
cfg.MODEL.CENTER_PRIOR = edict()
cfg.MODEL.CENTER_PRIOR.ENABLE = False    # 默认关闭
cfg.MODEL.CENTER_PRIOR.NUM_BINS = 16     # 距离离散化 bin 数
```

**新增 `cfg.TEST.KALMAN_FILTER` 配置块**

```python
cfg.TEST.KALMAN_FILTER = edict()
cfg.TEST.KALMAN_FILTER.ENABLE = False                     # 默认关闭
cfg.TEST.KALMAN_FILTER.DT = 1.0                           # 帧间间隔
cfg.TEST.KALMAN_FILTER.BASE_PROCESS_NOISE = 0.01          # 位置基础过程噪声
cfg.TEST.KALMAN_FILTER.VELOCITY_NOISE_SCALE = 10.0        # 速度噪声倍率
cfg.TEST.KALMAN_FILTER.SIZE_NOISE_SCALE = 2.0             # 尺寸噪声倍率
cfg.TEST.KALMAN_FILTER.BASE_OBS_NOISE = 5.0               # 观测基础噪声
cfg.TEST.KALMAN_FILTER.ADAPT_THRESHOLD = 5.0              # Mahalanobis 阈值
cfg.TEST.KALMAN_FILTER.ADAPT_GAIN = 2.0                   # Q 放大倍率
cfg.TEST.KALMAN_FILTER.DECAY_RATE = 0.95                  # Q 衰减率
cfg.TEST.KALMAN_FILTER.MAX_Q_SCALE = 10.0                 # Q 最大放大倍率
cfg.TEST.KALMAN_FILTER.CONF_THRESHOLD = 0.3               # 置信度门控阈值
cfg.TEST.KALMAN_FILTER.MIN_SEARCH_FACTOR = 2.5            # 自适应 sf 下限
cfg.TEST.KALMAN_FILTER.MAX_SEARCH_FACTOR = 5.0            # 自适应 sf 上限
cfg.TEST.KALMAN_FILTER.SAFETY_MARGIN = 1.5                # 物理所需 sf 的安全倍率
cfg.TEST.KALMAN_FILTER.MAX_LOW_CONF_FRAMES = 5            # 触发重检测的连续低置信帧数
```

---

## 三、配置使用指南

### 3.1 仅启用中心偏置（需重新训练/微调）

在实验 YAML 中添加：

```yaml
MODEL:
  CENTER_PRIOR:
    ENABLE: true
    NUM_BINS: 16
```

中心偏置嵌入新增约 `16 × 768 = 12K` 参数量，可作为 OPLoRA 之外独立训练的参数。

### 3.2 仅启用卡尔曼 + 自适应搜索因子（推理时，无需重训练）

在实验 YAML 中添加：

```yaml
TEST:
  KALMAN_FILTER:
    ENABLE: true
    # 其余参数使用默认值即可
```

### 3.3 同时启用全部三项

```yaml
MODEL:
  CENTER_PRIOR:
    ENABLE: true
    NUM_BINS: 16

TRAIN:
  OPLORA:
    ENABLE: true
    RANK: 8
    TOP_K: 16
    ALPHA: 8.0

TEST:
  KALMAN_FILTER:
    ENABLE: true
```

### 3.4 基于真实数据的参数推荐

> 数据来源：对 anti_uav / anti_uav410 / anti_uav600 共 610 个序列、600,937 帧转移的统计分析。

**统一默认值（所有数据集通用）：**

```yaml
TEST:
  KALMAN_FILTER:
    ENABLE: true
    MIN_SEARCH_FACTOR: 2.5
    MAX_SEARCH_FACTOR: 5.0
    SAFETY_MARGIN: 1.5
    CONF_THRESHOLD: 0.3
    MAX_LOW_CONF_FRAMES: 5
```

**实际覆盖率验证：**

| 搜索因子 | anti_uav | anti_uav410 | anti_uav600 |
|---------|----------|-------------|-------------|
| sf=2.5 | 98.26% | 98.17% | 97.50% |
| **sf=3.6 (基准)** | **98.94%** | **99.14%** | **98.66%** |
| sf=5.0 (自适应上限) | 99.38% | 99.61% | 99.38% |
| sf=6.0 | 99.54% | 99.79% | 99.62% |

**按数据集差异化建议：**

| 参数 | anti_uav | anti_uav410 | anti_uav600 |
|------|----------|-------------|-------------|
| MAX_SEARCH_FACTOR | 5.0 | 5.0 | 6.0 |
| VELOCITY_NOISE_SCALE | 10.0 | 8.0 | 8.0 |
| 原因 | 目标大 (P50=51px), P99位移=106px | 目标中等 (P50=30px) | 目标小 (P50=24px), 极少数异常位移 |
| sf=3.6 覆盖 | 98.94% | 99.14% | 98.66% |

---

## 四、向后兼容性

- `CENTER_PRIOR.ENABLE` 默认 `False` — 不启用时模型结构与原始完全一致
- `KALMAN_FILTER.ENABLE` 默认 `False` — 不启用时跟踪器行为与原始完全一致
- 所有新增代码均通过 `getattr` 读取配置，旧 YAML 文件无需修改即可运行
- 旧 checkpoint 加载不受影响：中心偏置嵌入在 `finetune_track` 中追加，不修改已有参数

---

## 五、参数量变化

| 组件 | 新增参数量 | 说明 |
|------|-----------|------|
| 中心偏置嵌入 | 16 × 768 = 12,288 | 可训练的 `nn.Embedding`，仅当 `CENTER_PRIOR.ENABLE=true` 时存在 |
| 自适应卡尔曼 | 0 | 纯推理算法，无额外模型参数 |
| 自适应搜索因子 | 0 | 推理时计算，无参数 |

中心偏置嵌入仅占总参数的约 0.014%（ViT-Base 约 86M），可忽略不计。

---

## 六、原理简述

### 6.1 中心偏置（静态空间先验）

每个搜索 token 获得一个可学习的嵌入向量，该嵌入由 token 到搜索区域中心的**欧几里得距离**确定。靠近中心的 token 和远离中心的 token 获得不同的嵌入，使 ViT 的注意力机制具备了**隐式的中心偏好**。

注入位置在 `pos_embed` 之后、transformer blocks 之前，因此所有 12 层 attention 都能感知到距离信息。嵌入是**逐 token 可学习**的，模型可以在训练中自行决定多信任这个先验。

### 6.2 自适应卡尔曼（动态空间先验）

使用恒定速度（CV）模型估计目标运动状态，每帧：
1. **预测**：用运动模型外推目标下一帧位置
2. **搜索**：以预测位置为中心裁剪搜索区域（而非"上一帧位置"）
3. **更新**：用模型输出作为观测修正卡尔曼状态

自适应性体现在**过程噪声 Q 的动态调整**：当模型预测与实际观测差距大时（机动），Q 自动放大让滤波器快速响应；平稳时 Q 衰减恢复平滑。

### 6.3 自适应搜索因子

```
搜索因子 = max(3.0, min(6.0, 3.6 + 2.0 × 预测速度 / 目标对角线))

速度越快 → 搜索因子越大 → 更大的搜索区域 → 容忍更大的帧间位移
```

搜索因子在 3.0~6.0 之间自动调整，平衡了跟踪鲁棒性和计算开销。

---

## 七、三个先验的协同关系

```
静态先验（中心偏置）
    ↓ 告诉注意力机制：关注搜索区域中心附近的 token
    ↓ [需要: 目标在搜索中心附近]

动态先验（卡尔曼滤波）
    ↓ 将搜索区域中心移动到目标预测位置
    ↓ [需要: 搜索区域足够大以包含目标]

自适应搜索因子
    ↓ 根据速度扩大搜索区域

三者协同：
  卡尔曼将目标"拉"到搜索中心 → 中心偏置的前提成立
  自适应搜索因子确保快速运动时目标仍在搜索区域内
  中心偏置帮助 CE 优先保留目标附近的 token
```
