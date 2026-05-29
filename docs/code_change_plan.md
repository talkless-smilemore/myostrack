# NS-OPLoRA 代码改动方案

## 改动概览

| 文件 | 操作 | 说明 |
|------|------|------|
| `lib/models/layers/oplora.py` | **修改** | 在 `OPLoRALinear` 基础上新增 `NeuronSelectiveOPLoRALinear` |
| `lib/models/layers/neuro_oplora.py` | **新建** | 层自适应注入逻辑 + 配置解析 |
| `lib/models/ostrack/ostrack.py` | **修改** | `build_ostrack` 支持 `NEURO_OPLORA` 配置分支 |
| `lib/config/ostrack/config.py` | **修改** | 新增 `TRAIN.NEURO_OPLORA` 默认配置段 |
| `experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_neuro_oplora.yaml` | **新建** | 反无人机层自适应微调实验配置 |

---

## 改动 1: `lib/models/layers/oplora.py`

### 改动内容

在现有 `OPLoRALinear` 类中新增 `NeuronSelectiveOPLoRALinear` 子类。

### 新增代码逻辑

```python
class NeuronSelectiveOPLoRALinear(OPLoRALinear):
    """
    OPLoRA + NeuroAda: 带神经元选择性掩码的正交投影 LoRA。

    在前向传播中对 LoRA 输出施加 per-neuron binary mask，
    只允许幅值最高的 top-p% 神经元接收非零更新。
    """

    def __init__(self, ..., neuron_keep_ratio=1.0):
        # 1. 调用父类 __init__，完成 SVD + Uk/Vk + lora_A/B 初始化
        # 2. 从 self.weight 计算每行(神经元)的幅值均值作为 importance
        # 3. 按 neuron_keep_ratio 选择 top-p% 神经元
        # 4. 构建 self.neuron_mask: [d_out] 的 0/1 buffer
```

### 关键改动点

- `__init__` 新增参数 `neuron_keep_ratio: float = 1.0`
- 新增 `self.register_buffer("neuron_mask", mask)`
- `forward()` 中，在 `u_r` 计算后插入：`u_r = u_r * self.neuron_mask`

### 参数变化

| 参数 | 原 OPLoRALinear | NeuronSelectiveOPLoRALinear |
|------|----------------|---------------------------|
| `in_features` | ✓ | ✓ |
| `out_features` | ✓ | ✓ |
| `bias` | ✓ | ✓ |
| `rank` | ✓ | ✓ |
| `top_k` | ✓ | ✓ |
| `alpha` | ✓ | ✓ |
| `weight` | ✓ | ✓ |
| `bias_tensor` | ✓ | ✓ |
| `neuron_keep_ratio` | — | **新增** |

### from_linear 类方法

新增参数传递 `neuron_keep_ratio`。

---

## 改动 2: `lib/models/layers/neuro_oplora.py` (新建)

### 文件功能

1. 定义默认的反无人机层配置
2. 实现 `inject_neuro_oplora_into_backbone()` 函数

### 默认层配置

```python
# 反无人机默认配置：浅层强适应，深层冻结
ANTI_UAV_DEFAULT_LAYER_CONFIG = [
    # Group A: 浅层 (blocks 0-2) — 关键，强适应
    {"blocks": [0, 1, 2], "rank": 8, "top_k": 16, "alpha": 8.0,
     "neuron_keep_ratio": 0.8,
     "targets": ["qkv", "proj", "fc1", "fc2"]},
    # Group B: CE 剪枝层 (blocks 3, 6, 9) — 关注注意力
    {"blocks": [3, 6, 9], "rank": 4, "top_k": 16, "alpha": 8.0,
     "neuron_keep_ratio": 0.6,
     "targets": ["qkv", "proj"]},
    # Group C: 中层过渡 (blocks 4, 5) — 轻量 MLP
    {"blocks": [4, 5], "rank": 4, "top_k": 16, "alpha": 8.0,
     "neuron_keep_ratio": 0.5,
     "targets": ["fc1", "fc2"]},
    # Group D: 深层 (blocks 7, 8, 10, 11) — 不注入 = 冻结
]
```

### injection 函数签名

```python
def inject_neuro_oplora_into_backbone(
    backbone: nn.Module,
    enable: bool,
    layer_configs: list[dict],
) -> tuple[int, int]:
    """
    按层配置向 backbone 注入 NeuronSelectiveOPLoRALinear。

    Args:
        backbone: VisionTransformer 或 VisionTransformerCE
        enable: 是否启用
        layer_configs: 层配置列表，每项含:
            - blocks: list[int]  要注入的 block 索引
            - rank: int          低秩维度
            - top_k: int         SVD 奇异向量数
            - alpha: float       缩放系数
            - neuron_keep_ratio: float  神经元保留比例
            - targets: list[str] 要替换的 Linear 名称

    Returns:
        (num_replaced, num_skipped): 替换计数
    """
```

### 实现逻辑

```
1. if not enable: return (0, 0)
2. for each layer_cfg in layer_configs:
3.     for block_idx in layer_cfg.blocks:
4.         block = backbone.blocks[block_idx]
5.         for target_name in layer_cfg.targets:
6.             if block has attr target_name and it's nn.Linear:
7.                 replace with NeuronSelectiveOPLoRALinear.from_linear(...)
8.                 replaced_count += 1
9. return (replaced_count, 0)
```

---

## 改动 3: `lib/models/ostrack/ostrack.py`

### 改动位置

`build_ostrack(cfg, training)` 函数中，OPLoRA 注入代码之后。

### 当前代码（约 L45-50）

```python
# 当前: 只有 OPLORA 分支
if cfg.TRAIN.OPLORA.ENABLE:
    inject_oplora_into_backbone(...)
```

### 修改后

```python
# 新增: NEURO_OPLORA 分支（与 OPLORA 互斥）
if getattr(cfg.TRAIN, "NEURO_OPLORA", None) and cfg.TRAIN.NEURO_OPLORA.ENABLE:
    from lib.models.layers.neuro_oplora import inject_neuro_oplora_into_backbone, ANTI_UAV_DEFAULT_LAYER_CONFIG
    layer_configs = cfg.TRAIN.NEURO_OPLORA.get("LAYER_CONFIGS", None)
    if layer_configs is None:
        layer_configs = ANTI_UAV_DEFAULT_LAYER_CONFIG
    n_replaced, _ = inject_neuro_oplora_into_backbone(backbone, True, layer_configs)
    logger.info(f"NS-OPLoRA: replaced {n_replaced} Linear layers")
elif cfg.TRAIN.OPLORA.ENABLE:
    # 原有 OPLORA 逻辑保持不变
    inject_oplora_into_backbone(...)
```

---

## 改动 4: `lib/config/ostrack/config.py`

### 新增配置段

在 `TRAIN` 段末尾，`OPLORA` 之后新增：

```python
# Neuron-Selective OPLoRA (融合 OPLoRA + NeuroAda)
cfg.TRAIN.NEURO_OPLORA = edict()
cfg.TRAIN.NEURO_OPLORA.ENABLE = False
cfg.TRAIN.NEURO_OPLORA.LAYER_CONFIGS = None  # None = 使用默认反无人机配置
```

### 说明

- `ENABLE = False`：默认关闭，不影响现有实验
- `LAYER_CONFIGS = None`：使用代码中的默认配置；用户可以在 YAML 中覆盖

---

## 改动 5: 新增实验配置 YAML

### 文件

`experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_neuro_oplora.yaml`

### 内容

基于现有 `vitb_384_mae_ce_32x4_ep300_uav_oplora.yaml`，将 `TRAIN.OPLORA` 替换为 `TRAIN.NEURO_OPLORA`：

```yaml
TRAIN:
  # 替换原有 OPLORA 配置
  OPLORA:
    ENABLE: false
  NEURO_OPLORA:
    ENABLE: true
    LAYER_CONFIGS: null  # 使用代码默认的反无人机配置
  # 其余配置与 uav_oplora.yaml 相同 ...
```

---

## 改动不涉及的文件

以下文件**不需要修改**：

| 文件 | 原因 |
|------|------|
| `lib/models/ostrack/vit.py` | Block 定义不变 |
| `lib/models/ostrack/vit_ce.py` | CEBlock 构建不变，注入在外部完成 |
| `lib/models/layers/attn.py` | Attention 不变 |
| `lib/models/layers/attn_blocks.py` | CEBlock/Block 不变 |
| `lib/models/layers/head.py` | Box head 不变 |
| `lib/train/train_script.py` | 训练逻辑不变 |
| `tracking/train.py` | 入口不变 |

---

## 测试验证清单

- [ ] 现有 `uav_oplora.yaml` 配置仍能正常运行（向后兼容）
- [ ] 新 `uav_neuro_oplora.yaml` 配置可以正常训练
- [ ] 日志输出确认：浅层 blocks 0-2 注入了 4 个 Linear 模块
- [ ] 日志输出确认：CE 层 blocks 3,6,9 只注入了 qkv, proj
- [ ] 日志输出确认：深层 blocks 7,8,10,11 未注入任何模块
- [ ] 训练后推理：merge 后的模型输出与 merge 前一致
- [ ] 显存对比：NS-OPLoRA 显存低于全层 OPLoRA

---

*代码改动方案 — 2026/05/26*
