# UAV-WSP: Weighted Spectral Projection for Anti-UAV Tracking

## 概述

**UAV-WSP** (Weighted Spectral Projection) 是对原 SGLoRA 的全面改造，旨在将参数高效微调模块重新打造为**专为反无人机追踪设计的小目标适配器**。

## 核心数学创新

### 1. 加权谱投影（Weighted Spectral Projection）

这是最主要的改动，也是与先前方法最根本的区别：

| | 原方法 (OPLoRA / SGLoRA) | 本方法 (UAV-WSP) |
|---|---|---|
| 输入投影 | $P_R = I - V_k V_k^T$ | $P_R(w) = I - V_k D V_k^T$ |
| 输出投影 | $P_L = I - U_k U_k^T$ | $P_L(w) = I - U_k D U_k^T$ |
| 谱权重 | $d_i \in \{0, 1\}$（一刀切） | $d_i = (\sigma_i / \sigma_1)^\beta \in (0, 1]$（渐变保护） |

**物理意义**：

```
原始硬投影:    σ₁ ─→ 完全移除 (d=1)      σₖ ─→ 完全移除 (d=1)
              小目标细节 → 丢失

加权谱投影:    σ₁ ─→ 强烈保护 (d≈1)       σₖ ─→ 适度保护 (d≈0.3)
              小目标细节 → 保留适应空间
```

- 大奇异值方向（背景场景、大物体模式）被强烈保护，不让 LoRA 改变它们
- 中奇异值方向（边缘、纹理、细节响应）保留更多适应空间，让小目标特征可以通过 LoRA 学到
- β 参数控制衰减曲线：β=0 → 全部不保护（退化为原始 LoRA），β=1 → 线性衰减（默认），β>1 → 保守衰减

### 2. 细节显著性门控初始化（Detail-Saliency Gate Init）

原方法所有门控从 sigmoid(0) = 0.5 初始化。本方法引入偏置：

```python
detail_importance[i] = Σⱼ (Uk[i,j]² · (1 - dⱼ))   # 输出通道 i 的"细节重要性"
gate_logit_init[i] = 0.6 · normalize(detail_importance) - 0.3
```

- 输出通道如果与小奇异值方向耦合强 → 初始门控略高（~0.55-0.65）
- 输出通道如果与大奇异值方向耦合强 → 初始门控略低（~0.35-0.45）
- 这给小目标相关的特征通道一个"头马效应"

### 3. 聚焦正则化（Focus Regularization / Gini Maximization）

新增一项正则化损失：

$$L_{focus} = -\lambda_f \cdot G(\sigma(s))$$

其中 $G$ 是门控值的 Gini 系数：

$$G(g) = \frac{\sum_{i=1}^n (2i - n - 1) \cdot g_{(i)}}{n \cdot \sum g_{(i)}}$$

- Gini 系数衡量"集中度"：高 Gini → 少数通道激活高，多数通道接近 0
- **反无人机动机**：小目标在特征空间中也只占据少数判别通道
- 最大化 Gini 推动适配器学习稀疏的目标显著性模式

## 与原 SGLoRA 的全面对比

| 维度 | 原 SGLoRA | UAV-WSP（本方法） |
|---|---|---|
| 投影方式 | 硬正交投影 $I - UU^T$ | **加权谱投影** $I - UDU^T$ |
| Gate 初始化 | 全 0（sigmoid=0.5） | **细节显著性偏置** |
| 正则化 | 熵 + Group-Lasso | 熵 + Group-Lasso + **Gini 聚焦** |
| β 参数 | 无 | 可配置，per-group **spectral_beta** |
| FOCUS_LAM_MAX | 无 | 可配置 |
| 反无人机叙事 | 无 | **小目标细节保护 + 稀疏显著性** |

## 文件改动清单

### 新增文件
| 文件 | 说明 |
|---|---|
| `lib/models/layers/uav_wsp.py` | WSP 适配器核心模块（原 sglora.py 的全面改造） |

### 修改文件
| 文件 | 改动内容 |
|---|---|
| `lib/models/ostrack/ostrack.py` | 移除 OPLoRA/Neuro-OPLoRA/SGLoRA 引用，新增 UAV-WSP 注入 |
| `lib/train/actors/ostrack.py` | 替换 import 和日志键名 |
| `lib/config/ostrack/config.py` | 新增 `UAV_WSP` 配置段，旧方法标为 deprecated |
| `experiments/ostrack/*_uav_sglora.yaml` | SGLORA → UAV_WSP 配置，新增 spectral_beta, focus_lam_max |
| `tracking/analysis_results.py` | display_name 更新 |

### 未修改文件（保持向后兼容）
| 文件 | 原因 |
|---|---|
| `lib/models/layers/oplora.py` | 旧实验可能仍需加载 checkpoint |
| `lib/models/layers/neuro_oplora.py` | 同上 |
| `lib/models/layers/sglora.py` | 保留但不被任何代码引用 |

## 配置指南

### YAML 配置示例

```yaml
TRAIN:
  UAV_WSP:
    ENABLE: true
    LAYER_CONFIGS:
      - blocks: [7, 1]
        rank: 8
        top_k: 16
        alpha: 8.0
        spectral_beta: 1.0            # ← 新增：谱衰减指数
        targets: ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]
      - blocks: [8]
        rank: 2
        top_k: 16
        alpha: 8.0
        spectral_beta: 0.6            # B8 遮挡层用更低 β → 更多适应空间
        targets: ["attn.qkv", "attn.proj"]
    ENTROPY_LAM_MAX: 5e-5
    WARMUP_RATIO: 0.5
    ANNEAL_RATIO: 0.25
    GROUP_LASSO_LAM_MAX: 1e-5
    FOCUS_LAM_MAX: 1e-5               # ← 新增：聚焦正则化强度
```

### 参数说明

| 参数 | 默认值 | 作用 | 推荐范围 |
|---|---|---|---|
| `spectral_beta` | 1.0（per-group 可覆盖） | 谱衰减指数。低值→更多适应自由 | 0.3~2.0 |
| `focus_lam_max` | 1e-5 | Gini 聚焦正则化峰值强度 | 1e-6~1e-4 |
| `entropy_lam_max` | 1e-4 | 熵正则化峰值，推动门控二值化 | 1e-5~5e-4 |
| `warmup_ratio` | 0.33 | 正则化预热占比 | 0.25~0.5 |
| `anneal_ratio` | 0.33 | 正则化爬坡占比 | 0.2~0.5 |
| `group_lasso_lam_max` | 1e-5 | B 矩阵行稀疏 | 1e-6~1e-4 |

### spectral_beta 调参建议

| 场景 | β 建议 | 理由 |
|---|---|---|
| 通用反无人机追踪 | 1.0 | 默认值，兼顾保护与适应 |
| 小目标极多（<10px） | **0.5~0.8** | 更多细节适应空间 |
| 遮挡频繁 | **0.5~0.7** | 中低奇异值方向对遮挡恢复重要 |
| 大目标/背景类似 | **1.2~2.0** | 更保守，防止过适应 |
| 退化为原始硬投影 | **0.0** | 所有 d_i=1 → P=I-UU^T（与 SGLoRA 相同） |

## 训练建议

1. 首次使用时，保持除 `spectral_beta` 和 `FOCUS_LAM_MAX` 外的参数与之前 SGLoRA 一致
2. 先试 `spectral_beta=1.0`，在验证集上对比 β=0（退化为硬投影）的效果差异
3. 如果目标特别小（< 15px），尝试降低 β 到 0.6~0.8
4. `FOCUS_LAM_MAX` 从 1e-5 开始，观察门控稀疏度，过小时可提升到 5e-5

## 公式总结

$$P_R(w) = I - V_k D V_k^T, \quad P_L(w) = I - U_k D U_k^T$$
$$D = \text{diag}(d_1, \ldots, d_k), \quad d_i = \left(\frac{\sigma_i}{\sigma_1}\right)^\beta$$
$$\Delta W = \frac{\alpha}{r} \cdot \text{diag}(\sigma(s)) \cdot P_L(w) \cdot B \cdot A \cdot P_R(w)$$
$$L_{reg} = \lambda_e \cdot H(\sigma(s)) + \lambda_g \cdot \sum_i \|B_{i,:}\|_2 - \lambda_f \cdot G(\sigma(s))$$
