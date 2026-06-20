# UAV-WSP V2: 反无人机小目标适配器完整改动文档

## 历史沿革

| 版本 | 核心方法 | 问题 |
|------|----------|------|
| OPLoRA | 硬正交投影 P=I-UUᵀ + LoRA | 静态门控，一刀切保护 |
| NS-OPLoRA | OPLoRA + NeuroAda 静态神经元掩码 | 掩码不可学习 |
| SGLoRA | 硬投影 + 可学习 soft gate + 熵正则化 | 过度约束，-3 点 vs OPLoRA |
| **UAV-WSP V1** | **加权投影 P=I-UDUᵀ** + 细节显著性 gate init + Gini 聚焦 | 比 SGLoRA 好一点，仍 -3 点 |
| **UAV-WSP V2** | V1 + 参数放松 + 时间连续性损失 | **当前版本** |

---

## 一、核心数学公式

### 1.1 加权谱投影（V1 已引入）

$$
\Delta W = \frac{\alpha}{r} \cdot \text{diag}(\sigma(s)) \cdot P_L(w) \cdot B \cdot A \cdot P_R(w)
$$

$$
P_R(w) = I - V_k D V_k^T, \quad P_L(w) = I - U_k D U_k^T
$$

$$
D = \text{diag}(d_1, \ldots, d_k), \quad d_i = \left(\frac{\sigma_i}{\sigma_1}\right)^\beta
$$

**关键区别**：原 OPLoRA/SGLoRA 的投影是 $P = I - U_k U_k^T$（硬切断，所有 top-k 方向一视同仁），我们的投影是 $P = I - U_k D U_k^T$（按奇异值比值渐变保护）。

| β 值 | 效果 |
|------|------|
| β=0 | $d_i=1$ → 退化为 OPLoRA 硬切断 |
| β=1 | $d_i = \sigma_i/\sigma_1$ → 线性衰减（默认） |
| β=2 | $d_i = (\sigma_i/\sigma_1)^2$ → 更保守 |
| **β=0.5** | **当前 V2 默认值 → 更自由的适应** |
| **β=0.3** | **遮挡层 B8/B4/B10 → 几乎不保护** |

### 1.2 细节显著性门控初始化（V1 已引入）

$$gate\_logit_i = 0.6 \cdot \frac{detail_i}{detail_{max}} - 0.3$$

$$detail_i = \sum_{j=1}^k U_{k}[i,j]^2 \cdot (1 - d_j)$$

- 与小奇异值（细节）方向耦合强的通道 → 初始门控偏高 (~0.55)
- 与大奇异值（背景）方向耦合强的通道 → 初始门控偏低 (~0.45)

### 1.3 时间连续性损失（V2 新增）

$$L_{temp} = \text{smooth}_{L1}( \text{predBox}_{search}, \text{predBox}_{next} )$$

$$L_{total} = L_{task} + L_{reg} + \lambda_{temp} \cdot L_{temp}$$

**物理意义**：无人机有惯性，连续两帧之间目标不可瞬移。这是反无人机场景独有的强物理先验，通用跟踪数据集（LaSOT、GOT-10k）不具备此性质。

### 1.4 总正则化

$$L_{reg} = \lambda_e(t) \cdot H(\sigma(s)) + \lambda_g(t) \cdot \sum_i \|B_{i,:}\|_2 - \lambda_f(t) \cdot G(\sigma(s))$$

| 项 | 含义 | V1 峰值 | V2 峰值 | 变化 |
|----|------|---------|---------|------|
| $H(\sigma(s))$ | 熵正则 → 门控二值化 | 5e-5 | **1e-5** | ↓ 80% |
| $\sum\|B_i\|_2$ | 行 group-lasso → 神经元级稀疏 | 1e-5 | **5e-6** | ↓ 50% |
| $-G(\sigma(s))$ | Gini 聚焦 → 稀疏显著性 | 1e-5 | **0** | 关闭 |

---

## 二、文件改动清单

### 2.1 新增文件

| 文件 | 描述 |
|------|------|
| `lib/models/layers/uav_wsp.py` | WSP 适配器核心模块（~500 行） |
| `experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp.yaml` | V2 实验配置 |
| `docs/uav_wsp_adaptation.md` | 算法文档 |

### 2.2 修改文件

| 文件 | 改动内容 | 行数 |
|------|----------|------|
| `lib/models/ostrack/ostrack.py` | 移除 OPLoRA/NeuroOPLoRA/SGLoRA，替换为 UAV-WSP | ~30 行 |
| `lib/train/actors/ostrack.py` | 替换 import + **新增时间连续性损失** | ~40 行 |
| `lib/train/trainers/base_trainer.py` | **新增 `_try_save_best()` 方法 + `save_checkpoint(tag)`** | ~40 行 |
| `lib/train/trainers/ltr_trainer.py` | 在 `train_epoch` 中 stats 重置前调用 `_try_save_best()` | ~3 行 |
| `lib/train/data/sampler.py` | **新增 `enable_temporal_smoothness` 参数 + t+1 帧采样** | ~20 行 |
| `lib/train/data/processing.py` | **处理 `search_next`：复用搜索帧的 crop 窗口** | ~25 行 |
| `lib/train/base_functions.py` | 传递 temporal flag 到 sampler | ~3 行 |
| `lib/config/ostrack/config.py` | 新增 `UAV_WSP`、`TEMPORAL_SMOOTHNESS`、`SAVE_BEST` 配置段 | ~20 行 |

### 2.3 未修改文件（保留向后兼容）

| 文件 | 原因 |
|------|------|
| `lib/models/layers/oplora.py` | 旧 checkpoint 可能需要加载 |
| `lib/models/layers/neuro_oplora.py` | 同上 |
| `lib/models/layers/sglora.py` | 同上 |

---

## 三、层选择策略（V2）

### 基于扰动实验的综合敏感性 = PatchShuffle + GaussBlur + PhaseScramble

| Block | 敏感性 | 分组 | rank | β | targets | 理由 |
|-------|--------|------|------|---|---------|------|
| B0 | 0.729 | ❄冻结 | - | - | - | patch embedding，纯投影 |
| **B1** | 1.224 | 🟢 A | **12** | 0.5 | qkv/proj/fc1/fc2 | 纹理点火，+4.6× jump |
| **B2** | 1.061 | 🟣 C | **8** | 0.5 | fc1/fc2 | 全局空间构建 |
| **B3** | 1.170 | 🟡 B | **12** | 0.5 | qkv/proj | CE-1 空间筛选入口 |
| **B4** | 1.089 | 🟠 **E** | **2** | 0.3 | qkv/fc1 | **V2 新增解冻** |
| **B5** | 1.380 | 🟣 C | **8** | 0.5 | fc1/fc2 | CE 后局部重建 |
| **B6** | 1.559 | 🟡 B | **12** | 0.5 | qkv/proj | CE-2 综合决策 #1 |
| **B7** | 1.491 | 🟢 A | **12** | 0.5 | qkv/proj/fc1/fc2 | 融合枢纽 #2 |
| **B8** | 0.843 | 🟠 D | 2 | 0.3 | qkv/proj | 最纯局部纹理，遮挡层 |
| **B9** | 1.479 | 🟡 B | **12** | 0.5 | qkv/proj | CE-3 纹理筛选 #3 |
| **B10** | 1.246 | 🟠 **E** | **2** | 0.3 | qkv/fc1 | **V2 新增解冻** |
| B11 | 1.025 | ❄冻结 | - | - | - | 稳定输出 |

**注入统计**：10 层 / 28 个 WSP 模块 / 2 层冻结 (B0, B11)

---

## 四、YAML 配置完整说明

```yaml
# === UAV-WSP 核心配置 ===
UAV_WSP:
  ENABLE: true
  LAYER_CONFIGS:
    # 每组指定 blocks / rank / top_k / alpha / spectral_beta / targets
    # β=0.5: 渐变保护，给中奇异值方向更多适应空间
    # β=0.3: 接近不保护，用于遮挡层/轻量解冻层

  # 正则化调度
  ENTROPY_LAM_MAX: 1e-5       # epoch 30~40 峰值，仅 1/5 原值
  WARMUP_RATIO: 0.5           # epoch 0~20: λ=0
  ANNEAL_RATIO: 0.25          # epoch 20~30: λ ramp
  GROUP_LASSO_LAM_MAX: 5e-6   # B 矩阵行稀疏，减半
  FOCUS_LAM_MAX: 0.0          # Gini 聚焦关闭

# === 时间连续性损失 ===
TEMPORAL_SMOOTHNESS:
  ENABLE: true
  LOSS_WEIGHT: 0.05           # L_temp = 0.05 × smooth_l1(pred, pred_next)

# === YOLO-style 最佳模型保存 ===
SAVE_BEST: true
SAVE_BEST_METRIC: "Loss/total"  # val 上 Loss/total 最低的才是 best
SAVE_EPOCHS: [20]               # epoch 20 再额外存一份中间 snapshot
```

---

## 五、训练命令

```bash
cd E:/code/UAV/OSTrack-main && OMP_NUM_THREADS=1 D:/miniconda3/envs/ostrack/python.exe -u lib/train/run_training.py --script ostrack --config vitb_256_mae_ce_32x4_ep300_uav_wsp --save_dir output/train
```

训练产物：
```
output/train/checkpoints/train/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp/
  ├── OSTrack_ep0020.pth.tar   ← epoch 20 snapshot
  ├── OSTrack_ep0036.pth.tar   ← 最后 5 个 epoch 自动保存
  ├── OSTrack_ep0037.pth.tar
  ├── OSTrack_ep0038.pth.tar
  ├── OSTrack_ep0039.pth.tar
  ├── OSTrack_ep0040.pth.tar
  └── OSTrack_best.pth.tar     ← 验证集 Loss/total 最低的那一次
```

---

## 六、参数调优建议

### spectral_beta 调节

| 场景 | β 建议 | 理由 |
|------|--------|------|
| 通用反无人机 | 0.5 | 当前默认，已验证 |
| 目标 < 10px | 0.3~0.4 | 需要更多细节适应 |
| 目标 ~30px+ | 0.7~1.0 | 接近通用跟踪，可保守 |
| 背景复杂 | 0.5~0.7 | 需要保留一些主成分 |
| 纯天空背景 | 0.2~0.3 | 主成分几乎全是背景 |

### 正则化权重调节

| 参数 | 推荐范围 | 太高会 | 太低会 |
|------|----------|--------|--------|
| ENTROPY_LAM_MAX | 1e-5 ~ 5e-5 | 门控过早二值化 | 门控学不到稀疏模式 |
| GROUP_LASSO_LAM_MAX | 1e-6 ~ 1e-5 | 神经元被过度修剪 | 冗余通道不压缩 |
| FOCUS_LAM_MAX | 0 ~ 5e-5 | 门控过于集中 | 无稀疏集中效果 |

### 时间连续性权重

| LOSS_WEIGHT | 效果 |
|-------------|------|
| 0.01 | 极弱约束，几乎不影响训练 |
| 0.05 | 推荐，温和正则化 |
| 0.1 | 较强约束，可能妨碍快速运动 |
| 0.5 | 过强，预测会过于平滑失去精度 |

---

## 七、推理时加载最佳模型

```python
# 加载 best model（而非最后 epoch）
from lib.train.trainers.base_trainer import BaseTrainer
# ...
trainer.load_checkpoint('output/train/checkpoints/.../OSTrack_best.pth.tar')
```

或直接在 TEST yaml 中：
```yaml
TEST:
  CHECKPOINT: "path/to/OSTrack_best.pth.tar"
```

---

## 八、已知限制与后续方向

| 方向 | 状态 | 预期收益 |
|------|------|----------|
| 运动模糊增强 | 待实现 | +0.5~1 点 |
| IR 对比度增强 | 待实现 | +0.3~0.5 点（仅 IR 数据） |
| 多尺度模板 | 待实现 | +0.5~1 点 |
| A 矩阵正则化 | 待探索 | 未知 |
