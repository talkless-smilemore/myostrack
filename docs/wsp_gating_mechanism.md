# UAV-WSP 门控机制（Gating Mechanism）完全说明

> 本文档详细解释 `lib/models/layers/uav_wsp.py` 中门控（gating）的实现原理，
> 包括参数定义、初始化、前向计算、正则化、以及推理合并五个阶段。

---

## 一、设计动机

在参数高效微调（PEFT）中，低秩适配器 $BA$ 的每个输出通道对最终结果的贡献应该不同——
有些通道编码了对**小目标**关键的特征，有些通道编码了背景或噪声。

门控（gate）的作用就是让模型**自己学会**每个输出通道的重要性权重：

- 重要通道 → 门控打开（接近 1），适配器更新充分传递
- 不重要通道 → 门控关闭（接近 0），适配器输出被抑制

这与反无人机追踪的需求高度吻合：小目标在特征空间中只占据极少数判别通道，
多数通道应被抑制。

---

## 二、参数定义

### 2.1 `gate_logit` — 每个输出通道一个可学习标量

**文件**：`uav_wsp.py` 第 311 行 / 第 357 行

```python
self.gate_logit = nn.Parameter(gate_logit_init)   # shape: [d_out]
```

- `d_out`：当前 Linear 层的输出维度（例如 qkv 的 768、MLP fc1 的 3072）
- 每个输出通道对应一个 `gate_logit` 值
- 类型是 `nn.Parameter` → 参与反向传播，通过梯度更新
- 取值范围：$(-\infty, +\infty)$，由 sigmoid 映射到 $(0, 1)$

### 2.2 与 LoRA 参数的关系

```
lora_A: [rank, d_in]     ← 低秩编码（输入 → 瓶颈）
lora_B: [d_out, rank]    ← 目标解码（瓶颈 → 输出）
gate_logit: [d_out]      ← 每个输出通道的门控权重 ← 本文重点
```

**关键**：门控是在 **B 矩阵之后**、**残差叠加之前**施加的，所以它控制的是"每个输出通道的适配器更新量"。

---

## 三、初始化

### 3.1 默认状态：全零初始化（无 SVD 时）

当 `rank <= 0` 或 SVD 失败时（第 311~312 行）：

```python
self.gate_logit = nn.Parameter(torch.zeros(out_features))  # 全 0
```

此时 `sigmoid(0) = 0.5`，所有通道的门控值从中性值开始学习。

### 3.2 细节显著性初始化（核心创新）

当 SVD 可用时（第 334~339 行）：

```python
gate_bias = _detail_saliency_from_svd(Uk, sigma, kk, self.spectral_beta)
gate_logit_init = gate_bias
```

#### 计算步骤（`_detail_saliency_from_svd()` 第 99~125 行）

**输入**：
- `Uk`: 左奇异矩阵 `[d_out, k]`，每列是一个奇异方向在输出通道上的投影
- `S`: 奇异值 `[k]`，$\sigma_1 > \sigma_2 > ... > \sigma_k$
- `k`, `beta`: 与 WSP 主模块一致

**Step 1 — 计算谱权重**：
```python
d = compute_spectral_weights(S, k, beta)   # [k], d_i = (σ_i / σ₁)^β
```
- $d_1 = 1$（最大奇异值方向完全保护）
- $d_k = (\sigma_k/\sigma_1)^\beta \ll 1$（最小奇异值方向几乎不保护）

**Step 2 — 计算每个输出通道的"细节重要性"**：
```python
detail = (Uk.pow(2) * (1.0 - d).unsqueeze(0)).sum(dim=1)   # [d_out]
```
**物理含义**：
- `Uk[i, j]²`：输出通道 i 在奇异方向 j 上的参与度
- `(1 - d_j)`：奇异方向 j 的"细节含量"——
  - 大奇异值方向（j=1）：$d_1 \approx 1 \rightarrow 1-d_1 \approx 0$ → 细节含量低（背景/粗粒度结构）
  - 小奇异值方向（j=k）：$d_k \ll 1 \rightarrow 1-d_k \approx 1$ → 细节含量高（边缘/纹理/小目标响应）
- 求和：通道 i 在所有 k 个奇异方向上的细节总参与度

**Step 3 — 归一化到 $[-0.3, 0.3]$**：
```python
detail = 0.6 * detail / detail.max() - 0.3
```
- 与细节方向耦合强的通道 → 正偏置 → `sigmoid ≈ 0.55~0.65`（初始打开）
- 与背景方向耦合强的通道 → 负偏置 → `sigmoid ≈ 0.35~0.45`（初始关闭）
- 偏置范围控制在 ±0.3，不会压倒任务损失的梯度

**为什么只给 ±0.3 的小偏置？**

如果初始化偏置过大，门控在训练初期就固化，阻碍任务损失对门控的调节。
±0.3 对应 sigmoid 的约 0.35~0.65，是一个温和的头马效应——
给小目标通道一个先发优势，但不禁止其他通道后续通过学习改变门控值。

---

## 四、前向计算

### 4.1 门控在完整前向中的位置

**文件**：`uav_wsp.py` 第 400~433 行

```
输入 x
  │
  ├── Frozen path（原始权重，不更新）:
  │     out = x @ W₀ᵀ + b
  │
  └── Adapter path（可学习，被门控调控）:
          P_R(w) · x                  ← 加权输入投影
          → z = x_r @ Aᵇ              ← 低秩编码   [d_in → r]
          → u = z @ Bᵇ                ← 目标解码   [r → d_out]
          → P_L(w) · u                ← 加权输出投影
          → g · u_r                   ← ★★★ 门控 ★★★
          → (α/r) · u_g               ← 缩放
  │
  └── out += 缩放后的适配器输出
```

### 4.2 门控的数学形式

```python
g = torch.sigmoid(self.gate_logit)     # [d_out], 每个输出通道 ∈ (0, 1)
ug = ur * g                             # [B, N, d_out], 逐元素乘法（broadcast）
return out + scale * ug                 # 残差连接
```

- `g` 的形状是 `[d_out]`，在批维度 `[B, N, d_out]` 上 broadcast
- `g[i]` 控制第 i 个输出通道的适配器更新幅度

**极端情况**：
- 若 `g[i] → 1`（门控全开）：
  第 i 通道的适配器输出完全传递，`Δy[i] = (α/r) · u_r[i]`
- 若 `g[i] → 0`（门控关闭）：
  第 i 通道的适配器输出被抑制，`Δy[i] ≈ 0`，退化为纯 frozen path

### 4.3 门控与投影的关系对比

WSP 中有两种"保护/筛选"机制，但作用于不同层面：

| 机制 | 作用位置 | 方式 | 是否可学习 |
|------|---------|------|-----------|
| **加权谱投影** $P= I-UDU^T$ | 适配器的**输入和输出** | 在频率域保护 top-k 奇异方向不被 LoRA 改变 | **否**（基于 SVD 固定） |
| **门控** $\text{sigmoid}(s)$ | 适配器的**最终输出** | 在通道域筛选哪些输出通道的更新生效 | **是**（梯度驱动） |

两者互补：投影确保 LoRA 不去破坏 pretrained 权重的重要频率成分，
门控确保只有对当前任务有用的输出通道才传递适配器信号。

---

## 五、正则化（Regularisation）

门控本身是 nn.Parameter，如果没有任何约束，它只会被任务损失（GIoU + L1 + Focal）驱动。
WSP 引入了**三项**正则化来塑造门控的行为模式。

**文件**：`uav_wsp.py` 第 471~505 行

### 5.1 熵正则化 — 推动门控二值化

```python
if lam_e > 0:
    g = torch.sigmoid(self.gate_logit)
    entropy = -(g * log(g) + (1-g) * log(1-g))
    loss += lam_e * entropy.mean()
```

**效果**：
- 熵 $H(g) = -[g \log g + (1-g) \log(1-g)]$
- 当 $g=0.5$ 时熵最大（$H=\log 2$），当 $g\to 0$ 或 $g\to 1$ 时熵最小（$H\to 0$）
- 最小化熵 → 推动门控两极分化：要么开、要么关

**反无人机意义**：小目标只需要少数判别通道响应，其他通道应该被彻底关闭。

### 5.2 Group-Lasso — 神经元级稀疏（间接影响门控）

```python
if lam_g > 0:
    row_norms = torch.norm(self.lora_B, p=2, dim=1)   # [d_out]
    loss += lam_g * row_norms.mean()
```

对 `lora_B` 的行施加 L2 正则 → 如果某行的所有列都接近 0，该输出通道的适配器输出就消失。
这间接与门控协同：门控关 + B 行归零 → 彻底禁用该通道。

### 5.3 Gini 聚焦正则化 — 推动门控集中（V2 关闭）

```python
if lam_f > 0:
    g = torch.sigmoid(self.gate_logit)
    gini = self._gini_coefficient(g)
    loss -= lam_f * gini    # 负号 = 最大化 Gini
```

**Gini 系数**（第 440~454 行）：
$$
G(g) = \frac{\sum_{i=1}^n (2i - n - 1) \cdot g_{(i)}}{n \cdot \sum g_{(i)}}
$$
- $0 \leq G \leq 1$，越大表示门控值越集中在少数通道
- 最大化 Gini → 强制只有极少数通道的 gate > 0.5，其他全部趋向 0

**V2 关闭原因**：Gini 正则化对门控的约束过于激进，导致适配器可用的有效通道太少，
在小目标上的表现反而不如不加。V2 的 `FOCUS_LAM_MAX=0.0` 禁用了该项。

### 5.4 正则化调度

```python
def _schedule_lam(self):
    p = self._global_progress       # [0, 1], 由 set_progress() 设置
    if p < self.warmup_ratio:       # epoch 0~20:  λ=0
        ramp = 0.0
    elif p < self.warmup_ratio + self.anneal_ratio:  # epoch 20~30:  ramp
        ramp = (p - self.warmup_ratio) / self.anneal_ratio
    else:                           # epoch 30~40:  λ=max
        ramp = 1.0
    return (lam_max * ramp, ...)
```

在训练早期（前 20 epoch）正则化权重为 0，门控自由学习；
在中期（20~30 epoch）逐步施加正则化压力；
在后期（30~40 epoch）正则化全开，门控被推向预期模式。

---

## 六、推理阶段：门控合并

**文件**：`uav_wsp.py` 第 511~540 行

训练完成后，门控值被**永久性地合并到权重矩阵中**，实现推理零开销：

```python
def merge_to_weight(self):
    g = torch.sigmoid(self.gate_logit)        # [d_out]
    BA = self.lora_B @ self.lora_A              # [d_out, d_in]
    # 加权投影 ...
    BA = BA - BA_Vk @ Vk.t()                    # P_R 投影
    BA = BA - Uk @ UkT_BA                       # P_L 投影
    delta = scale * g.unsqueeze(1) * BA          # ★ 门控乘入
    W_merged = self.weight.data + delta           # 合并到 frozen weight
    return W_merged, bias_merged
```

**门控在推理时的位置**：
1. `g[0]` 控制了 `delta[0, :]` 整行——输出通道 0 的所有输入连接都乘以相同的门控值
2. 合并后，推理时不再有独立的门控计算，结构等价于一个普通的 Linear 层

---

## 七、数据流总览

### 7.1 门控值生命周期

```
初始化阶段:
  gate_logit = 0                          → sigmoid = 0.5 (均匀中性)
  gate_logit += detail_saliency_bias      → sigmoid = 0.35~0.65 (细节偏置)

训练阶段:
  gate_logit 通过梯度传播更新
  ↓
  熵正则化:   推动 gate → 0 或 1
  Group-Lasso: 协同关闭不重要通道
  Gini聚焦:   [V2 关闭] 推动集中在少数通道

推理阶段:
  gate_logit → sigmoid(gate_logit) → 乘入合并权重
  ↓
  WeightedSpectralProjection 退化回 nn.Linear 的计算图
```

### 7.2 代码调用链

```
训练循环:
  ltr_trainer.py::train_epoch()
    → data['epoch'] 传递给 actor
    → actor 中调用:
        WeightedSpectralProjection.set_progress(epoch/EPOCH)  ← 更新 λ 调度
        self.net(template, search, ...)                        ← 前向
          → WeightedSpectralProjection.forward()
            → sigmoid(gate_logit) * ur                        ← 门控应用
        collect_wsp_regularisation(self.net)                   ← 收集正则
          → WeightedSpectralProjection.regularisation_loss()
            → 熵 + Group-Lasso + Gini                         ← 门控正则化
        loss = task_loss + reg_loss                            ← 合并

推理:
  WeightedSpectralProjection.merge_to_weight()                 ← 门控合并
  → 存储合并后的 weight
  → 之后所有前向 = torch.nn.functional.linear(x, W_merged, bias)
```

---

## 八、关键超参数与调优

### 8.1 门控相关参数

| 参数 | 位置 | 默认(V2) | 作用 | 调优方向 |
|------|------|---------|------|---------|
| `rank` | per-group | 2~12 | 低秩瓶颈维度，rank 高则 gate 控制更多通道 | 目标小→可降低 |
| `ENTROPY_LAM_MAX` | 全局 | 1e-5 | 门控二值化强度 | 高→gate 更硬，低→gate 可中值 |
| `WARMUP_RATIO` | 全局 | 0.5 | 正则化延迟生效的比例 | 训练长→可提高 |
| `ANNEAL_RATIO` | 全局 | 0.25 | 正则化爬坡比例 | 决定门控从自由到固化的速度 |
| `FOCUS_LAM_MAX` | 全局 | 0.0 | Gini 集中度 | V2 关闭，如需开启从 1e-6 开始试 |

### 8.2 门控行为的可视化判断

训练日志中关注 `Loss/wsp_reg` 的值：

- **Entropy 下降**：门控在二值化（好迹象）
- **Gini 上升**（如果启用）：门控在集中（好迹象）
- `Loss/wsp_reg` 最终稳定在一个低值：门控模式固化完成

检查点中查看门控分布：
```python
g = torch.sigmoid(model.gate_logit)
print(f"gate mean={g.mean():.3f}, <0.1={g.lt(0.1).float().mean():.3f}, >0.9={g.gt(0.9).float().mean():.3f}")
```
期望结果：多数通道 < 0.1，少数通道 > 0.9，中间值很少。

---

## 九、代码对照索引

| 功能 | 文件:行号 |
|------|----------|
| `gate_logit` 参数声明 | `uav_wsp.py:357` |
| 细节显著性初始化 | `uav_wsp.py:99-125`, 被 `337-339` 调用 |
| 前向门控应用 | `uav_wsp.py:430-431` |
| 门控合并到权重 | `uav_wsp.py:521, 537` |
| 熵正则化（门控二值化） | `uav_wsp.py:487-491` |
| Gini 系数计算 | `uav_wsp.py:440-454` |
| Gini 聚焦正则化（V2 关闭） | `uav_wsp.py:498-503` |
| 正则化调度 | `uav_wsp.py:456-469` |
| 全局进度设置（actor 中调用） | `uav_wsp.py:239-242` |
| 正则收集（actor 中调用） | `uav_wsp.py:547-565` |

---

## 十、总结

```
gate_logit  [d_out]          ← 每个输出通道一个可学习标量
    │
    ├── 初始化: (±0.3 偏置)  ← 小目标通道先打开一点
    │
    ├── 前向: sigmoid → (0,1) → × ur  ← 逐通道缩放适配器输出
    │
    ├── 正则化: 熵 → 二值化   ← 推动 gate 要么 0 要么 1
    │          Group-Lasso → 协同抑制
    │          Gini → 集中   ← V2 关闭
    │
    └── 推理: 乘入 BA → merge 到 W₀  ← 零开销
```
