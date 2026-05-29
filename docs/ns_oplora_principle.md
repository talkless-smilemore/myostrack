# NS-OPLoRA 原理公式图

## 1. 总体结构

```mermaid
graph TB
    subgraph "输入"
        X["x ∈ R<sup>B×N×d_in</sup>"]
    end

    subgraph "Frozen 主干 (不可训练)"
        W["W ∈ R<sup>d_out×d_in</sup><br/>MAE 预训练权重<br/>❄ frozen"]
        B_BIAS["b ∈ R<sup>d_out</sup><br/>❄ frozen"]
    end

    X --> W
    W --> OUT_BASE["out_base = xWᵀ + b"]
    X --> ADAPTER

    subgraph "NS-OPLoRA Adapter (可训练 ★)"
        ADAPTER["────── 正交投影 + 低秩 + 神经元选择 ──────"]
    end

    ADAPTER --> OUT_LORA["out_lora = (α/r) · (M ⊙ u_r)"]
    OUT_BASE --> SUM((("+")))
    OUT_LORA --> SUM
    SUM --> Y["y = out_base + out_lora"]
```

## 2. Adapter 内部展开

```mermaid
graph TB
    subgraph "Step 1: 输入正交投影 (OPLoRA)"
        X2["x"] --> PR["P_R @ x<br/>= x - (x·V_k)·V_kᵀ"]
        PR_DESC["V_k: W 的 top-k_svd 右奇异向量<br/>P_R = I - V_k V_kᵀ<br/>将 x 投影到 V_k 的正交补空间"]
    end

    subgraph "Step 2: 低秩变换 (OPLoRA)"
        PR --> A["z = x_r · Aᵀ<br/>A ∈ R<sup>r×d_in</sup> ★"]
        A --> B["u = z · Bᵀ<br/>B ∈ R<sup>r×d_out</sup> ★"]
        LR_DESC["r ≪ min(d_in, d_out)<br/>默认 r=4 或 r=8"]
    end

    subgraph "Step 3: 输出正交投影 (OPLoRA)"
        B --> PL["P_L @ u<br/>= u - (u·U_k)·U_kᵀ"]
        PL_DESC["U_k: W 的 top-k_svd 左奇异向量<br/>P_L = I - U_k U_kᵀ"]
    end

    subgraph "Step 4: 神经元选择 (NeuroAda)"
        PL --> MASK["u_r = P_L(u)"]
        MASK --> GATE["u_r' = M ⊙ u_r"]
        GATE_DESC["M[i] = 1 当 i ∈ TopK(importance, d_out×p)<br/>importance[i] = mean(|W[i,:]|)<br/>M ∈ {0,1}<sup>d_out</sup>, ❄ static"]
    end

    subgraph "Step 5: 缩放输出"
        GATE --> SCALE["(α/r) · u_r'"]
    end
```

## 3. 神经元选择 (NeuroAda) 细节

```mermaid
graph LR
    subgraph "预训练权重矩阵 W"
        WM["W ∈ R<sup>d_out × d_in</sup>"]
        ROW1["神经元1: w₁₁ w₁₂ ... w₁_din"]
        ROW2["神经元2: w₂₁ w₂₂ ... w₂_din"]
        ROW3["..."]
        ROWN["神经元d_out: ..."]
    end

    WM --> IMP["按行计算 importance<br/>importance[i] = mean(|W[i,:]|)"]
    IMP --> SORT["排序 → Top-p%"]
    SORT --> MASK_BUILD["M[i]=1 for selected<br/>M[i]=0 for pruned"]

    MASK_BUILD --> VIS1["d_out×0.8 活跃神经元<br/>●●●●●●●●○○"]
    MASK_BUILD --> VIS2["d_out×0.5 活跃神经元<br/>●●●●●○○○○○"]
    MASK_BUILD --> VIS3["d_out×0.0 (完全冻结)<br/>○○○○○○○○○○"]
```

## 4. 完整前向公式

```
给定:
  W ∈ R^{d_out×d_in}   frozen 预训练权重
  b ∈ R^{d_out}         frozen bias
  U_k ∈ R^{d_out×k_svd} top-k_svd 左奇异向量
  V_k ∈ R^{d_in×k_svd}  top-k_svd 右奇异向量
  A ∈ R^{r×d_in}        可训练 ★
  B ∈ R^{d_out×r}       可训练 ★
  M ∈ {0,1}^{d_out}     静态神经元掩码 ❄
  α ∈ R                 缩放因子
  r ∈ N                 低秩维度

Forward:
  out_base = x @ W^T + b                             ... 冻结通路

  x_r   = x - (x @ V_k) @ V_k^T                      ... 输入正交投影
  z     = x_r @ A^T                                   ... 降维 r
  u     = z @ B^T                                     ... 升维 d_out
  u_r   = u - (u @ U_k) @ U_k^T                       ... 输出正交投影
  u_r'  = M ⊙ u_r                                     ... 神经元掩码
  out   = out_base + (α/r) * u_r'                     ... 残差融合

梯度:
  ∂L/∂A  ← 仅通过被选中神经元反向传播
  ∂L/∂B[i,:] = 0  当 M[i] = 0
  ∂L/∂W = 0, ∂L/∂U_k = 0, ∂L/∂V_k = 0, ∂L/∂M = 0   ... 全部冻结

推理 (merge):
  W_merged = W + (α/r) * diag(M) @ B @ A
  → 恢复为标准 nn.Linear，零额外开销
```

## 5. 层间差异化配置

```mermaid
graph TB
    subgraph "Block 0-2: 浅层 → 高适应"
        S0["M: d_out×0.80 活跃<br/>r=8, k_svd=16<br/>★ attn(qkv+proj) ★ mlp(fc1+fc2)"]
    end
    subgraph "Block 3,6,9: CE剪枝 → 注意力聚焦"
        S1["M: d_out×0.60 活跃<br/>r=4, k_svd=16<br/>★ attn(qkv+proj) ◇ mlp"]
    end
    subgraph "Block 4,5: 中层 → MLP轻调"
        S2["M: d_out×0.50 活跃<br/>r=4, k_svd=16<br/>◇ attn ★ mlp(fc1+fc2)"]
    end
    subgraph "Block 7,8,10,11: 深层 → 全冻"
        S3["M: d_out×0.00 活跃<br/>(不注入，无参数)<br/>◇ attn ◇ mlp"]
    end

    S0 --> S1 --> S2 --> S3
```

---

| 符号 | 来源 | 含义 |
|------|------|------|
| U_k, V_k, P_L, P_R | **OPLoRA** | SVD 正交投影，保护预训练主成分 |
| A, B, r, α | **OPLoRA** | 低秩适配矩阵，在正交子空间内学习 |
| M, importance[i] | **NeuroAda** | 神经元级幅值选择，每行独立决策 |
| per-block config | **本方案** | 层自适应差异化，按反无人机任务定制 |
