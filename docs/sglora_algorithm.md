# SGLoRA 算法流程图

## 1. 完整前向传播

```mermaid
graph TB
    X["输入 x ∈ ℝ<sup>B×N×d_in</sup>"] --> FROZEN
    X --> ADAPTER

    subgraph FROZEN["❄ 冻结通路"]
        W0["W₀ ∈ ℝ<sup>d_out×d_in</sup><br/>MAE 预训练权重"]
        B0["b ∈ ℝ<sup>d_out</sup>"]
        OUT_BASE["out_base = x·W₀ᵀ + b"]
    end

    subgraph ADAPTER["★ SGLoRA 可训练通路  ΔW = (α/r)·diag(g)·P_L·B·A·P_R"]
        subgraph P_R["P_R: 输入正交投影"]
            Vk["V_k: W₀ 的 top-k 右奇异向量"]
            XR["x_r = x - (x·V_k)·V_kᵀ<br/>= (I - V_k V_kᵀ)·x"]
        end

        subgraph LR["低秩变换"]
            A["z = x_r·Aᵀ<br/>A ∈ ℝ<sup>r×d_in</sup> ★"]
            B["u = z·Bᵀ<br/>B ∈ ℝ<sup>d_out×r</sup> ★"]
        end

        subgraph P_L["P_L: 输出正交投影"]
            Uk["U_k: W₀ 的 top-k 左奇异向量"]
            UR["u_r = u - (u·U_k)·U_kᵀ<br/>= (I - U_k U_kᵀ)·u"]
        end

        subgraph GATE["频谱门控"]
            S["s ∈ ℝ<sup>d_out</sup> ★ 可学习门控参数<br/>初始化为 0 → g = σ(0) = 0.5"]
            UG["u_g = σ(s) ⊙ u_r<br/>逐神经元软门控 ∈ [0,1]"]
        end

        X --> Vk
        Vk --> XR
        XR --> A
        A --> B
        B --> Uk
        Uk --> UR
        UR --> S
        S --> UG
    end

    OUT_BASE --> SUM(("+"))
    UG --> SCALE["× (α/r)"]
    SCALE --> SUM
    SUM --> Y["输出 y ∈ ℝ<sup>B×N×d_out</sup>"]
```

## 2. 初始化流程

```mermaid
graph TB
    START["W₀ ∈ ℝ<sup>d_out×d_in</sup> (冻结)"] --> SVD["SVD: W₀ = U·Σ·Vᵀ"]
    SVD --> TOPK["取 top-k 奇异向量<br/>U_k = U[:, :k] ∈ ℝ<sup>d_out×k</sup><br/>V_k = V[:k, :]ᵀ ∈ ℝ<sup>d_in×k</sup>"]

    TOPK --> P_BUFFERS["❄ 冻结 buffer:<br/>P_R = I - V_k V_kᵀ<br/>P_L = I - U_k U_kᵀ"]
    TOPK --> GATE_INIT["★ 门控初始化: s = 0 ∈ ℝ<sup>d_out</sup><br/>→ g = σ(s) = 0.5 (全部中性)<br/>P_L 已保护主成分，门控无需重复打压"]

    TOPK --> LORA_INIT["★ LoRA 初始化:<br/>A ~ Kaiming Uniform<br/>B = 0"]

    P_BUFFERS --> READY["模块就绪"]
    GATE_INIT --> READY
    LORA_INIT --> READY

    NOTE["关键设计: 门控不基于 W₀ 幅值或谱重要性初始化。<br/>P_L 已经投影掉了 U_k 方向的成分，<br/>门控从 0.5 起步，让数据决定每个神经元的稀疏模式。"]
    READY -.-> NOTE
```

## 3. 训练过程 & 熵调度

```mermaid
graph TB
    subgraph EPOCH0["Epoch 0 ~ 13 (warmup_ratio=0.33)"]
        WARM["λ_e = 0, λ_g = 0"]
        FREE["门控自由学习<br/>无稀疏压力<br/>g ∈ (0,1) 任意取值"]
    end

    subgraph EPOCH1["Epoch 13 ~ 26 (anneal_ratio=0.33)"]
        RAMP["λ_e: 0 → 5e-5, λ_g: 0 → 1e-5"]
        GRADUAL["门控逐渐收紧<br/>H(g) 推动 g → 0 或 1<br/>行 group-lasso 推动 B 行稀疏"]
    end

    subgraph EPOCH2["Epoch 26 ~ 40"]
        FULL["λ_e = 5e-5, λ_g = 1e-5"]
        BINARY["门控近似二值<br/>≈ 推理时可直接 merge<br/>或被 0 门控的神经元 = 永久关闭"]
    end

    WARM --> RAMP --> FULL

    LOSS["总损失: L = L_task + λ_e(t)·H(g) + λ_g(t)·Σ_i‖B_i‖₂"]
    ENTROPY["H(g) = -g·log(g) - (1-g)·log(1-g)<br/>最小化 → g → 0 或 g → 1  (二值化)"]
    GLASSO["Σ_i‖B_i‖₂: 行 group-lasso<br/>推动整行 → 0  (神经元级稀疏)"]

    FULL --> LOSS
    LOSS --> ENTROPY
    LOSS --> GLASSO
```

## 4. 推理 / Merge

```mermaid
graph LR
    subgraph TRAIN["训练态"]
        T_FWD["y = x·W₀ᵀ + (α/r)·diag(σ(s))·P_L·B·A·P_R·x"]
    end

    subgraph MERGE["Merge (一次性)"]
        M1["g = σ(s) ∈ [0,1]<sup>d_out</sup>"]
        M2["ΔW = (α/r)·diag(g)·P_L·B·A·P_R"]
        M3["W_merged = W₀ + ΔW"]
    end

    subgraph INFER["推理态 (零额外开销)"]
        I_FWD["y = x·W_mergedᵀ + b<br/>等价于标准 nn.Linear"]
    end

    TRAIN --> MERGE --> INFER
```

## 5. δW 结构展开

```mermaid
graph TB
    subgraph DELTA["ΔW = (α/r) · diag(g) · P_L · B · A · P_R"]
        direction LR
        G["diag(g)<br/>d_out×d_out<br/>对角门控"]
        PL["P_L = I - U_k U_kᵀ<br/>d_out×d_out<br/>输出投影"]
        B2["B<br/>d_out×r"]
        A2["A<br/>r×d_in"]
        PR["P_R = I - V_k V_kᵀ<br/>d_in×d_in<br/>输入投影"]
    end

    NOTE_DELTA["维度: d_out×d_in  秩 ≤ r<br/>投影到 W₀ 主成分的正交补空间<br/>门控选择性地激活输出神经元"]
```

## 关键公式总结

| 组件 | 公式 | 状态 |
|------|------|------|
| 冻结权重 | W₀, b | ❄ 不可训练 |
| 输入投影 | P_R = I - V_k V_kᵀ | ❄ SVD 缓存 |
| 输出投影 | P_L = I - U_k U_kᵀ | ❄ SVD 缓存 |
| 低秩矩阵 | A ∈ ℝ^{r×d_in}, B ∈ ℝ^{d_out×r} | ★ 可训练 |
| 频谱门控 | g = σ(s), s ∈ ℝ^{d_out} | ★ 可训练，从 0.5 初始化 |
| 熵正则 | H(g) = -Σ[g·log(g) + (1-g)·log(1-g)] | 动态调度 λ_e(t) |
| 行稀疏 | Σ_i ‖B_{i,:}‖₂ | 动态调度 λ_g(t) |
| 最终适配 | ΔW = (α/r)·diag(g)·P_L·B·A·P_R | — |

## 与 NS-OPLoRA 的差异

| | NS-OPLoRA | SGLoRA |
|---|---|---|
| 公式 | ΔW = (α/r)·M⊙P_L·B·A·P_R | ΔW = (α/r)·diag(g)·P_L·B·A·P_R |
| 门控 | M ∈ {0,1}, 静态, 基于 mean(\|W₀\|) | g = σ(s) ∈ [0,1], 可学习, 中性初始化 |
| 稀疏化 | 固定 keep_ratio | 熵 + group-lasso 动态调度 |
| 门控和 P_L 的关系 | 无交互（先后独立作用） | 门控放 P_L 之后，互补不重复 |

![[mermaid-diagram.png]]