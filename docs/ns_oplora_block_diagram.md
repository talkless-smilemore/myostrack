# NS-OPLoRA Block Injection Diagram

```mermaid
graph TB
    subgraph "Template + Search Tokens"
        Z[模板 token z<br/>8×8=64 tokens]
        X[搜索 token x<br/>16×16=256 tokens]
    end

    Z --> CAT["concat(z, x)<br/>320 tokens"]
    X --> CAT

    CAT --> B0

    subgraph "Block 0 ─ 浅层 Group A — 强适应"
        B0[CEBlock]
        B0_N1["norm1 ✓ frozen"] --> B0_A
        B0_A["attn ★ NS-OPLoRA<br/>qkv: lora_A/B<br/>proj: lora_A/B"]
        B0_A --> B0_CE["CE 剪枝 (keep 70%)"]
        B0_CE --> B0_N2["norm2 ✓ frozen"]
        B0_N2 --> B0_M["mlp ★ NS-OPLoRA<br/>fc1: lora_A/B<br/>fc2: lora_A/B"]
    end

    B0 --> B1

    subgraph "Block 1 ─ 浅层 Group A — 强适应"
        B1[CEBlock]
        B1_N1["norm1 ✓ frozen"] --> B1_A
        B1_A["attn ★ NS-OPLoRA<br/>qkv: lora_A/B<br/>proj: lora_A/B"]
        B1_A --> B1_N2["norm2 ✓ frozen"]
        B1_N2 --> B1_M["mlp ★ NS-OPLoRA<br/>fc1: lora_A/B<br/>fc2: lora_A/B"]
    end

    B1 --> B2

    subgraph "Block 2 ─ 浅层 Group A — 强适应"
        B2[CEBlock]
        B2_N1["norm1 ✓ frozen"] --> B2_A
        B2_A["attn ★ NS-OPLoRA<br/>qkv: lora_A/B<br/>proj: lora_A/B"]
        B2_A --> B2_N2["norm2 ✓ frozen"]
        B2_N2 --> B2_M["mlp ★ NS-OPLoRA<br/>fc1: lora_A/B<br/>fc2: lora_A/B"]
    end

    B2 --> B3

    subgraph "Block 3 ─ CE剪枝层 Group B — 注意力聚焦"
        B3[CEBlock]
        B3_N1["norm1 ✓ frozen"] --> B3_A
        B3_A["attn ★ NS-OPLoRA<br/>qkv: lora_A/B<br/>proj: lora_A/B"]
        B3_A --> B3_CE["CE 剪枝 (keep 70%)"]
        B3_CE --> B3_N2["norm2 ✓ frozen"]
        B3_N2 --> B3_M["mlp ◇ 冻结<br/>fc1: weight+bias<br/>fc2: weight+bias"]
    end

    B3 --> B4

    subgraph "Block 4 ─ 中层 Group C — MLP轻调"
        B4[CEBlock]
        B4_N1["norm1 ✓ frozen"] --> B4_A
        B4_A["attn ◇ 冻结<br/>qkv: weight+bias<br/>proj: weight+bias"]
        B4_A --> B4_N2["norm2 ✓ frozen"]
        B4_N2 --> B4_M["mlp ★ NS-OPLoRA<br/>fc1: lora_A/B<br/>fc2: lora_A/B"]
    end

    B4 --> B5

    subgraph "Block 5 ─ 中层 Group C — MLP轻调"
        B5[CEBlock]
        B5_N1["norm1 ✓ frozen"] --> B5_A
        B5_A["attn ◇ 冻结<br/>qkv: weight+bias<br/>proj: weight+bias"]
        B5_A --> B5_N2["norm2 ✓ frozen"]
        B5_N2 --> B5_M["mlp ★ NS-OPLoRA<br/>fc1: lora_A/B<br/>fc2: lora_A/B"]
    end

    B5 --> B6

    subgraph "Block 6 ─ CE剪枝层 Group B — 注意力聚焦"
        B6[CEBlock]
        B6_N1["norm1 ✓ frozen"] --> B6_A
        B6_A["attn ★ NS-OPLoRA<br/>qkv: lora_A/B<br/>proj: lora_A/B"]
        B6_A --> B6_CE["CE 剪枝 (keep 70%)"]
        B6_CE --> B6_N2["norm2 ✓ frozen"]
        B6_N2 --> B6_M["mlp ◇ 冻结<br/>fc1: weight+bias<br/>fc2: weight+bias"]
    end

    B6 --> B7

    subgraph "Block 7 ─ 深层 Group D — 完全冻结"
        B7[CEBlock]
        B7_N1["norm1 ✓ frozen"] --> B7_A
        B7_A["attn ◇ 冻结<br/>qkv: weight+bias<br/>proj: weight+bias"]
        B7_A --> B7_N2["norm2 ✓ frozen"]
        B7_N2 --> B7_M["mlp ◇ 冻结<br/>fc1: weight+bias<br/>fc2: weight+bias"]
    end

    B7 --> B8

    subgraph "Block 8 ─ 深层 Group D — 完全冻结"
        B8[CEBlock]
        B8_N1["norm1 ✓ frozen"] --> B8_A
        B8_A["attn ◇ 冻结<br/>qkv: weight+bias<br/>proj: weight+bias"]
        B8_A --> B8_N2["norm2 ✓ frozen"]
        B8_N2 --> B8_M["mlp ◇ 冻结<br/>fc1: weight+bias<br/>fc2: weight+bias"]
    end

    B8 --> B9

    subgraph "Block 9 ─ CE剪枝层 Group B — 注意力聚焦"
        B9[CEBlock]
        B9_N1["norm1 ✓ frozen"] --> B9_A
        B9_A["attn ★ NS-OPLoRA<br/>qkv: lora_A/B<br/>proj: lora_A/B"]
        B9_A --> B9_CE["CE 剪枝 (keep 70%)"]
        B9_CE --> B9_N2["norm2 ✓ frozen"]
        B9_N2 --> B9_M["mlp ◇ 冻结<br/>fc1: weight+bias<br/>fc2: weight+bias"]
    end

    B9 --> B10

    subgraph "Block 10 ─ 深层 Group D — 完全冻结"
        B10[CEBlock]
        B10_N1["norm1 ✓ frozen"] --> B10_A
        B10_A["attn ◇ 冻结<br/>qkv: weight+bias<br/>proj: weight+bias"]
        B10_A --> B10_N2["norm2 ✓ frozen"]
        B10_N2 --> B10_M["mlp ◇ 冻结<br/>fc1: weight+bias<br/>fc2: weight+bias"]
    end

    B10 --> B11

    subgraph "Block 11 ─ 深层 Group D — 完全冻结"
        B11[CEBlock]
        B11_N1["norm1 ✓ frozen"] --> B11_A
        B11_A["attn ◇ 冻结<br/>qkv: weight+bias<br/>proj: weight+bias"]
        B11_A --> B11_N2["norm2 ✓ frozen"]
        B11_N2 --> B11_M["mlp ◇ 冻结<br/>fc1: weight+bias<br/>fc2: weight+bias"]
    end

    B11 --> NORM["backbone.norm ✓ frozen"]
    NORM --> HEAD["Box Head (Center)<br/>★ 全部可训练"]
```

---

## 图例

| 符号 | 含义 |
|------|------|
| ★ NS-OPLoRA | 注入了 NeuronSelectiveOPLoRALinear，含 lora_A/B + neuron_mask |
| ◇ 冻结 | 保留原始 nn.Linear，weight+bias 不可训练 |
| ✓ frozen | LayerNorm / 无参数的组件，随 backbone 冻结 |

## 分组汇总

```
Block:  0    1    2    3    4    5    6    7    8    9    10   11
        ────────────────         ───         ─────────         ──────
attn:   ★    ★    ★    ★    ◇    ◇    ★    ◇    ◇    ★    ◇    ◇
mlp:    ★    ★    ★    ◇    ★    ★    ◇    ◇    ◇    ◇    ◇    ◇
CE:     -    -    -    ✓    -    -    ✓    -    -    ✓    -    -

Group:  A──────────    B    C────    B    D────────    B    D────────
```

