# OPLoRA 集成到 OSTrack 的说明

本文说明论文 **OPLoRA: Orthogonal Projection LoRA** 的核心思想、原工程行为、代码改动位置与用法。PDF 位于仓库根目录：`D:\code\UAV\OSTrack-main\Xiong和Xie - OPLoRA Orthogonal Projection LoRA Prevents Catastrophic Forgetting During Parameter-Efficient Fine-.pdf`。

---

## 1. 论文里 OPLoRA 在算什么

- **普通 LoRA**：在冻结的 \(W_0\) 上加低秩项 \(\Delta W = BA\)，前向是 \(y = W_0 x + \frac{\alpha}{r} BAx\)。  
  论文指出：\(BA\) 不受约束时，容易和 \(W_0\) 的**大奇异值方向**重叠，相当于动到预训练里「最重要」的子空间，带来**灾难性遗忘**。

- **OPLoRA**：对 \(W_0\) 做 SVD，取前 \(k\) 个左奇异向量 \(U_k\)、右奇异向量 \(V_k\)，定义  
  \(P_L = I - U_k U_k^\top\)，\(P_R = I - V_k V_k^\top\)，令  
  \(\Delta W = P_L B A P_R\)。  
  这样更新严格落在**主导奇异子空间的正交补**里；论文命题 2 说明：**前 \(k\) 个奇异三元组保持不变**，有利于保留预训练知识，同时仍用低秩 \(B,A\) 做任务适配。

**实现上**从不显式构造大矩阵 \(P_L,P_R\)，而是用：

1. \(x_r = x - (x V_k) V_k^\top\)（输入侧投影）
2. \(z = x_r A^\top\)，\(u = z B^\top\)
3. \(y_\text{lora} = u - (u U_k) U_k^\top\)（输出侧投影）  

再乘以 \(\alpha/r\) 加到 \(W_0 x + b\) 上。

---

## 2. 原 OSTrack 代码是什么样

- ViT 主干里，每个 block 的注意力是 `nn.Linear`：`qkv`、`proj`；MLP 里是 timm 的 `Mlp`：`fc1`、`fc2`（见 `lib/models/ostrack/vit.py` 的 `Attention` / `Block`，以及 CE 版本 `lib/models/layers/attn_blocks.py` 与 `lib/models/layers/attn.py`）。
- **没有** LoRA / PEFT：这些层全部是标准 `nn.Linear`，预训练加载后要么整网微调，要么靠 `BACKBONE_MULTIPLIER` 等只调学习率，**没有**「只训低秩适配器、主干权重冻结」的结构。

---

## 3. 代码改动：改了哪里、怎么改的

### 3.1 新文件 `lib/models/layers/oplora.py`

- 新增 **`OPLoRALinear`**：在构造时从现有 `nn.Linear` 拷出 `weight` / `bias`，设为 **`requires_grad=False`**（对应论文里固定的 \(W_0\)）。
- 对 `weight` 做 **`torch.linalg.svd`**，把 **`Uk`、`Vk` 注册为 buffer**（不参与优化，只存子空间）。
- 可训练部分：与 LoRA 一样 **`lora_A`**（\(r \times d_\text{in}\)）、**`lora_B`**（\(d_\text{out} \times r\)），\(B\) 初值为 0，`A` 用 Kaiming（与常见 LoRA 一致）。
- 前向即上面的三步投影 + 缩放 \(\alpha/r\)。
- 新增 **`inject_oplora_into_backbone`**：只在 `backbone` 子树里递归查找子模块名为 `qkv`、`proj`、`fc1`、`fc2` 的 `nn.Linear`，替换为 `OPLoRALinear`（**不动检测头** `box_head`）。

**为什么单独文件、为什么用注入而不是改遍每个 Attention 类**：集中实现 SVD 与投影，避免在 `vit.py` / `attn.py` / `attn_blocks.py` 里复制多份逻辑；注入只动主干里这几类名字，与论文在 LLM 上对 `q_proj` / `v_proj` / … 的选择一致（都是注意力与 MLP 的线性层）。

### 3.2 `lib/models/ostrack/ostrack.py` 中的 `build_ostrack`

- 在 **`load_state_dict` 加载 OSTrack 预训练之后**（若配置了 `OSTrack` 权重），若打开 OPLoRA，则调用 **`inject_oplora_into_backbone(model.backbone, ...)`**。

**为什么放在加载预训练之后**：\(U_k, V_k\) 必须由**最终用作 \(W_0\) 的权重**做 SVD；先 `load_state_dict` 再替换，保证 SVD 对准你真正要保留的那份预训练（与论文设定一致）。

### 3.3 `lib/config/ostrack/config.py`

增加默认块 **`cfg.TRAIN.OPLORA`**：

| 字段 | 含义 |
|------|------|
| `ENABLE` | 是否启用（默认 `False`，行为与原来一致） |
| `RANK` | LoRA 秩 \(r\) |
| `TOP_K` | 投影用的前 \(k\) 个奇异方向（论文里的 \(k\)） |
| `ALPHA` | \(\alpha\)（与常见 LoRA 缩放一致） |
| `TARGETS` | 要替换的线性层名字列表，默认 `qkv`, `proj`, `fc1`, `fc2` |

**为什么用配置**：\(k, r, \alpha\) 和是否启用都应由实验 YAML 控制，而不是写死在模型里。

---

## 4. 实验 YAML 用法示例

在任意 OSTrack 实验 yaml 的 `TRAIN` 下增加（与 `config.py` 里已有键一致即可被 merge）：

```yaml
TRAIN:
  OPLORA:
    ENABLE: true
    RANK: 8
    TOP_K: 16
    ALPHA: 8.0
    TARGETS: ["qkv", "proj", "fc1", "fc2"]
```

- **`TOP_K: 0`**：不做左右投影，退化为**普通 LoRA**（仍冻结 \(W_0\)，只训 \(A,B\)），便于对比。
- **`TOP_K` 较大**（如论文的 128）：保留的奇异子空间更大，适配自由度更小、更偏「保预训练」；**较小**则更偏「可塑」。可按显存与任务试。

---

## 5. 训练行为相对原来的变化

- **原来**：`qkv` / `proj` / `fc1` / `fc2` 的 `weight` 通常随主干一起训练（或整层冻结，取决于别处是否改 `requires_grad`）。
- **现在（`ENABLE: true`）**：这些层的 **`weight` / `bias` 冻结**，只更新 **`lora_A` / `lora_B`**（以及你未冻结的其它参数，例如 `pos_embed_z` / `pos_embed_x` 等）。优化器仍用 `get_optimizer_scheduler` 里「backbone vs 非 backbone」两组学习率；OPLoRA 参数在 backbone 里，会走 **`LR * BACKBONE_MULTIPLIER`**。

若希望 **只训 OPLoRA、其余 backbone 全冻**，需要在训练脚本里额外把除 `lora_` 以外的 backbone 参数设为 `requires_grad=False`（默认未加，以免改变现有微调习惯）。

---

## 6. 涉及文件一览

| 路径 | 作用 |
|------|------|
| `lib/models/layers/oplora.py` | **新增**：`OPLoRALinear` + `inject_oplora_into_backbone` |
| `lib/models/ostrack/ostrack.py` | 构建模型并在预训练加载后 **注入** OPLoRA |
| `lib/config/ostrack/config.py` | **默认配置** `TRAIN.OPLORA.*` |

---

## 7. 恢复训练（resume）时的注意点

构建顺序为：先用标准 `nn.Linear` 加载 yaml 中的 `PRETRAIN_FILE`（若含 OSTrack），再 **注入** OPLoRA 并初始化 `lora_A` / `lora_B`。若训练器再从 checkpoint **resume**，会覆盖整个 `net` 的 state_dict，此时已训练的 LoRA 权重会正确恢复。若把「已带 OPLoRA 的 checkpoint」当作**唯一**的 `PRETRAIN_FILE` 冷启动，当前逻辑会先按 `nn.Linear` 加载能匹配的键，再注入会重置未匹配到的 LoRA 参数；**继续训练**应优先用训练器的 resume 路径而不是依赖该行为。
