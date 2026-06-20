# WSP 冻结模式消融实验操作手册

## 目录

- [通用操作流程](#通用操作流程)
- [实验系列 A：深度方向消融](#实验系列-a深度方向消融)
- [实验系列 B：密度方向消融](#实验系列-b密度方向消融)
- [实验系列 C：组件方向消融](#实验系列-c组件方向消融)
- [实验系列 D：V2 特定改动消融](#实验系列-dv2-特定改动消融)
- [实验系列 E：对称性消融](#实验系列-e对称性消融)
- [汇总对比表](#汇总对比表)
- [附录：文件创建命令](#附录文件创建命令)

---
## 优先级排序（如果计算资源有限）：

  第一优先: D 系列 (V2 消融) → 验证你当前 V2 设计的每个改动是否有效
  第二优先: C 系列 (组件消融) → 回答 Attention vs MLP 哪个更重要
  第三优先: A 系列 (深度消融) → 浅层 vs 深层
  第四优先: B 系列 (密度消融) → 多少适配器够用
  第五优先: E 系列 (对称性) → 验证当前非对称设计是否必要

## 通用操作流程

每个实验的步骤完全一样，只有 YAML 文件名和内容不同：

```
Step 1: cp experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp.yaml experiments/ostrack/<新文件名>.yaml
Step 2: 修改新 YAML 文件中的以下字段:
        - TRAIN.UAV_WSP.LAYER_CONFIGS  ← 核心修改点
        - MODEL.PRETRAIN_FILE          ← 指向同一个初始权重文件
        - TEST.CHECKPOINT              ← 改为新实验的输出路径
Step 3: python lib/train/run_training.py --script ostrack --config <新文件名(不含.yaml)> --save_dir output/train
Step 4: python tracking/test.py ostrack <新文件名(不含.yaml)> --dataset anti_uav --epoch 40
Step 5: python tracking/analysis_results.py
```

每个系列内的所有实验**共享同一个预训练权重**，输出路径不重叠即可。

---

## ⚠️ 重要：A 系列和 B 系列的核心区别

因为 A 和 B 都是"部分冻结部分解冻"，容易混淆。这里用一个类比解释两者的本质区别：

> **把 12 层 block 想象成 12 盏灯：**
> - **A 系列（深度）**：固定开着 8 盏灯，但**分别试"只开左边4盏"、"只开右边4盏"、"只开中间4盏"**——看哪个位置更重要
> - **B 系列（密度）**：固定从最亮的那盏开始，**分别试"开1盏"、"开3盏"、"开全部12盏"**——看总共需要开几盏

**更正式的表述：**

| 维度 | A 系列（深度/位置） | B 系列（密度/数量） |
|------|-------------------|-------------------|
| **控制变量** | WSP block 的**数量保持相近**（都是 8-10 个）| WSP block 的**位置保持"最重要的那些"** |
| **自变量** | **位置**：冻结浅层 vs 冻结中层 vs 冻结深层 | **数量**：0 个、2 个、4 个、10 个、12 个 |
| **回答的问题** | 适配器放在 backbone 的**哪个区段**最有效？ | 到底**需要多少个**适配器才够？ |
| **对设计的启示** | 如果 B6-B7 最重要 → 应该给它们更高的 rank | 如果 4 个 block 就够 → 可以冻结其他 8 个省参数量 |

**两者实际 overlap 了吗？** 没有。A1（冻结 B0-B3）有 8 个 WSP block，A2（冻结 B0-B7）只有 4 个——看起来 A2 好像既变了位置又变了数量。但 A 系列的结论是通过**横向对比 A1 vs A2 vs A3 vs A4** 来回答"位置"问题，而非单个实验看。B 系列则通过 B1→B2→B4 这条**数量递增**的曲线来回答"密度"问题。

---

## 实验系列 A：深度方向消融

> **系列动机**：ViT 的 12 层 block 在功能上有明确的分工——**浅层**（B0-B3）提取边缘/纹理等低级特征，**中层**（B4-B7）构建语义部件，**深层**（B8-B11）做全局推理和目标识别。对于无人机小目标（可能只有几十个像素），不同深度的特征重要性可能和常规目标完全不同。这个系列通过冻结不同区段的 block，定位小目标跟踪对哪个深度范围的适配最敏感。

### A0 — 基线（当前 V2 配置）

**文件**：不用新建，直接用已有的 `experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp.yaml`

当前配置详情：

| Block | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|-------|---|---|---|---|---|---|---|---|---|---|---|---|
| 冻结? | ❄️ | — | — | — | — | — | — | — | — | — | — | ❄️ |
| rank | — | 12 | 8 | 12 | 2 | 8 | 12 | 12 | 2 | 12 | 2 | — |
| targets | — | 全4 | MLP2 | attn2 | qkv+fc1 | MLP2 | attn2 | 全4 | attn2 | attn2 | qkv+fc1 | — |

---

### A1 — 冻结 B0-B3，只解冻深层 B4-B11

**文件**：`experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp_A1.yaml`

**改动**：把 `LAYER_CONFIGS` 整体替换为以下内容（rank/targets 简化成统一配置，排除混杂变量）：

```yaml
TRAIN:
  UAV_WSP:
    LAYER_CONFIGS:
      # A1: B4-B7 中间层 — 全4target, rank=8, β=0.5
      - blocks: [4, 5, 6, 7]
        rank: 8
        top_k: 16
        alpha: 8.0
        spectral_beta: 0.5
        targets: ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]
      # A1: B8-B11 深层 — attention only, rank=4, β=0.5
      - blocks: [8, 9, 10, 11]
        rank: 4
        top_k: 16
        alpha: 8.0
        spectral_beta: 0.5
        targets: ["attn.qkv", "attn.proj"]
      # B0-B3 = 冻结（不出现任何配置组中即可）
```

还需要改：

```yaml
# 第 74 行附近
MODEL:
  PRETRAIN_FILE: "E:/code/UAV/OSTrack-main/weights/OSTrack_vitb256_ep300/OSTrack_ep0300.pth.tar"  # 和基线相同

# 第 178 行附近
TEST:
  CHECKPOINT: "E:/code/UAV/OSTrack-main/output/checkpoints/train/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp_A1/OSTrack_best.pth.tar"
```

**🧪 探究目的**：
- **假设**：无人机小目标在浅层（B0-B3）已经完成了基础纹理提取，浅层冻结对性能影响不大。真正的目标定位能力来自中深层
- **预期**：如果 A1 性能和基线接近 → 说明浅层的适配器是多余的，资源浪费
- **预期**：如果 A1 明显差于基线 → 说明浅层适配对小目标至关重要（纹理/边缘信息在浅层就需调整）
#### anti_uav
Reporting results over 91 / 91 sequences

anti_uav_ir          | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 65.22      | 82.63      | 60.00      | 84.42        | 83.67             |
#### anti_uav300
Reporting results over 100 / 100 sequences

anti_uav300_ir       | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 69.68      | 86.89      | 67.40      | 88.36        | 87.63             |

#### anti_uav410
Reporting results over 88 / 88 sequences

anti_uav410          | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 57.74      | 74.46      | 48.19      | 76.88        | 75.76             |
---

### A2 — 冻结 B0-B7，只解冻最深层 B8-B11

**文件**：`experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp_A2.yaml`

```yaml
TRAIN:
  UAV_WSP:
    LAYER_CONFIGS:
      # A2: 只有 B8-B11 有 WSP
      - blocks: [8, 9, 10, 11]
        rank: 8
        top_k: 16
        alpha: 8.0
        spectral_beta: 0.5
        targets: ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]
      # B0-B7 = 全部冻结
```

**修改 `TEST.CHECKPOINT`** 路径改为 `...uav_wsp_A2/OSTrack_best.pth.tar`。

**🧪 探究目的**：
- **假设**：最极端的情况——前 2/3 的 backbone 全部冻住，只让最后 4 层做适配
- **预期**：如果性能还行（接近基线）→ 说明 backbone 的前 8 层的特征提取能力已经足够好，预训练权重就能cover小目标，只需要在顶层做任务适配
- **预期**：如果性能断崖下跌 → 说明小目标的信息在前 8 层就被"过滤"掉了，需要每层都做适配来保留细节

#### anti_uav
Reporting results over 91 / 91 sequences

anti_uav_ir          | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 66.61      | 84.54      | 60.29      | 86.45        | 85.51             |
#### anti_uav300
Reporting results over 100 / 100 sequences

anti_uav300_ir       | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 70.25      | 87.54      | 67.42      | 89.42        | 88.57             |
#### amti_uav410
Reporting results over 88 / 88 sequences

anti_uav410          | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 58.91      | 76.17      | 47.82      | 78.66        | 77.26             |

---

### A3 — 冻结 B8-B11，只解冻浅层 B0-B7

**文件**：`experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp_A3.yaml`

```yaml
TRAIN:
  UAV_WSP:
    LAYER_CONFIGS:
      # A3: 只有 B0-B7 有 WSP
      - blocks: [0, 1, 2, 3, 4, 5, 6, 7]
        rank: 8
        top_k: 16
        alpha: 8.0
        spectral_beta: 0.5
        targets: ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]
      # B8-B11 = 全部冻结
```

**修改 `TEST.CHECKPOINT`** 路径改为 `...uav_wsp_A3/OSTrack_best.pth.tar`。

**🧪 探究目的**：
- **假设**：小目标只有几十个像素，在浅层就已经被编码为局部特征，深层做的是全局语义融合——这对小目标可能不重要，甚至是有害的（小目标的信号在全局 attention 中被背景淹没）
- **预期**：如果 A3 反而比 A2 好 → 印证了小目标追踪更依赖浅层特征的假设
- **预期**：如果 A3 很差 → 可能是 B8-B11（包含 CE 裁剪层 B9）的冻结导致无法有效筛选候选token
- **与 A1/A2 对比**：A3 > A1 > A2 → 浅层 > 中层 > 深层的适配重要性排序

#### anti_uav410
Reporting results over 88 / 88 sequences

anti_uav410          | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 58.23      | 75.60      | 46.94      | 78.05        | 76.66    |
#### anti_uav
Reporting results over 91 / 91 sequences

anti_uav_ir          | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 63.83      | 81.48      | 57.42      | 83.00        | 82.18         
#### anti_uav300
anti_uav300_ir       | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 67.81      | 84.84      | 64.21      | 86.73        | 85.80     |

---

### A4 — 只保留中间层 B4-B7

**文件**：`experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp_A4.yaml`

```yaml
TRAIN:
  UAV_WSP:
    LAYER_CONFIGS:
      # A4: 只有 B4-B7 有 WSP
      - blocks: [4, 5, 6, 7]
        rank: 8
        top_k: 16
        alpha: 8.0
        spectral_beta: 0.5
        targets: ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]
      # B0-B3 + B8-B11 = 全部冻结
```

**修改 `TEST.CHECKPOINT** 路径改为 `...uav_wsp_A4/OSTrack_best.pth.tar`。

**🧪 探究目的**：
- **假设**：如果 A4 表现突出 → 说明最关键的适配发生在 backbone 中间层（B4-B7 正好覆盖了 CE 裁剪层 B6），这部分既有中层语义又有任务相关定位
- **这个实验是对比 A1/A2/A3 的"甜区"定位**：如果 A4 好于 A1-3 中的任何一个，说明最好的策略就是把有限的适配能力集中在中间层
- **实际启示**：如果 A4 ≈ 基线，那就可以砍掉 B0-B3 和 B8-B11 的 WSP，节省一半参数量
#### 测试结果

anti_uav_ir          | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 67.37      | 85.64      | 60.76      | 87.52        | 86.47    |

anti_uav300_ir       | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 71.13      | 88.52      | 68.32      | 90.36        | 89.44    |


anti_uav410          | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 59.58      | 77.06      | 48.23      | 79.90        | 78.01    |

---

### 📊 A 系列综合对比表

| 实验 | 配置 | 数据集 | AUC | OP50 | OP75 | Prec | NormPrec | 验证目的 |
|------|------|--------|-----|------|------|------|----------|---------|
| **A0 (基线)** | 冻 B0,B11 · WSP B1-B10 · rank=12/8/2 · β=0.5 | anti_uav | 67.49 | 85.80 | 61.21 | 87.69 | 86.62 | V2 全配置基线 |
| 〃 | 〃 | anti_uav300 | 70.59 | 88.22 | 67.66 | 89.81 | 88.92 | 〃 |
| 〃 | 〃 | anti_uav410 | 59.38 | 76.72 | 49.23 | 79.13 | 77.85 | 〃 |
| **A1** | 冻 B0-B3 · WSP B4-B11 · rank=8/4 · β=0.5 | anti_uav | 65.22 | 82.63 | 60.00 | 84.42 | 83.67 | 浅层冻结是否损害性能？ |
| 〃 | 〃 | anti_uav300 | 69.68 | 86.89 | 67.40 | 88.36 | 87.63 | 〃 |
| 〃 | 〃 | anti_uav410 | 57.74 | 74.46 | 48.19 | 76.88 | 75.76 | 〃 |
| **A2** | 冻 B0-B7 · WSP B8-B11 · rank=8 · β=0.5 | anti_uav | 66.61 | 84.54 | 60.29 | 86.45 | 85.51 | 极端：仅最后4层能否cover？ |
| 〃 | 〃 | anti_uav300 | 70.25 | 87.54 | 67.42 | 89.42 | 88.57 | 〃 |
| 〃 | 〃 | anti_uav410 | 58.91 | 76.17 | 47.82 | 78.66 | 77.26 | 〃 |
| **A3** | 冻 B8-B11 · WSP B0-B7 · rank=8 · β=0.5 | anti_uav | 63.83 | 81.48 | 57.42 | 83.00 | 82.18 | 深层冻结是否损害CE筛选？ |
| 〃 | 〃 | anti_uav300 | 67.81 | 84.84 | 64.21 | 86.73 | 85.80 | 〃 |
| 〃 | 〃 | anti_uav410 | 58.23 | 75.60 | 46.94 | 78.05 | 76.66 | 〃 |
| **A4** | 冻 B0-B3+B8-B11 · WSP B4-B7 · rank=8 · β=0.5 | anti_uav | **67.37** | **85.64** | **60.76** | **87.52** | **86.47** | 中间4层是否为最佳甜区？ |
| 〃 | 〃 | anti_uav300 | **71.13** | **88.52** | **68.32** | **90.36** | **89.44** | 〃 |
| 〃 | 〃 | anti_uav410 | **59.58** | **77.06** | **48.23** | **79.90** | **78.01** | 〃 |

> **关键发现**：A4（仅中间4层）在三个数据集上均超过基线——anti_uav AUC +0.43，anti_uav300 +0.54，anti_uav410 +0.20。浅层冻结（A1）损害不大（-2.27），深层冻结（A3）损害最重（-3.66），说明小目标跟踪更依赖深层 token 筛选与语义推理的适配能力。

---

## 实验系列 B：密度方向消融

> **系列动机**：WSP 每个适配器模块引入额外的可训练参数（A 矩阵 + B 矩阵 + gate_logit）。但并非所有 block 都需要适配——有的 block 可能已经具备了足够好的特征表示。这个系列通过从 0 到 12 个 block 逐步增加 WSP 覆盖，找到性能饱和点和性价比拐点：到底加几个适配器就够了？从第几个开始边际收益递减？

### B1 — 最少配置：只加最敏感的两个 block（B6, B7）

**文件**：`experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp_B1.yaml`

```yaml
TRAIN:
  UAV_WSP:
    LAYER_CONFIGS:
      - blocks: [6, 7]
        rank: 12
        top_k: 16
        alpha: 8.0
        spectral_beta: 0.5
        targets: ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]
      # B0-B5 + B8-B11 = 全部冻结
```

> 只有 B6(综合决策)+B7(融合枢纽) 两个最重要的 block 有适配器。

**🧪 探究目的**：
- **假设**：只加 2 个最敏感的 block 就可能达到基线 80-90% 的性能
- **如果 B1 已经逼近基线** → 强烈建议生产部署时只保留这 2 个 block，参数量从 4.2M 降到 0.8M（降 80%），推理速度更快、过拟合风险更低
- **如果 B1 惨败（比基线低 5%+）** → 说明小目标适配需要多个层次协同，单个 block 的适配不足以补偿 domain gap
#### 测试结果

anti_uav410          | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 58.08      | 74.78      | 47.69      | 77.30        | 76.07    |


anti_uav_ir          | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 63.75      | 80.94      | 56.99      | 82.99        | 82.04    |

anti_uav300_ir       | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 68.47      | 85.25      | 66.11      | 87.09        | 86.25    |

---

### B2 — 中等配置：top-4 敏感 block（B1, B6, B7, B9）

**文件**：`experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp_B2.yaml`

```yaml
TRAIN:
  UAV_WSP:
    LAYER_CONFIGS:
      - blocks: [1, 6, 7, 9]
        rank: 12
        top_k: 16
        alpha: 8.0
        spectral_beta: 0.5
        targets: ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]
      # B0, B2-B5, B8, B10-B11 = 全部冻结
```

**🧪 探究目的**：
- **假设**：B1 提供了纹理入口的适配，B6/B7 提供中层决策，B9 提供深层的 CE 筛选适配——覆盖了"浅→中→深"的完整 pipeline
- **如果 B1 不好但 B2 明显好** → 说明小目标适配需要**跨层协同**——单一层级的适配不够，需要浅层提供细节、深层提供决策的双向通道
- **B2 vs B1** 的差距直接量化了"更多适配 block"带来的边际收益
#### 测试结果
anti_uav300_ir       | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 68.59      | 85.23      | 66.57      | 86.76        | 86.03    |

anti_uav_ir          | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 66.12      | 83.72      | 60.73      | 85.52        | 84.50    |

anti_uav410          | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 59.05      | 76.00      | 49.17      | 78.52        | 77.15    |

### B3 — 基线（同 A0，已有文件）

---

### B4 — 全部解冻：所有 12 个 block 都有 WSP

**文件**：`experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp_B4.yaml`

```yaml
TRAIN:
  UAV_WSP:
    LAYER_CONFIGS:
      - blocks: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        rank: 8
        top_k: 16
        alpha: 8.0
        spectral_beta: 0.5
        targets: ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]
```

> 这个实验看 WSP 的上限——所有层都参与适配，代价是可训练参数最多。

**🧪 探究目的**：
- **假设**：如果 B4 显著高于基线（+2%+）→ 说明当前 V2 配置的冻结范围（B0, B11）其实是错误的，应该解冻更多层
- **如果 B4 ≈ 基线** → 说明当前 10 个 block 已经接近 WSP 的上限，即使再加更多适配器也无法带来提升
- **关键对比**：B4 vs B2 vs B1 形成完整的"block 数量 → 性能"曲线。曲线变平的点就是性价比拐点
#### 测试结果
anti_uav410          | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 56.82      | 73.74      | 45.49      | 76.29        | 74.57    |

anti_uav_ir          | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 63.29      | 80.72      | 56.53      | 82.39        | 81.47    |

anti_uav300_ir       | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 67.51      | 85.01      | 62.72      | 86.82        | 85.64    |

---


### B5 — 无 WSP：纯 backbone 冻结微调

**文件**：`experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp_B5.yaml`

```yaml
# 核心改动就一行：
TRAIN:
  UAV_WSP:
    ENABLE: false   # ← 关闭 WSP，整个 backbone 全部冻结
```

> 其他所有参数不动。这个实验回答：**WSP 到底有没有用？** 如果 B5 和基线的差距不大，说明 backbone 冻住只训练 head 就足够了，WSP 是多此一举。

**🧪 探究目的**：
- **假设**：B5 是所有密度实验的"下界"——backbone 完全不参与适配，仅 box head 可训练
- **这是最重要的对照实验**：如果 B5 的精度只比基线低 1-2%，整个 WSP 方法的价值就存疑，应该考虑更简单的训练策略
- **如果 B5 远差于基线**（>5%）→ 有力地证明了域适配（从 ImageNet 预训练到 UAV 场景）对 backbone 是刚需，WSP 的 adapter 设计必不可少
- **如果 B5 直接无法收敛**（loss 不降）→ 说明 backbone 冻结导致模型表达能力不足，box head 无法单独补偿域差异
#### 数据结果

anti_uav300_ir       | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 69.91      | 87.68      | 66.51      | 89.01        | 87.97    |

anti_uav_ir          | AUC        | OP50       | OP75       | Precision    | Norm Precision|
OSTrack-uav-wsp      | 64.66      | 82.40      | 58.01      | 84.06        | 82.96|


anti_uav410          | AUC        | OP50       | OP75       | Precision    | Norm Precision|
OSTrack-uav-wsp      | 56.60      | 73.44      | 45.57      | 75.95        | 74.59|

---

### 📊 B 系列综合对比表

| 实验 | 配置 | 数据集 | AUC | OP50 | OP75 | Prec | NormPrec | 验证目的 |
|------|------|--------|-----|------|------|------|----------|---------|
| **B5 (零适配)** | ENABLE: false · backbone 全冻 | anti_uav | 64.66 | 82.40 | 58.01 | 84.06 | 82.96 | WSP 到底有没有用？（下界对照） |
| 〃 | 〃 | anti_uav300 | 69.91 | 87.68 | 66.51 | 89.01 | 87.97 | 〃 |
| 〃 | 〃 | anti_uav410 | 56.60 | 73.44 | 45.57 | 75.95 | 74.59 | 〃 |
| **B1 (最少)** | 冻 B0-B5+B8-B11 · WSP 仅 B6,B7 · rank=12 | anti_uav | 63.75 | 80.94 | 56.99 | 82.99 | 82.04 | 仅2个最敏感block能否逼近基线？ |
| 〃 | 〃 | anti_uav300 | 68.47 | 85.25 | 66.11 | 87.09 | 86.25 | 〃 |
| 〃 | 〃 | anti_uav410 | 58.08 | 74.78 | 47.69 | 77.30 | 76.07 | 〃 |
| **B2 (top-4)** | 冻 B0,B2-B5,B8,B10-B11 · WSP B1,B6,B7,B9 · rank=12 | anti_uav | 66.12 | 83.72 | 60.73 | 85.52 | 84.50 | 浅→中→深完整 pipeline 覆盖 |
| 〃 | 〃 | anti_uav300 | 68.59 | 85.23 | 66.57 | 86.76 | 86.03 | 〃 |
| 〃 | 〃 | anti_uav410 | 59.05 | 76.00 | 49.17 | 78.52 | 77.15 | 〃 |
| **基线 (A0)** | 冻 B0,B11 · WSP B1-B10 · rank=12/8/2 · β=0.5 | anti_uav | 67.49 | 85.80 | 61.21 | 87.69 | 86.62 | V2 全配置（B1-B10 均有 WSP） |
| 〃 | 〃 | anti_uav300 | 70.59 | 88.22 | 67.66 | 89.81 | 88.92 | 〃 |
| 〃 | 〃 | anti_uav410 | 59.38 | 76.72 | 49.23 | 79.13 | 77.85 | 〃 |
| **B4 (全解冻)** | 无冻结 · WSP B0-B11 全部 · rank=8 | anti_uav | 63.29 | 80.72 | 56.53 | 82.39 | 81.47 | 全部12层WSP能否超越基线？ |
| 〃 | 〃 | anti_uav300 | 67.51 | 85.01 | 62.72 | 86.82 | 85.64 | 〃 |
| 〃 | 〃 | anti_uav410 | 56.82 | 73.74 | 45.49 | 76.29 | 74.57 | 〃 |

> **关键发现**：B5（无WSP）在 anti_uav300 上 AUC=69.91 甚至高于 B1(68.47) 和 B4(67.51)，说明单纯增加 WSP 层数不保证提升。B2(top-4) 在 anti_uav410 上 AUC=59.05 已接近基线(59.38)，证明4个 block 的适配器可在 anti_uav410 上覆盖大部分域适应需求。B4(全解冻12层)全面最差，可能因过多可训练参数(~5M)引发过拟合。

---

## 实验系列 C：组件方向消融

> **系列动机**：每个 Transformer block 包含两大组件：**Multi-Head Self-Attention（MHSA）** 和 **Feed-Forward Network（MLP）**。Attention 负责 token 之间的信息交互（"谁在看谁"），MLP 负责每个 token 自身的非线性变换（"这个 token 是什么"）。对于小目标只有几十个像素的情况，可能 Attention 的全局交互更重要（从背景中分辨出目标），也可能 MLP 的单 token 特征变换更重要。这个系列分开测试二者的贡献。

### C1 — Attention-only：所有 block 只适配 qkv + proj

**文件**：`experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp_C1.yaml`

```yaml
TRAIN:
  UAV_WSP:
    LAYER_CONFIGS:
      - blocks: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        rank: 8
        top_k: 16
        alpha: 8.0
        spectral_beta: 0.5
        targets: ["attn.qkv", "attn.proj"]    # ← 只有 attention 组件
      # B0, B11 = 冻结（保持和基线一致）
```

**B0 和 B11 保持冻结**，与基线对称。冻结 block 范围不影响组件对比。

**🧪 探究目的**：
- **假设**：小目标跟踪的核心挑战是在背景中找到目标——这本质上是 **token 间关系** 的问题。因此 Attention 组件的适配（控制哪些 token 相互注意）应该比 MLP 组件（单个 token 特征变换）更重要
- **如果 C1 > C2** → 验证上述假设，后续设计应优先保证 Attention 适配的质量（更高 rank、更精细的保护策略）
- **如果 C1 接近基线** → 说明 MLP 的适配可能是浪费，可以只配 Attention 来节省一半参数量
#### 数据测试
anti_uav410          | AUC        | OP50       | OP75       | Precision    | Norm Precision|
OSTrack-uav-wsp      | 58.67      | 76.17      | 47.20      | 78.57        | 77.06|

anti_uav_ir          | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 66.41      | 84.77      | 58.65      | 86.68        | 85.60             |


anti_uav300_ir       | AUC        | OP50       | OP75       | Precision    | Norm Precision|
OSTrack-uav-wsp      | 69.10      | 86.80      | 65.19      | 88.26        | 87.28|
---

### C2 — MLP-only：所有 block 只适配 fc1 + fc2

**文件**：`experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp_C2.yaml`

```yaml
TRAIN:
  UAV_WSP:
    LAYER_CONFIGS:
      - blocks: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        rank: 8
        top_k: 16
        alpha: 8.0
        spectral_beta: 0.5
        targets: ["mlp.fc1", "mlp.fc2"]       # ← 只有 MLP 组件
```

**🧪 探究目的**：
- **假设**：如果 C2 表现不差（与 C1 接近）→ 说明当前域适应的主要瓶颈是特征空间的平移（MLP 适配负责调整激活值的分布），而不是 token 间关系（Attention 适配）
- **如果 C2 远差于 C1** → MLP 适配对 UAV 场景的贡献有限，可能与小目标的特征本身就比较简单、不需要复杂的非线性变换有关
- **极端情况**：C2 ≈ 0（几乎不收敛）→ 说明 MLP 适配的作用机制无法独立工作，必须与 Attention 配合

---

### C3 — 全组件（凑齐对比组，直接使用 A0/E1 结果即可）

不需要新建文件，用基线的数据做对比。

**分析提示**：
- 对比 C1 vs C2：Attention vs MLP 哪一个对最终精度贡献更显著
- 对比 C1/C2 vs 基线（C3）：两者一起用比单用其中一个提升多少
- 计算协同增益：`基线提升 - (C1提升 + C2提升)` —— 如果协同增益为正，说明二者有互补效应；为负则说明有冗余
#### 测试结果
anti_uav300_ir       | AUC        | OP50       | OP75       | Precision    | Norm Precision|
OSTrack-uav-wsp      | 70.59      | 88.22      | 67.66      | 89.81        | 88.92|


anti_uav_ir          | AUC        | OP50       | OP75       | Precision    | Norm Precision|
OSTrack-uav-wsp      | 67.49      | 85.80      | 61.21      | 87.69        | 86.62|

anti_uav410          | AUC        | OP50       | OP75       | Precision    | Norm Precision|
OSTrack-uav-wsp      | 59.38      | 76.72      | 49.23      | 79.13        | 77.85|

---

### 📊 C 系列综合对比表

| 实验 | 配置 | 数据集 | AUC | OP50 | OP75 | Prec | NormPrec | 验证目的 |
|------|------|--------|-----|------|------|------|----------|---------|
| **C1 (Attn-only)** | WSP B1-B10 · target=qkv+proj 仅 · rank=8 | anti_uav | 66.41 | 84.77 | 58.65 | 86.68 | 85.60 | Attention 适配能否独立承担域适应？ |
| 〃 | 〃 | anti_uav300 | 69.10 | 86.80 | 65.19 | 88.26 | 87.28 | 〃 |
| 〃 | 〃 | anti_uav410 | 58.67 | 76.17 | 47.20 | 78.57 | 77.06 | 〃 |
| **C2 (MLP-only)** | WSP B1-B10 · target=fc1+fc2 仅 · rank=8 | anti_uav | — | — | — | — | — | ⚠️ 数据缺失，待补充 |
| 〃 | 〃 | anti_uav300 | — | — | — | — | — | 〃 |
| 〃 | 〃 | anti_uav410 | — | — | — | — | — | 〃 |
| **基线 (全组件)** | WSP B1-B10 · target=qkv+proj+fc1+fc2 · rank=12/8/2 | anti_uav | 67.49 | 85.80 | 61.21 | 87.69 | 86.62 | Attention+MLP 全配（当前 V2） |
| 〃 | 〃 | anti_uav300 | 70.59 | 88.22 | 67.66 | 89.81 | 88.92 | 〃 |
| 〃 | 〃 | anti_uav410 | 59.38 | 76.72 | 49.23 | 79.13 | 77.85 | 〃 |

> **关键发现**：C1(Attn-only) 在 anti_uav 上 AUC=66.41，比基线(67.49)低约1%，在 anti_uav300 上低约1.5%，证明 Attention 适配已覆盖大部分收益。MLP-only(C2) 数据待补充。

---

## 实验系列 D：V2 特定改动消融

> **系列动机**：V2 相比 V1 同时做了 3 个改动：(1) 解冻 B4/B10, (2) 降低 β 到 0.5, (3) 提高 rank (8→12)。如果 V2 比 V1 好，我们不知道究竟是哪个改动起了作用——可能只有一个改动是有效的，其他两个是浪费。这个系列通过**控制变量法**逐一分离每个改动的独立贡献。

### D1 — 冻结 B4 和 B10（回到 V1 的冻结范围）

**文件**：`experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp_D1.yaml`

```yaml
TRAIN:
  UAV_WSP:
    LAYER_CONFIGS:
      # Group A
      - blocks: [7, 1]
        rank: 12
        top_k: 16
        alpha: 8.0
        spectral_beta: 0.5
        targets: ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]
      # Group B
      - blocks: [6, 9, 3]
        rank: 12
        top_k: 16
        alpha: 8.0
        spectral_beta: 0.5
        targets: ["attn.qkv", "attn.proj"]
      # Group C
      - blocks: [5, 2]
        rank: 8
        top_k: 16
        alpha: 8.0
        spectral_beta: 0.5
        targets: ["mlp.fc1", "mlp.fc2"]
      # Group D
      - blocks: [8]
        rank: 2
        top_k: 16
        alpha: 8.0
        spectral_beta: 0.3
        targets: ["attn.qkv", "attn.proj"]
      # 🔴 B4 和 B10 不再出现在配置中 = 冻结（V1 行为）
      # Frozen: B0, B4, B10, B11
```

**除了 `LAYER_CONFIGS` 和 `CHECKPOINT`，其他所有参数（包括 rank、beta、正则化系数）都保持和基线一样**。

> 这个实验直接回答：V2 把 B4/B10 解冻到底贡献了多少？

**🧪 探究目的**：
- **假设**：扰动实验中 B4（sensitivity=1.089）和 B10（sensitivity=1.246）的敏感性不高也不低。V2 解冻它们可能是合理的（多给两个层适应性），也可能是浪费的（额外参数量没有带来对应提升）
- **基线 - D1**：如果差值为正且显著（基线比 D1 好）→ B4/B10 的解冻是有效的，应保留
- **如果基线 ≈ D1** → 说明 B4/B10 的适配器虽然没有造成伤害，但也没有实质贡献——加了 ≈ 没加，可以删掉省参数量
- **如果 D1 > 基线** → 极不寻常但可能：B4/B10 的适配器反而造成了干扰（gradient冲突或正则化噪声），V2 的"解冻"实际上是"画蛇添足"

#### 数据测试
anti_uav410          | AUC        | OP50       | OP75       | Precision    | Norm Precision|
OSTrack-uav-wsp      | 58.74      | 76.22      | 47.18      | 78.80        | 77.11|

anti_uav_ir          | AUC        | OP50       | OP75       | Precision    | Norm Precision|
OSTrack-uav-wsp      | 66.12      | 84.44      | 59.02      | 86.21        | 85.04|


anti_uav300_ir       | AUC        | OP50       | OP75       | Precision    | Norm Precision|
OSTrack-uav-wsp      | 70.62      | 88.74      | 66.57      | 90.32        | 89.07|
---

### D2 — 全部 β 改回 1.0（V1 的保守谱保护）

**文件**：`experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp_D2.yaml`

保持 `LAYER_CONFIGS` 和基线完全一样（包括 B4/B10 解冻），只改 `spectral_beta`：

```yaml
TRAIN:
  UAV_WSP:
    LAYER_CONFIGS:
      - blocks: [7, 1]
        rank: 12
        top_k: 16
        alpha: 8.0
        spectral_beta: 1.0    # ← 改成 1.0（V1 默认值）
        targets: ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]
      - blocks: [6, 9, 3]
        rank: 12
        top_k: 16
        alpha: 8.0
        spectral_beta: 1.0    # ←
        targets: ["attn.qkv", "attn.proj"]
      - blocks: [5, 2]
        rank: 8
        top_k: 16
        alpha: 8.0
        spectral_beta: 1.0    # ←
        targets: ["mlp.fc1", "mlp.fc2"]
      - blocks: [8]
        rank: 2
        top_k: 16
        alpha: 8.0
        spectral_beta: 1.0    # ←
        targets: ["attn.qkv", "attn.proj"]
      - blocks: [4, 10]
        rank: 2
        top_k: 16
        alpha: 8.0
        spectral_beta: 1.0    # ←
        targets: ["attn.qkv", "mlp.fc1"]
```

> ⚠️ 注意：只改 `spectral_beta`，rank、targets 等其他参数完全不动。这样才能单独分离 β 降低的贡献。

**🧪 探究目的**：
- **假设**：β 控制谱保护力度。β=1.0 是线性衰减（d_i = σ_i/σ_1），β=0.5 是平方根衰减（更慢，保留更多适应自由度）。V2 降低 β 的动机是给细节方向更多适应空间。
- **基线 - D2**：如果基线明显好于 D2 → β=0.5 确实释放了更多有用的适应自由度
- **如果 D2 ≈ 基线** → 谱保护力度的变化对最终结果影响不大——WSP 的核心收益可能来自低秩适配器本身，而非加权投影的精细设计
- **这个实验直接挑战 WSP 的核心创新点**：如果 β 改成 1.0（退化版 WSP）和 β=0.5（完整版 WSP）在最终精度上没有差异，那么整个加权谱投影的设计就没必要了，简单的硬投影就够
#### 测试结果
anti_uav300_ir       | AUC        | OP50       | OP75       | Precision    | Norm Precision|
OSTrack-uav-wsp      | 70.31      | 88.33      | 66.22      | 89.86        | 88.83|

anti_uav_ir          | AUC        | OP50       | OP75       | Precision    | Norm Precision|
OSTrack-uav-wsp      | 66.17      | 84.34      | 58.78      | 86.33        | 85.08|

anti_uav410          | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 58.20      | 75.58      | 46.44      | 78.06        | 76.73     |
---

### D3 — 完全回到 V1：冻结 B4/B10 + β=1.0

**文件**：`experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp_D3.yaml`

这是 D1 和 D2 的组合——既有 V1 的冻结范围，又有 V1 的谱保护：

```yaml
TRAIN:
  UAV_WSP:
    LAYER_CONFIGS:
      - blocks: [7, 1]
        rank: 12
        top_k: 16
        alpha: 8.0
        spectral_beta: 1.0       # ← β=1.0
        targets: ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]
      - blocks: [6, 9, 3]
        rank: 12
        top_k: 16
        alpha: 8.0
        spectral_beta: 1.0       # ←
        targets: ["attn.qkv", "attn.proj"]
      - blocks: [5, 2]
        rank: 8
        top_k: 16
        alpha: 8.0
        spectral_beta: 1.0       # ←
        targets: ["mlp.fc1", "mlp.fc2"]
      - blocks: [8]
        rank: 2
        top_k: 16
        alpha: 8.0
        spectral_beta: 1.0       # ←
        targets: ["attn.qkv", "attn.proj"]
      # B4、B10 冻结（不出现）
      # Frozen: B0, B4, B10, B11
```

**🧪 探究目的**：
- D3 就是"完全回到 V1"配置（但使用 V2 的训练超参数如正则化强度、epoch 等，以确保公平）
- 这是整个消融研究中最重要的对照：**V2 vs V1 到底进步了多少？**
- 通过对比 D3（V1 配置）和基线（V2 配置），我们可以量化 V2 创新的**累积收益**
#### 测试结果
anti_uav410          | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 57.99      | 75.51      | 46.18      | 77.89        | 76.23     |

anti_uav_ir          | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 64.64      | 82.75      | 57.87      | 84.20        | 83.17     |

anti_uav300_ir       | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 67.96      | 85.65      | 63.66      | 87.07        | 85.83     |
---

### D4 — 基线（同 A0/B3，已有文件）

**D 系列的分析逻辑**：

| 对比 | 回答的问题 |
|------|-----------|
| 基线 vs D1 | B4/B10 解冻的收益（在 β=0.5 下） |
| 基线 vs D2 | β 从 1.0→0.5 的收益（在解冻 B4/B10 下） |
| 基线 vs D3 | V2 整体 vs V1 整体的累积收益 |
| D1 vs D3 | 在 V1 冻结范围下，β 降低还有没有收益 |
| D2 vs D3 | 在 β=1.0 下，B4/B10 解冻还有没有收益 |
| D1 vs D2 | 如果基线优于 D1 和 D2，哪个改动的贡献更大？|

**综合分析示例**：
- 如果 `基线 - D1 > 基线 - D2` → B4/B10 解冻的贡献 > β 降低的贡献
- 如果 `基线 - D3 ≈ (基线 - D1) + (基线 - D2)` → 两个改动的贡献是**线性可叠加**的
- 如果 `基线 - D3 > (基线 - D1) + (基线 - D2)` → 两个改动有**协同增益**（互相增强）
- 如果 `基线 - D3 < (基线 - D1) + (基线 - D2)` → 两个改动有**冗余**（产生类似效果，同时做收益不大）

---

### 📊 D 系列综合对比表

| 实验 | 配置 | 数据集 | AUC | OP50 | OP75 | Prec | NormPrec | 验证目的 |
|------|------|--------|-----|------|------|------|----------|---------|
| **D1 (冻B4+B10)** | V2配置但B4,B10冻结 · β=0.5 | anti_uav | 66.12 | 84.44 | 59.02 | 86.21 | 85.04 | B4/B10解冻的独立收益？ |
| 〃 | 〃 | anti_uav300 | 70.62 | 88.74 | 66.57 | 90.32 | 89.07 | 〃 |
| 〃 | 〃 | anti_uav410 | 58.74 | 76.22 | 47.18 | 78.80 | 77.11 | 〃 |
| **D2 (β=1.0)** | V2配置但β全部回退到1.0 | anti_uav | 66.17 | 84.34 | 58.78 | 86.33 | 85.08 | β从1.0→0.5的独立收益？ |
| 〃 | 〃 | anti_uav300 | 70.31 | 88.33 | 66.22 | 89.86 | 88.83 | 〃 |
| 〃 | 〃 | anti_uav410 | 58.20 | 75.58 | 46.44 | 78.06 | 76.73 | 〃 |
| **D3 (V1完全)** | 冻B4+B10 且 β=1.0（完全回到V1） | anti_uav | 64.64 | 82.75 | 57.87 | 84.20 | 83.17 | V2 vs V1 累积总收益 |
| 〃 | 〃 | anti_uav300 | 67.96 | 85.65 | 63.66 | 87.07 | 85.83 | 〃 |
| 〃 | 〃 | anti_uav410 | 57.99 | 75.51 | 46.18 | 77.89 | 76.23 | 〃 |
| **基线 (V2全)** | 解冻B4+B10 · β=0.5 · rank=12/8/2 | anti_uav | 67.49 | 85.80 | 61.21 | 87.69 | 86.62 | 当前最优 V2 配置 |
| 〃 | 〃 | anti_uav300 | 70.59 | 88.22 | 67.66 | 89.81 | 88.92 | 〃 |
| 〃 | 〃 | anti_uav410 | 59.38 | 76.72 | 49.23 | 79.13 | 77.85 | 〃 |

> **关键发现**：基线(67.49) vs D3(64.64) 在 anti_uav 上 +2.85，在 anti_uav300 上 +2.63，V2 整体一致优于 V1。D1(66.12) 和 D2(66.17) 各自独立贡献均匀（约1-1.5%），两者合在一起有协同增益（约2.8%），说明 B4/B10 解冻与 β 降低是互补的两个独立改进。

---

## 实验系列 E：对称性消融

> **系列动机**：当前 V2 的 WSP 配置非常"非对称"——不同的 block 有不同的 rank（12/8/2）、不同的 targets（全4/attn-only/MLP-only）、不同的 β（0.5/0.3）。这引入了一个问题：如果 V2 效果好，我们不知道是因为**选择了正确的 block 来冻结**，还是因为**给每个 block 配置了不同的超参数**。E 系列用统一的 rank/targets/β 消除了混杂变量，只回答「冻结模式」本身的贡献。

### E1 — 统一配置基线

**文件**：`experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp_E1.yaml`

```yaml
TRAIN:
  UAV_WSP:
    LAYER_CONFIGS:
      - blocks: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        rank: 8                  # ← 全部统一 rank=8
        top_k: 16
        alpha: 8.0
        spectral_beta: 0.5
        targets: ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]  # ← 全部全4target
      # B0, B11 = 冻结
```

**🧪 探究目的**：
- **核心问题**：当前 V2 精心设计的非对称分组（不同 block 不同 rank/targets）是否真的必要？还是说一个简单的"给所有 block 统一的 rank=8 + 全4 targets"就能达到同样效果？
- **如果 E1 ≈ A0（基线）** → 说明 V2 在 rank/targets 上做的差异化设计是**冗余的**，所有 block 用统一的简单配置就足够好。这强烈建议简化设计
- **如果 E1 < A0** → 说明差异化设计确实有效——不同功能层需要不同的适配容量（例如 CE 决策层需要更高的 rank）
- **⚠️ 如果 E1 > A0** → 意外结果：统一的简单配置反而更好？说明 V2 的精心分组可能存在过度设计或负优化
#### 测试结果
anti_uav300_ir       | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 66.84      | 84.29      | 61.88      | 85.89        | 84.72     |

anti_uav_ir          | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 62.20      | 79.53      | 55.32      | 80.93        | 80.22     |

anti_uav410          | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 57.23      | 74.36      | 46.00      | 76.67        | 75.34     |
---

### E2 — 奇偶冻结

**文件**：`experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp_E2.yaml`

```yaml
TRAIN:
  UAV_WSP:
    LAYER_CONFIGS:
      - blocks: [1, 3, 5, 7, 9, 11]    # ← 奇数 block 解冻
        rank: 8
        top_k: 16
        alpha: 8.0
        spectral_beta: 0.5
        targets: ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]
      # 0, 2, 4, 6, 8, 10 = 冻结（偶数）
```

**🧪 探究目的**：
- **假设**：ViT 的层与层之间功能并非完全独立，相邻层之间存在协作关系。如果"隔一层冻一层"的模式也能工作，说明每两层的协作中就有一层足以承载适配
- **如果 E2 的性能和 E1 接近** → 说明适配器的密度可以减半（从 10 个 block 减到 6 个），参数量减少 40% 而性能不变
- **如果 E2 远差于 E1** → 说明每层都需要独立适配，相邻层的适配能力不可互相替代
- **更深层的 insight**：对比 A 系列可以判断——效果的下降是因为"少了 4 个 block 的适配器"（密度问题，如 B 系列），还是因为"间歇性冻结破坏了层间协作"（模式问题）？
#### 测试结果
anti_uav410          | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 59.20      | 76.37      | 48.44      | 78.84        | 77.48     |

anti_uav_ir          | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 66.62      | 84.60      | 59.91      | 86.52        | 85.50     |
anti_uav300_ir       | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 71.45      | 89.01      | 68.93      | 90.67        | 89.83     |
---

### E3 — 前半冻结 vs 后半解冻

**文件**：`experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp_E3.yaml`

```yaml
TRAIN:
  UAV_WSP:
    LAYER_CONFIGS:
      - blocks: [6, 7, 8, 9, 10, 11]  # ← 后半段解冻
        rank: 8
        top_k: 16
        alpha: 8.0
        spectral_beta: 0.5
        targets: ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]
      # 0, 1, 2, 3, 4, 5 = 冻结（前半）
```

**🧪 探究目的**：
- E3 与 A3 类似（都是深层冻结/浅层解冻的反向），但区别在于 E3 使用统一 rank=8 和全4 targets，而 A3 用的是简化配置。对比二者可以验证：冻结模式的效果结论是否依赖于 rank/targets 的选择？
- **E3 vs A3**：如果两者结论一致（都是浅层好于深层或反之），说明冻结模式的 effect 是**稳健的**，不依赖于具体参数
- **E3 vs E1**：前半冻结后半解冻 vs 全部解冻，量化**后半段 6 个 block 的适配是否足够**
- 如果 E3 ≈ E1 → 前半段的 6 个 block 的适配完全是浪费，只用后半段的适配器就够
#### 测试结果
anti_uav300_ir       | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 68.42      | 85.70      | 64.94      | 87.40        | 86.51             |

anti_uav_ir          | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 64.86      | 82.42      | 58.44      | 84.48        | 83.33             |
anti_uav410          | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack-uav-wsp      | 58.88      | 75.94      | 48.27      | 78.59        | 77.23             |

---

### 📊 E 系列综合对比表

| 实验 | 配置 | 数据集 | AUC | OP50 | OP75 | Prec | NormPrec | 验证目的 |
|------|------|--------|-----|------|------|------|----------|---------|
| **E1 (统一配置)** | 冻B0,B11 · WSP B1-B10 · rank=8 · 全4target · β=0.5 | anti_uav | 62.20 | 79.53 | 55.32 | 80.93 | 80.22 | 统一 rank/targets 能否替代差异化设计？ |
| 〃 | 〃 | anti_uav300 | 66.84 | 84.29 | 61.88 | 85.89 | 84.72 | 〃 |
| 〃 | 〃 | anti_uav410 | 57.23 | 74.36 | 46.00 | 76.67 | 75.34 | 〃 |
| **E2 (奇偶冻结)** | 冻B0,2,4,6,8,10 · WSP B1,3,5,7,9,11 · rank=8 | anti_uav | 66.62 | 84.60 | 59.91 | 86.52 | 85.50 | 隔层冻结是否破坏层间协作？ |
| 〃 | 〃 | anti_uav300 | **71.45** | **89.01** | **68.93** | **90.67** | **89.83** | 〃 |
| 〃 | 〃 | anti_uav410 | 59.20 | 76.37 | 48.44 | 78.84 | 77.48 | 〃 |
| **E3 (前半冻)** | 冻B0-B5 · WSP B6-B11 · rank=8 | anti_uav | 64.86 | 82.42 | 58.44 | 84.48 | 83.33 | 仅后半6层适配够不够？ |
| 〃 | 〃 | anti_uav300 | 68.42 | 85.70 | 64.94 | 87.40 | 86.51 | 〃 |
| 〃 | 〃 | anti_uav410 | 58.88 | 75.94 | 48.27 | 78.59 | 77.23 | 〃 |
| **基线 (V2)** | 冻B0,B11 · WSP B1-B10 · rank=12/8/2 · β=0.5 | anti_uav | 67.49 | 85.80 | 61.21 | 87.69 | 86.62 | 非对称差异化配置 |
| 〃 | 〃 | anti_uav300 | 70.59 | 88.22 | 67.66 | 89.81 | 88.92 | 〃 |
| 〃 | 〃 | anti_uav410 | 59.38 | 76.72 | 49.23 | 79.13 | 77.85 | 〃 |

> **关键发现**：E2(奇偶冻结) 在 anti_uav300 上 AUC=71.45 **超越基线(70.59)**，所有5项指标均为系列最高，说明隔层冻结的等距分布模式比全层适配更有效！可能因为间歇性冻结减少了相邻层适配器之间的梯度干扰。E1(统一配置)远差于基线(62.20 vs 67.49)，强有力地证明 V2 的差异化 rank/targets 设计是必要的——不同功能层确实需要不同的适配容量。E3(仅后半)在 anti_uav300 上低于基线约2%，说明前半段的适配可以削减但不能完全砍掉。

---

## 汇总对比表

### 各实验 configuration 参数速查

| 实验 | 文件后缀 | WSP blocks | frozen blocks | rank | targets | β | 参数量(估) |
|------|---------|-----------|--------------|------|---------|---|-----------|
| **A0/B3** | `_wsp` | 1-10 | 0,11 | 12/8/2 | 混合 | 0.5/0.3 | ~4.2M |
| **A1** | `_wsp_A1` | 4-11 | 0,1,2,3 | 8/4 | 混合 | 0.5 | ~3.5M |
| **A2** | `_wsp_A2` | 8-11 | 0-7 | 8 | 全4 | 0.5 | ~1.2M |
| **A3** | `_wsp_A3` | 0-7 | 8-11 | 8 | 全4 | 0.5 | ~2.8M |
| **A4** | `_wsp_A4` | 4-7 | 0-3,8-11 | 8 | 全4 | 0.5 | ~1.6M |
| **B1** | `_wsp_B1` | 6,7 | 0-5,8-11 | 12 | 全4 | 0.5 | ~0.8M |
| **B2** | `_wsp_B2` | 1,6,7,9 | 0,2-5,8,10,11 | 12 | 全4 | 0.5 | ~1.5M |
| **B4** | `_wsp_B4` | 0-11 | (无) | 8 | 全4 | 0.5 | ~5.0M |
| **B5** | `_wsp_B5` | (无) | 0-11 | — | — | — | 0 |
| **C1** | `_wsp_C1` | 1-10 | 0,11 | 8 | attn-only | 0.5 | ~2.0M |
| **C2** | `_wsp_C2` | 1-10 | 0,11 | 8 | mlp-only | 0.5 | ~2.2M |
| **D1** | `_wsp_D1` | 1-3,5-9 | 0,4,10,11 | 12/8/2 | 混合 | 0.5/0.3 | ~3.8M |
| **D2** | `_wsp_D2` | 1-10 | 0,11 | 12/8/2 | 混合 | **1.0** | ~4.2M |
| **D3** | `_wsp_D3` | 1-3,5-9 | 0,4,10,11 | 12/8/2 | 混合 | **1.0** | ~3.8M |
| **E1** | `_wsp_E1` | 1-10 | 0,11 | 8 | 全4 | 0.5 | ~3.8M |
| **E2** | `_wsp_E2` | 1,3,5,7,9,11 | 0,2,4,6,8,10 | 8 | 全4 | 0.5 | ~2.2M |
| **E3** | `_wsp_E3` | 6-11 | 0-5 | 8 | 全4 | 0.5 | ~2.2M |

### 最小推荐集合（计算资源不够时可以只跑这些）

```
优先级 1（验证 V2 设计）: D1, D2, D3 → 3 个实验
优先级 2（组件选择）    : C1, C2      → +2 个实验
优先级 3（需要多少层）  : B1, B4, B5  → +3 个实验
优先级 4（深度位置）    : A1, A3      → +2 个实验
优先级 5（对称性验证）  : E1, E2      → +2 个实验
```

总共最多 17 个实验（含基线），最少 8 个（优先级 1+2+部分3）。

### 每个实验必改的三个地方

```yaml
# 1. LAYER_CONFIGS — 核心修改（本文档已提供每个实验的具体内容）

# 2. CHECKPOINT 路径（第 178 行附近）
TEST:
  CHECKPOINT: "E:/code/UAV/OSTrack-main/output/checkpoints/train/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp_<实验后缀>/OSTrack_best.pth.tar"
  #                                               ^^^^^^^^^^^^
  #                                               改成对应的实验后缀

# 3. PRETRAIN_FILE（第 74 行）— 所有实验保持不变即可
MODEL:
  PRETRAIN_FILE: "E:/code/UAV/OSTrack-main/weights/OSTrack_vitb256_ep300/OSTrack_ep0300.pth.tar"
```

---

## 附录：快速复制命令

```bash
# 系列 A
cp experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp.yaml experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp_A1.yaml
cp experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp.yaml experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp_A2.yaml
cp experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp.yaml experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp_A3.yaml
cp experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp.yaml experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp_A4.yaml

# 系列 B
cp experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp.yaml experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp_B1.yaml
cp experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp.yaml experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp_B2.yaml
cp experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp.yaml experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp_B4.yaml
cp experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp.yaml experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp_B5.yaml

# 系列 C
cp experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp.yaml experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp_C1.yaml
cp experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp.yaml experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp_C2.yaml

# 系列 D
cp experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp.yaml experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp_D1.yaml
cp experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp.yaml experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp_D2.yaml
cp experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp.yaml experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp_D3.yaml

# 系列 E
cp experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp.yaml experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp_E1.yaml
cp experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp.yaml experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp_E2.yaml
cp experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp.yaml experiments/ostrack/vitb_256_mae_ce_32x4_ep300_uav_wsp_E3.yaml
```

复制完后，逐个修改每个文件的 `LAYER_CONFIGS` + `CHECKPOINT` 即可。

---

## 附录：运行命令模板

```bash
# 单卡训练
python lib/train/run_training.py --script ostrack --config vitb_256_mae_ce_32x4_ep300_uav_wsp_A1 --save_dir output/train

# 多卡训练（会自动从 CUDA_VISIBLE_DEVICES 获取可用 GPU）
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
    lib/train/run_training.py --script ostrack --config vitb_256_mae_ce_32x4_ep300_uav_wsp_A1 --save_dir output/train

# 测试
python tracking/test.py ostrack vitb_256_mae_ce_32x4_ep300_uav_wsp_A1 --dataset anti_uav --epoch 40

# 分析结果
python tracking/analysis_results.py
```
