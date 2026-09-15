#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIGS=(
  asc_lora_ablation_F0_full
  asc_lora_ablation_F1_no_channel_gate
  asc_lora_ablation_F2_no_spectral_complement
  asc_lora_ablation_F3_hard_spectral_complement
  asc_lora_ablation_F4_no_complement_constraint
  asc_lora_ablation_F5_standard_lora
)
DATASETS=(anti_uav anti_uav300 anti_uav410)

for config in "${CONFIGS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    python "$ROOT/tracking/test.py" ostrack "$config" --dataset_name "$dataset" \
      --threads 3 --num_gpus 1
  done
done
