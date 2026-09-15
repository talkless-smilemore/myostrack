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

for config in "${CONFIGS[@]}"; do
  experiment="${config#asc_lora_ablation_}"
  python "$ROOT/tracking/train.py" --script ostrack --config "$config" \
    --save_dir "$ROOT/output/asc_lora_ablations/$experiment" --mode single
done
