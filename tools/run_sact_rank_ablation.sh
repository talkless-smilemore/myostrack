#!/usr/bin/env bash
set -euo pipefail

# Run the SACT ASC-LoRA rank ablation with every protocol variable inherited
# from the current best config. Override DATASET/NUM_GPUS/PYTHON as needed.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output/sact_rank_ablation}"
DATASET="${DATASET:-anti_uav}"
NUM_GPUS="${NUM_GPUS:-1}"

ranks=(2 4 8 16 32)

for rank in "${ranks[@]}"; do
    experiment="SACT_ASCLoRA_rank${rank}"
    config="${experiment}"
    experiment_dir="${OUTPUT_DIR}/${experiment}"

    mkdir -p "${experiment_dir}"
    cp "${ROOT}/experiments/ostrack/${config}.yaml" "${experiment_dir}/config.yaml"

    echo "[SACT rank ablation] training ${experiment}"
    "${PYTHON}" "${ROOT}/tracking/train.py" \
        --script ostrack \
        --config "${config}" \
        --save_dir "${experiment_dir}" \
        --mode single

    echo "[SACT rank ablation] evaluating ${experiment} on ${DATASET}"
    "${PYTHON}" "${ROOT}/tracking/test.py" ostrack "${config}" \
        --dataset_name "${DATASET}" \
        --threads 3 \
        --num_gpus "${NUM_GPUS}" \
        --save_dir "${experiment_dir}"
done
