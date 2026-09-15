#!/usr/bin/env bash
# 批量运行测试（针对 anti_uav / anti_uav300 / anti_uav410 三套数据集）
set -o pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
EXP_DIR="${PROJECT_DIR}/experiments/ostrack"
LOG_DIR="${PROJECT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

PYTHON="${PYTHON:-python}"
EXIT_ON_ERROR="${EXIT_ON_ERROR:-false}"
CUDA_VISIBLE=""
THREADS=1
NUM_GPUS=1
MODALITY="both"

declare -A SERIES_MAP
SERIES_MAP[A]="
  vitb_256_mae_ce_32x4_ep300_asc_lora_A1
  vitb_256_mae_ce_32x4_ep300_asc_lora_A2
  vitb_256_mae_ce_32x4_ep300_asc_lora_A3
  vitb_256_mae_ce_32x4_ep300_asc_lora_A4
"
SERIES_MAP[B]="
  vitb_256_mae_ce_32x4_ep300_asc_lora_B1
  vitb_256_mae_ce_32x4_ep300_asc_lora_B2
  vitb_256_mae_ce_32x4_ep300_asc_lora_B4
  vitb_256_mae_ce_32x4_ep300_asc_lora_B5
"
SERIES_MAP[C]="
  vitb_256_mae_ce_32x4_ep300_asc_lora_C1
  vitb_256_mae_ce_32x4_ep300_asc_lora_C2
"
SERIES_MAP[D]="
  vitb_256_mae_ce_32x4_ep300_asc_lora_D1
  vitb_256_mae_ce_32x4_ep300_asc_lora_D2
  vitb_256_mae_ce_32x4_ep300_asc_lora_D3
"
SERIES_MAP[E]="
  vitb_256_mae_ce_32x4_ep300_asc_lora_E1
  vitb_256_mae_ce_32x4_ep300_asc_lora_E2
  vitb_256_mae_ce_32x4_ep300_asc_lora_E3
"

usage() {
    echo "用法: bash $0 <系列名> [<系列名> ...]"
    echo "   bash $0 all"
    echo "   bash $0 custom <yaml1> [<yaml2> ...]"
    echo "选项:"
    echo "  -e        测试失败时中止后续（默认继续)" 
    echo "  -c IDS    指定 CUDA_VISIBLE_DEVICES（例如 -c \"0,1\"）"
    echo "  -p PATH   指定 python 解释器路径"
    echo "  -t N      线程数（tracking/test.py 的 --threads，默认 1)"
    echo "  -g N      num_gpus（tracking/test.py 的 --num_gpus，默认 1）"
    echo "  -m MODE   模态: both | ir | rgb  （默认 both）"
    exit 1
}

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# 尝试找到 yaml 文件（容错文件名中多余空格或扩展名变体）
find_yaml() {
    local base="$1"
    # 直接匹配
    if [ -f "${EXP_DIR}/${base}.yaml" ]; then
        echo "${EXP_DIR}/${base}.yaml"
        return 0
    fi
    # 兼容可能的空格或小差异，使用 glob
    shopt -s nullglob
    candidates=("${EXP_DIR}/${base}"*.yaml "${EXP_DIR}/${base}"*.yml)
    shopt -u nullglob
    if [ ${#candidates[@]} -gt 0 ]; then
        # 返回第一个匹配项
        echo "${candidates[0]}"
        return 0
    fi
    return 1
}

run_single_test() {
    local yaml_path="$1"
    local dataset_name="$2"
    local exp_basename
    exp_basename=$(basename "${yaml_path%.*}")

    mkdir -p "$LOG_DIR"
    local log_file="${LOG_DIR}/${exp_basename}_${dataset_name}_${TIMESTAMP}.log"

    log "▶ 开始测试: ${exp_basename}  数据集=${dataset_name}"
    if [ -n "$CUDA_VISIBLE" ]; then
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE}"
        log "   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE}"
    fi

    # 调用 tracking/test.py（强制使用 threads=1，避免多进程导致 CPU 瓶颈）
    "$PYTHON" tracking/test.py ostrack "${exp_basename}" --dataset_name "${dataset_name}" --threads 1 --num_gpus ${NUM_GPUS} 2>&1 | tee "$log_file"
    rc=${PIPESTATUS[0]}

    if [ $rc -eq 0 ]; then
        log "✅ 测试完成: ${exp_basename} (${dataset_name})"
    else
        log "❌ 测试失败: ${exp_basename} (${dataset_name})  exit=${rc}"
    fi
    return $rc
}

run_tests_for_exp() {
    local exp_name="$1"  # 例如 vitb_256_mae_ce_32x4_ep300_asc_lora_A1
    local dataset_tokens=("uav" "uav300" "uav410")
    declare -A map
    if [ "$MODALITY" = "ir" ]; then
        map[uav]=anti_uav_ir
        map[uav300]=anti_uav300_ir
        map[uav410]=anti_uav410
    elif [ "$MODALITY" = "rgb" ]; then
        map[uav]=anti_uav_rgb
        map[uav300]=anti_uav300_rgb
        map[uav410]=anti_uav410
    else
        map[uav]=anti_uav
        map[uav300]=anti_uav300
        map[uav410]=anti_uav410
    fi

    for token in "${dataset_tokens[@]}"; do
        # 用现有 exp_name 中的 uav/ uav300/ uav410 段替换为目标 token
        candidate="$exp_name"
        candidate="${candidate/_uav300_/_${token}_}"
        candidate="${candidate/_uav410_/_${token}_}"
        candidate="${candidate/_uav_/_${token}_}"

        yaml_path=$(find_yaml "$candidate") || yaml_path=""
        if [ -z "$yaml_path" ]; then
            log "⚠️ 未找到配置: ${candidate}.yaml，跳过 ${token} 测试"
            continue
        fi

        run_single_test "$yaml_path" "${map[$token]}"
        rc=$?
        if [ $rc -ne 0 ]; then
            if [ "$EXIT_ON_ERROR" = "true" ]; then
                log "🛑 因 -e 选项, 在 ${exp_name} 中止批量测试"
                return $rc
            fi
        fi
    done
}

# 解析选项
while getopts "ec:p:t:g:m:h" opt; do
    case $opt in
        e) EXIT_ON_ERROR=true ;;
        c) CUDA_VISIBLE="$OPTARG" ;;
        p) PYTHON="$OPTARG" ;;
        t) THREADS="$OPTARG" ;;
        g) NUM_GPUS="$OPTARG" ;;
        m) MODALITY="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done
shift $((OPTIND-1))

if [ $# -lt 1 ]; then
    usage
fi

declare -a EXPERIMENTS

if [ "$1" = "all" ]; then
    for series in A B C D E; do
        for exp in ${SERIES_MAP[$series]}; do
            EXPERIMENTS+=("$exp")
        done
    done
elif [ "$1" = "custom" ]; then
    shift
    for arg in "$@"; do
        exp_name="$(basename "$arg" .yaml)"
        EXPERIMENTS+=("$exp_name")
    done
else
    for series in "$@"; do
        series_upper=$(echo "$series" | tr '[:lower:]' '[:upper:]')
        if [ -z "${SERIES_MAP[$series_upper]}" ]; then
            log "⚠️ 未知系列: ${series}，可用系列: A B C D E"
            exit 1
        fi
        for exp in ${SERIES_MAP[$series_upper]}; do
            EXPERIMENTS+=("$exp")
        done
    done
fi

log "批量测试计划: 共 ${#EXPERIMENTS[@]} 个实验（每个实验尝试三套数据集）"

TOTAL_FAILED=0

for i in "${!EXPERIMENTS[@]}"; do
    exp="${EXPERIMENTS[$i]}"
    log "\n===== [$(($i+1))/${#EXPERIMENTS[@]}] 处理 ${exp} ====="
    run_tests_for_exp "$exp"
    rc=$?
    if [ $rc -ne 0 ]; then
        TOTAL_FAILED=$((TOTAL_FAILED+1))
    fi
done

log "批量测试结束。失败的实验组数: ${TOTAL_FAILED}"
exit $TOTAL_FAILED
