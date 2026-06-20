#!/usr/bin/env bash
# =============================================================================
# 消融实验批量运行脚本
# =============================================================================
# 用法：
#   bash tools/run_ablation_series.sh <系列名>
#  bash tools/run_ablation_series.sh -c "1" -g 1 A#
#
# 示例：
#   bash tools/run_ablation_series.sh A      # 跑 A1→A2→A3→A4
#   bash tools/run_ablation_series.sh D      # 跑 D1→D2→D3
#   bash tools/run_ablation_series.sh all    # 跑全部系列
#   bash tools/run_ablation_series.sh A B D  # 跑 A 和 B 和 D 系列
#
# 也可以手动指定实验列表：
#   bash tools/run_ablation_series.sh custom vitb_256_mae_ce_32x4_ep300_uav_wsp_A1.yaml vitb_256_mae_ce_32x4_ep300_uav_wsp_C1.yaml
#
# 输出说明：
#   - 每个实验的日志保存在 logs/<实验名>.log
#   - 训练完成后自动记录结束时间和 exit code
#   - 某个实验失败时默认会继续下一个（除非设置了 -e 选项）
# =============================================================================

set -o pipefail

# ─── 配置 ────────────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
EXP_DIR="${PROJECT_DIR}/experiments/ostrack"
LOG_DIR="${PROJECT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 可选的 python 解释器（支持 conda 环境）
PYTHON="${PYTHON:-python}"

# 是否在实验失败时中止整个序列（默认 false）
EXIT_ON_ERROR="${EXIT_ON_ERROR:-false}"

# 是否使用分布式多卡训练（默认 false，单卡）
DISTRIBUTED="${DISTRIBUTED:-false}"
NUM_GPUS="${NUM_GPUS:-4}"

# CUDA_VISIBLE_DEVICES — 控制使用哪些 GPU
# 不设置 = 使用所有可用 GPU
# 示例: CUDA_VISIBLE_DEVICES="0,2,3" → 只用 GPU 0, 2, 3
CUDA_VISIBLE=""

# ─── 系列 → 实验配置映射 ────────────────────────────────────────────────────
declare -A SERIES_MAP
SERIES_MAP[A]="
  vitb_256_mae_ce_32x4_ep300_uav_wsp_A1
  vitb_256_mae_ce_32x4_ep300_uav_wsp_A2
  vitb_256_mae_ce_32x4_ep300_uav_wsp_A3
  vitb_256_mae_ce_32x4_ep300_uav_wsp_A4
"
SERIES_MAP[B]="
  vitb_256_mae_ce_32x4_ep300_uav_wsp_B1
  vitb_256_mae_ce_32x4_ep300_uav_wsp_B2
  vitb_256_mae_ce_32x4_ep300_uav_wsp_B4
  vitb_256_mae_ce_32x4_ep300_uav_wsp_B5
"
SERIES_MAP[C]="
  vitb_256_mae_ce_32x4_ep300_uav_wsp_C1
  vitb_256_mae_ce_32x4_ep300_uav_wsp_C2
"
SERIES_MAP[D]="
  vitb_256_mae_ce_32x4_ep300_uav_wsp_D1
  vitb_256_mae_ce_32x4_ep300_uav_wsp_D2
  vitb_256_mae_ce_32x4_ep300_uav_wsp_D3
"
SERIES_MAP[E]="
  vitb_256_mae_ce_32x4_ep300_uav_wsp_E1
  vitb_256_mae_ce_32x4_ep300_uav_wsp_E2
  vitb_256_mae_ce_32x4_ep300_uav_wsp_E3
"

# =============================================================================
# 辅助函数
# =============================================================================

usage() {
    echo "用法: bash $0 <系列名> [<系列名> ...]"
    echo "     bash $0 all"
    echo "     bash $0 custom <yaml1> [<yaml2> ...]"
    echo ""
    echo "系列名: A, B, C, D, E   (可同时指定多个，如 'bash $0 A D')"
    echo "选项:"
    echo "  -e        实验失败时中止后续实验（默认继续）"
    echo "  -d        使用分布式多卡训练（默认单卡）"
    echo "  -g N      GPU 数量（配合 -d 使用，默认 4）"
    echo "  -c IDS    指定 GPU 编号，如 -c \"0,1,2,3\" 或 -c \"1\""
    echo "            不指定 = 使用所有可用 GPU"
    echo "  -p PATH   指定 python 解释器路径"
    echo ""
    echo "示例:"
    echo "  bash $0 -c \"0,1\" A                  # 用 GPU 0,1 跑 A 系列（分布式中 g=2 时）"
    echo "  bash $0 -c \"1\" -g 1 A                # 只用 GPU 1 单卡跑 A 系列"
    echo "  bash $0 -d -c \"0,2,3\" -g 3 A         # 用 GPU 0,2,3 三卡分布式跑 A 系列"
    exit 1
}

# 打印带时间戳的日志
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# 运行单个实验
run_experiment() {
    local exp_name="$1"
    local yaml_file="${EXP_DIR}/${exp_name}.yaml"
    local log_file="${LOG_DIR}/${exp_name}_${TIMESTAMP}.log"
    local status_file="${LOG_DIR}/${exp_name}_${TIMESTAMP}.status"

    # 检查 YAML 是否存在
    if [ ! -f "$yaml_file" ]; then
        log "❌ 文件不存在: ${yaml_file}"
        return 1
    fi

    # 确保 logs 目录存在
    mkdir -p "$LOG_DIR"

    log "══════════════════════════════════════════════════════════════"
    log "🚀 开始实验: ${exp_name}"
    log "   YAML:  ${yaml_file}"
    log "   日志:  ${log_file}"
    if [ -n "$CUDA_VISIBLE" ]; then
        log "   GPU:   ${CUDA_VISIBLE}"
    fi
    log "══════════════════════════════════════════════════════════════"

    START_TIME=$(date +%s)

    # 设置 CUDA_VISIBLE_DEVICES（如果指定了 -c）
    if [ -n "$CUDA_VISIBLE" ]; then
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE}"
    fi

    # 从 yaml 文件名提取实验名（去掉 .yaml 后缀）
    local exp_basename="$(basename "$exp_name" .yaml)"

    if [ "$DISTRIBUTED" = "true" ]; then
        # 分布式多卡训练
        # torch.distributed.launch 会自动设置 --local_rank
        $PYTHON -m torch.distributed.launch \
            --nproc_per_node="$NUM_GPUS" \
            "${PROJECT_DIR}/lib/train/run_training.py" \
            --script ostrack --config "$exp_basename" \
            --save_dir output/train 2>&1 | tee "$log_file"
    else
        # 单卡训练
        $PYTHON "${PROJECT_DIR}/lib/train/run_training.py" \
            --script ostrack --config "$exp_basename" \
            --save_dir output/train 2>&1 | tee "$log_file"
    fi

    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    DURATION_MIN=$((DURATION / 60))
    DURATION_SEC=$((DURATION % 60))

    # 保存状态
    echo "exit_code=${EXIT_CODE}" > "$status_file"
    echo "start=${START_TIME}" >> "$status_file"
    echo "end=${END_TIME}" >> "$status_file"
    echo "duration_sec=${DURATION}" >> "$status_file"

    if [ $EXIT_CODE -eq 0 ]; then
        log "✅ 实验完成: ${exp_name}  (${DURATION_MIN}m${DURATION_SEC}s)"
    else
        log "❌ 实验失败: ${exp_name}  (exit=${EXIT_CODE}, ${DURATION_MIN}m${DURATION_SEC}s)"
    fi

    return $EXIT_CODE
}

# =============================================================================
# 主流程
# =============================================================================

# 解析命令行选项
while getopts "edg:c:p:h" opt; do
    case $opt in
        e) EXIT_ON_ERROR=true ;;
        d) DISTRIBUTED=true ;;
        g) NUM_GPUS="$OPTARG" ;;
        c) CUDA_VISIBLE="$OPTARG" ;;
        p) PYTHON="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done
shift $((OPTIND-1))

if [ $# -lt 1 ]; then
    usage
fi

# 收集要运行的实验列表
declare -a EXPERIMENTS

if [ "$1" = "all" ]; then
    # 跑所有系列
    for series in A B C D E; do
        for exp in ${SERIES_MAP[$series]}; do
            EXPERIMENTS+=("$exp")
        done
    done
elif [ "$1" = "custom" ]; then
    # 手动指定实验列表（去掉 .yaml 后缀和路径前缀）
    shift
    for arg in "$@"; do
        exp_name="$(basename "$arg" .yaml)"
        EXPERIMENTS+=("$exp_name")
    done
else
    # 指定系列名
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

# 打印实验计划
log "══════════════════════════════════════════════════════════════"
log "📋 实验计划（共 ${#EXPERIMENTS[@]} 个）"
log "   工作目录: ${PROJECT_DIR}"
log "   Python:   ${PYTHON}"
if [ -n "$CUDA_VISIBLE" ]; then
    log "   GPU:      ${CUDA_VISIBLE}"
fi
log "   模式:     $([ "$DISTRIBUTED" = "true" ] && echo "分布式(${NUM_GPUS}卡)" || echo "单卡")"
log "   失败策略: $([ "$EXIT_ON_ERROR" = "true" ] && echo "中止" || echo "继续")"
log "══════════════════════════════════════════════════════════════"

for i in "${!EXPERIMENTS[@]}"; do
    log "   $((i+1)). ${EXPERIMENTS[$i]}"
done

log "══════════════════════════════════════════════════════════════"

TOTAL_START=$(date +%s)
SUCCESS=0
FAILED=0

# 逐实验运行
for i in "${!EXPERIMENTS[@]}"; do
    exp="${EXPERIMENTS[$i]}"
    total=$((i + 1))

    log ""
    log "▶▶▶▶▶ [${total}/${#EXPERIMENTS[@]}] ${exp} ◀◀◀◀◀"

    run_experiment "$exp"
    rc=$?

    if [ $rc -eq 0 ]; then
        SUCCESS=$((SUCCESS + 1))
    else
        FAILED=$((FAILED + 1))
        if [ "$EXIT_ON_ERROR" = "true" ]; then
            log "🛑 ${exp} 失败，因 -e 选项中止批量运行"
            break
        fi
    fi
done

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MIN=$(((TOTAL_DURATION % 3600) / 60))

# 最终汇总
log ""
log "══════════════════════════════════════════════════════════════"
log "🏁 批量运行结束"
log "   总用时: ${TOTAL_HOURS}h${TOTAL_MIN}m"
log "   成功:   ${SUCCESS}"
log "   失败:   ${FAILED}"
log "══════════════════════════════════════════════════════════════"

exit $FAILED
