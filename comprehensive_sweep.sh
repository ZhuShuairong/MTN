#!/bin/bash

mkdir -p logs/gpustat logs/training logs/configs

TEXT_PROMPT="a tiger dressed as a doctor"
BASELINE_ITERS=6000
BASE_SEED=3407

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
GPU_LOG="logs/gpustat/gpu_sweep_${TIMESTAMP}.log"
echo "Starting GPU monitoring: $GPU_LOG"
gpustat --watch 60 --show-pid --show-power --show-fan > "$GPU_LOG" &
GPUSTAT_PID=$!

cleanup() {
    echo "Stopping GPU monitoring..."
    kill $GPUSTAT_PID 2>/dev/null
    exit
}
trap cleanup SIGINT SIGTERM EXIT

run_exp() {
    local name=$1
    local extra_args=$2
    local sd_version=$3
    
    local workspace="sweep_${sd_version}_${name}"
    local log_file="logs/training/${workspace}.log"
    local completed_check="sweep_${sd_version}_${name}"
    
    # Check if already completed in previous run
    if grep -q "$completed_check" logs/completed.log 2>/dev/null; then
        echo "⏩ SKIPPING (already completed): $workspace"
        return 0
    fi
    
    echo ""
    echo "========================================"
    echo "Experiment: $workspace"
    echo "SD Version: $sd_version"
    echo "Args: $extra_args"
    echo "Time: $(date)"
    
    # Check if resuming
    if [ -f "$workspace/checkpoints/df_ep0029.pth" ] || [ -f "$workspace/checkpoints/df_latest.pth" ]; then
        echo "🔄 RESUMING from checkpoint..."
    else
        echo "🚀 STARTING fresh..."
    fi
    
    gpustat
    echo "========================================"
    
    # Removed --ckpt scratch to allow auto-resume (default is 'latest')
    python main.py --text="$TEXT_PROMPT" --iters="$BASELINE_ITERS" --seed="$BASE_SEED" -O --save_mesh --grad_clip 1.0 \
        --workspace "$workspace" \
        --sd_version "$sd_version" \
        $extra_args \
        2>&1 | tee "$log_file"
    
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "⚠️  FAILED: $workspace" | tee -a logs/failures.log
    else
        echo "✅ COMPLETED: $workspace" | tee -a logs/completed.log
    fi
    sleep 5
    return $exit_code
}

echo "Starting resumable hyperparameter sweep..."
echo "Total experiments: 27 runs (will skip already completed)"
echo ""

# ==========================================
# 1. BASELINES (All defaults)
# ==========================================
for sd in 1.5 2.0 2.1; do
    run_exp "baseline" "" "$sd"
done

# ==========================================
# 2. LEARNING RATE (default: 1e-3)
# ==========================================
for sd in 1.5 2.0 2.1; do
    run_exp "lr_low" "--lr 5e-4" "$sd"
    run_exp "lr_high" "--lr 2e-3" "$sd"
done

# ==========================================
# 3. GUIDANCE SCALE (default: 100)
# ==========================================
for sd in 1.5 2.0 2.1; do
    run_exp "guidance_low" "--guidance_scale 50" "$sd"
    run_exp "guidance_high" "--guidance_scale 150" "$sd"
done

# ==========================================
# 4. RESOLUTION (default: 64×64)
# ==========================================
for sd in 1.5 2.0 2.1; do
    run_exp "res_low" "--w 48 --h 48" "$sd"
    run_exp "res_high" "--w 96 --h 96" "$sd"
done

# ==========================================
# 5. DATASET SIZE (default: 200)
# ==========================================
for sd in 1.5 2.0 2.1; do
    run_exp "dataset_low" "--dataset_size_train 100" "$sd"
    run_exp "dataset_high" "--dataset_size_train 400" "$sd"
done

echo ""
echo "========================================"
echo "SWEEP COMPLETED!"
echo "========================================"
echo "Completed: $(cat logs/completed.log 2>/dev/null | wc -l || echo 0)"
echo "Failed: $(cat logs/failures.log 2>/dev/null | wc -l || echo 0)"
