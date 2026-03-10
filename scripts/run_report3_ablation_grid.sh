#!/usr/bin/env bash

set -uo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
TEXT_PROMPT="${TEXT_PROMPT:-a tiger dressed as a doctor}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-report_runs}"
GPUSTAT_INTERVAL="${GPUSTAT_INTERVAL:-60}"

LR_DEFAULT="${LR_DEFAULT:-0.001}"
LR_CHANGED="${LR_CHANGED:-0.0005}"

BATCH_SIZE_DEFAULT="${BATCH_SIZE_DEFAULT:-1}"
BATCH_SIZE_CHANGED="${BATCH_SIZE_CHANGED:-2}"

RESOLUTION_DEFAULT="${RESOLUTION_DEFAULT:-64}"
RESOLUTION_CHANGED="${RESOLUTION_CHANGED:-96}"

SD_VERSION_DEFAULT="${SD_VERSION_DEFAULT:-2.1}"
SD_VERSION_CHANGED="${SD_VERSION_CHANGED:-2.0}"

if ! command -v gpustat >/dev/null 2>&1; then
    echo "gpustat is required for this script. Install it with: pip install gpustat" >&2
    exit 1
fi

mkdir -p "$WORKSPACE_ROOT"

run_experiment() {
    local lr_label="$1"
    local views_label="$2"
    local res_label="$3"
    local sd_label="$4"
    local lr_value="$5"
    local batch_size_value="$6"
    local resolution_value="$7"
    local sd_version_value="$8"

    local workspace_name="default"
    if [[ "$lr_label" != "default" || "$views_label" != "default" || "$res_label" != "default" || "$sd_label" != "default" ]]; then
        workspace_name="lr-${lr_label}_views-${views_label}_res-${res_label}_sd-${sd_label}"
    fi

    local workspace_path="${WORKSPACE_ROOT}/${workspace_name}"
    local gpustat_pid=""
    local exit_code=0

    mkdir -p "$workspace_path"

    printf '%s\n' \
        "text=${TEXT_PROMPT}" \
        "lr=${lr_value}" \
        "batch_size=${batch_size_value}" \
        "resolution=${resolution_value}" \
        "sd_version=${sd_version_value}" \
        > "${workspace_path}/run_config.txt"

    printf '%q ' "$PYTHON_BIN" main.py --text "$TEXT_PROMPT" --workspace "$workspace_path" -O --lr "$lr_value" --batch_size "$batch_size_value" --w "$resolution_value" --h "$resolution_value" --sd_version "$sd_version_value" > "${workspace_path}/command.sh"
    printf '\n' >> "${workspace_path}/command.sh"

    gpustat --show-full-cmd > "${workspace_path}/gpustat_start.log" 2>&1 || true
    gpustat --show-full-cmd --watch "$GPUSTAT_INTERVAL" > "${workspace_path}/gpustat_watch.log" 2>&1 &
    gpustat_pid=$!

    {
        echo "Starting workspace ${workspace_name}"
        echo "Logging to ${workspace_path}/train.log"
        "$PYTHON_BIN" main.py \
            --text "$TEXT_PROMPT" \
            --workspace "$workspace_path" \
            -O \
            --lr "$lr_value" \
            --batch_size "$batch_size_value" \
            --w "$resolution_value" \
            --h "$resolution_value" \
            --sd_version "$sd_version_value"
    } |& tee "${workspace_path}/train.log"
    exit_code=${PIPESTATUS[0]}

    if [[ -n "$gpustat_pid" ]]; then
        kill "$gpustat_pid" >/dev/null 2>&1 || true
        wait "$gpustat_pid" 2>/dev/null || true
    fi

    gpustat --show-full-cmd > "${workspace_path}/gpustat_end.log" 2>&1 || true
    printf '%s\n' "$exit_code" > "${workspace_path}/exit_code.txt"

    if [[ "$exit_code" -ne 0 ]]; then
        echo "Run failed for ${workspace_name} with exit code ${exit_code}" >&2
    fi

    return "$exit_code"
}

failures=()

for lr_label in default changed; do
    if [[ "$lr_label" == "default" ]]; then
        lr_value="$LR_DEFAULT"
    else
        lr_value="$LR_CHANGED"
    fi

    for views_label in default changed; do
        if [[ "$views_label" == "default" ]]; then
            batch_size_value="$BATCH_SIZE_DEFAULT"
        else
            batch_size_value="$BATCH_SIZE_CHANGED"
        fi

        for res_label in default changed; do
            if [[ "$res_label" == "default" ]]; then
                resolution_value="$RESOLUTION_DEFAULT"
            else
                resolution_value="$RESOLUTION_CHANGED"
            fi

            for sd_label in default changed; do
                if [[ "$sd_label" == "default" ]]; then
                    sd_version_value="$SD_VERSION_DEFAULT"
                else
                    sd_version_value="$SD_VERSION_CHANGED"
                fi

                if ! run_experiment "$lr_label" "$views_label" "$res_label" "$sd_label" "$lr_value" "$batch_size_value" "$resolution_value" "$sd_version_value"; then
                    failures+=("lr-${lr_label}_views-${views_label}_res-${res_label}_sd-${sd_label}")
                fi
            done
        done
    done
done

if [[ "${#failures[@]}" -gt 0 ]]; then
    printf '%s\n' "Completed with failed runs:" >&2
    printf '%s\n' "${failures[@]}" >&2
    exit 1
fi

echo "All 16 runs completed successfully."