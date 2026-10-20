#!/bin/bash
set -xeuo pipefail

data_seed=0

epsilon=0.01
ptas_epsilon=0.01
mordechai_epsilon=0.01
false_positive_rate=0.1
false_negative_rate=0.1
search_budgets=(0 1 2)
num_trials=5
figsize=(4.5 4)

source .venv/bin/activate

run_runtime_by_m() {
    local family="$1"
    local n="$2"
    local heavy_x_max="$3"
    shift 3
    local logy="false"
    if [ "${1:-}" = "true" ] || [ "${1:-}" = "false" ]; then
        logy="$1"
        shift
    fi
    local ms=("$@")

    run_runtime_experiment "$family" "m" "$n" "$heavy_x_max" "$logy" "${ms[@]}"
}

run_runtime_by_n() {
    local family="$1"
    local m="$2"
    local heavy_x_max="$3"
    shift 3
    local logy="false"
    if [ "${1:-}" = "true" ] || [ "${1:-}" = "false" ]; then
        logy="$1"
        shift
    fi
    local ns=("$@")

    run_runtime_experiment "$family" "n" "$m" "$heavy_x_max" "$logy" "${ns[@]}"
}

run_runtime_experiment() {
    local family="$1"
    local x_axis="$2"
    local fixed_value="$3"
    local heavy_x_max="$4"
    local logy="$5"
    shift 5
    local x_values=("$@")

    local x m n instance
    for x in "${x_values[@]}"; do
        if [ "$x_axis" = "m" ]; then
            m="$x"
            n="$fixed_value"
        else
            m="$fixed_value"
            n="$x"
        fi
        echo "Running for family=$family, m=$m, n=$n"
        instance="${family}_${m}x${n}_seed${data_seed}"

        python -m experiments.prepare_data \
            --family "$family" \
            --m "$m" \
            --n "$n" \
            --seed "$data_seed"

        python -m experiments.run_reference \
            --instance "$instance"

        for K in "${search_budgets[@]}"; do
            python -m experiments.run_prediction_guided \
                --instance "$instance" \
                --epsilon "$epsilon" \
                --K "$K" \
                --false-negative-rate "$false_negative_rate" \
                --false-positive-rate "$false_positive_rate" \
                --num-seeds "$num_trials"
        done

        for trial_id in $(seq 0 $((num_trials - 1))); do
            python -m experiments.run_classical \
                --instance "$instance" \
                --algorithm "lst_two_approx" \
                --trial-id "$trial_id"

            if [ "$x" -le "$heavy_x_max" ]; then
                python -m experiments.run_classical \
                    --instance "$instance" \
                    --algorithm "lst_fixed_m_ptas" \
                    --epsilon "$ptas_epsilon" \
                    --trial-id "$trial_id"

                python -m experiments.run_classical \
                    --instance "$instance" \
                    --algorithm "mordechai" \
                    --epsilon "$mordechai_epsilon" \
                    --trial-id "$trial_id"
            fi
        done
    done

    local plot_args=(
        python -m experiments.plot_runtime
        --family "$family"
        --x-axis "$x_axis"
        --x-values "${x_values[@]}"
        --data-seed "$data_seed"
        --epsilon "$epsilon"
        --ptas-epsilon "$ptas_epsilon"
        --false-negative-rate "$false_negative_rate"
        --false-positive-rate "$false_positive_rate"
        --search-budgets "${search_budgets[@]}"
        --num-seeds "$num_trials"
        --num-trials "$num_trials"
        --heavy-x-max "$heavy_x_max"
        --figsize "${figsize[@]}"
    )
    if [ "$x_axis" = "m" ]; then
        plot_args+=(--n "$fixed_value")
    else
        plot_args+=(--m "$fixed_value")
    fi
    if [ "$logy" = "true" ]; then
        plot_args+=(--logy)
    fi
    "${plot_args[@]}"
}

ms=(2 3 4 5 6 7 8 9 10)
ns=(5 6 7 8 9 10 11 12 13 14 15)

run_runtime_by_m "U_1_100" 10 10 "${ms[@]}"

figsize=(6 4)
run_runtime_by_n "U_1_100" 5 12 "${ns[@]}"
