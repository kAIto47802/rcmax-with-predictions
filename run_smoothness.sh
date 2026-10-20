#!/bin/bash
set -xeuo pipefail

epsilon=0.01
fixed_false_positive_rate=0.0
fixed_false_negative_rate=0.0
search_budgets=(0 1 2)
num_workers=16

source .venv/bin/activate

run_smoothness_experiment() {
    local family="$1"
    local m="$2"
    local n="$3"
    local data_seed="${4:-0}"
    local fig_width="${5:-9}"
    local fig_height="${6:-4}"
    local instance="${family}_${m}x${n}_seed${data_seed}"

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
            --false-negative-rate "auto" \
            --false-positive-rate "$fixed_false_positive_rate" \
            --num-workers "$num_workers"
    done

    for K in "${search_budgets[@]}"; do
        python -m experiments.run_prediction_guided \
            --instance "$instance" \
            --epsilon "$epsilon" \
            --K "$K" \
            --false-negative-rate "$fixed_false_negative_rate" \
            --false-positive-rate "auto" \
            --num-workers "$num_workers"
    done

    python -m experiments.run_classical \
        --instance "$instance" \
        --algorithm "lst_two_approx"

    python -m experiments.plot_smoothness \
        --reference-name "rcmax_${instance}_reference.pkl" \
        --baseline-name "rcmax_${instance}_lst_two_approx_trial0.pkl" \
        --instance "$instance" \
        --epsilon "$epsilon" \
        --search-budgets "${search_budgets[@]}" \
        --false-negative-rates "auto" \
        --false-positive-rates "auto" \
        --fixed-false-negative-rate "$fixed_false_negative_rate" \
        --fixed-false-positive-rate "$fixed_false_positive_rate" \
        --num-seeds 30 \
        --use-errors-for-xaxis \
        --figsize "$fig_width" "$fig_height"
}

run_smoothness_experiment "U_1_100" 10 100
run_smoothness_experiment "U_1_100" 10 50 0 10 4
run_smoothness_experiment "U_1_100" 5 100 0 10 4
run_smoothness_experiment "machine_correlated" 10 100 0 10 4
