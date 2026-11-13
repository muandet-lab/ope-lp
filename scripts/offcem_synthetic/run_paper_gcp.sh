#!/usr/bin/env bash
# Paper setup at full scale on the GCP VM (offcem-runner, n2-highmem-16). World = the final2 runs: 1000 actions, 8000 rounds, 200 users, 50 DGP clusters, seed 7000.
# Pipeline: lambda = 100, OffCEM two-step model f = g(x,c) + h(x,a) with a RIDGE first stage g (one ridge per cluster on the user context; --g-model ridge),
#   cross-fitted by user, clipped to the cluster's logged range, eta = rho = eps, L estimated from logs.
#   The paper's MLP first stage made the stack infeasible at full scale (a few variables with predictions near -30); the ridge first stage removes that (necessary checks pass at floor(eps_max)).
#   sweep : constraints (13)-(15) only, eps 1 2 3 4                         -> S1 (dgp), S2 (kmeans)
#   stack : base, + Lipschitz (B3c), probability beliefs PB_sev / PB_sev+ext alone and on top of Lipschitz, and + Lipschitz + dominance (B3c+B7_ord) for comparison
#           at eps = floor(eps_max) - 2, - 1, floor, eps_max calibrated from the logs of this world (small world: kmeans 7 8 9, dgp 5 6 7)
# Four processes in parallel (dgp/kmeans x sweep/stack), 4 threads each. Each LP with Lipschitz took ~4300 s per solve in final2, so the stack runs take many hours.
# Usage (on the VM, inside tmux, from the repo root or anywhere):   bash scripts/offcem_synthetic/run_paper_gcp.sh [tag] [seed] [extra scale_experiment.py flags...]
#   -> results_offcem_global/scale_<tag>_{sweep,stack}_<clustering>_seed<seed>.json, logs in results_offcem_global/logs/
# Resume after an interruption: re-run the same command; finished items are skipped. Infeasible items are recorded as infeasible (add --soft-t 1 2 4 8 16 to also find the slack).
set -euo pipefail
TAG=${1:-paper_ridgeg}; SEED=${2:-7000}; shift $(( $# > 2 ? 2 : $# )); EXTRA=("$@")
cd "$(dirname "$0")"
LOG=../../results_offcem_global/logs; mkdir -p "$LOG"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
COMMON=(--K 1000 --n 8000 --users 200 --C 50 --def-frac 0.5 --sigma 3 --seed "$SEED" --folds user --lam 100 --model two_step --g-model ridge --clip-model cluster_logged --L auto --time-limit 0)
STACK_BELIEFS=(B3c PB_sev PB_sev+ext B3c+PB_sev B3c+PB_sev+ext B3c+B7_ord)

launch () {   # name, clustering, eps flags (space separated), beliefs...
  local name=$1 m=$2 eps=$3; shift 3
  # shellcheck disable=SC2086
  nohup conda run -n offcem --no-capture-output python scale_experiment.py "${COMMON[@]}" --clusterings "$m" $eps --beliefs "$@" --tag "${TAG}_${name}_${m}" "${EXTRA[@]}" \
    > "$LOG/scale_${TAG}_${name}_${m}_seed${SEED}.log" 2>&1 &
  echo "started ${name} ${m} (pid $!)"
}
launch sweep dgp    "--eps-grid 1 2 3 4" baseline
launch sweep kmeans "--eps-grid 1 2 3 4" baseline
launch stack kmeans "--eps-below-max"   "${STACK_BELIEFS[@]}"
launch stack dgp    "--eps-below-max"   "${STACK_BELIEFS[@]}"
wait
echo "all runs finished: $(date)"
