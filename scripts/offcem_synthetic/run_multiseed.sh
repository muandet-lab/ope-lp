#!/usr/bin/env bash
# Multi-world runs (5 worlds / seeds 7000-7004) for the robustness table and the per-world figures (S1 / S2 / S3d / S3e; real data R1 / R2 / R3a / R4a).
#   synthetic (small world 'sw', full scale 'full'): eps = floor(eps_max) x {1/2, 3/4, 1} of THAT world (--eps-floor-fracs); S1 / S2 sweeps at eps = 1, 2, 3, 4.
#   real data (EUR-Lex 4K, Wiki10-31K; K-means): eps = eps_max x {1/2, 3/4, 1} (eps_max < 1 there).
#   eta = eps everywhere. No slack: an infeasible item is recorded as infeasible.
#   Per world: main (base + beliefs), noband ((13)+(14): level band off, same eps grid), range ((13) only); synthetic also sweep (base, eps 1-4).
#   seed = world (synthetic: contexts, q, clusters, logs) / user pool, per-action noise, logs, rewards (real data, same dataset).
#   Full scale uses one b per (user, cluster) (--cells user): same bounds as per round, checked at seed 7000 (base / PB / Lipschitz identical to 4 decimals,
#   36-80x faster); about 30 min per world for both clusterings.
# Usage (from scripts/offcem_synthetic):  bash run_multiseed.sh [sw|real|full|all] [parallel jobs, default 4]     ('all' = sw + real; run 'full' on the VM)
#   -> ../../results_offcem_global/multiseed/scale_ms_<arm>_<kind>_s<seed>_seed<seed>.json, logs next to them.
set -euo pipefail
WHAT=${1:-all}; JOBS=${2:-4}
cd "$(dirname "$0")"
OUT=../../results_offcem_global/multiseed; mkdir -p "$OUT"
SEEDS=(7000 7001 7002 7003 7004)
SW=(--K 200 --n 1000 --users 100 --C 10 --def-frac 0.5 --sigma 3 --folds user --lam 100 --model two_step --g-model ridge --clip-model cluster_logged --L auto --time-limit 0 --clusterings dgp kmeans)
FS=(--K 1000 --n 8000 --users 200 --C 50 --def-frac 0.5 --sigma 3 --folds user --lam 100 --model two_step --g-model ridge --clip-model cluster_logged --L auto --time-limit 0 --clusterings dgp kmeans --cells user)
RD=(--data-dir ../../data --logger true_q --beta 8 --eps-target 0.1 --n 8000 --users 300 --C 100 --lam 100 --folds user --model two_step --g-model ridge --alpha cv --clip-model outcome_cell --binomial-refs --L auto --eps-min 0.02 --time-limit 0 --clusterings kmeans)
SYN_GRID=(--eps-floor-fracs 0.5 0.75 1)      # not in the range / sweep runs: a grid rule would override their --eps-grid
REAL_GRID=(--eps-fracs 0.5 0.75 1)
SYN_BELIEFS=(B3c PB_sev PB_sev+ext B3c+PB_sev B3c+PB_sev+ext)
REAL_BELIEFS=(B3c PB_fail0.3 PB_fail0.4 B3c+PB_fail0.3 B3c+PB_fail0.4)
CMDS=$(mktemp)
syn () {   # arm, base args
  local arm=$1; shift; local args=("$@")
  echo "python scale_experiment.py ${args[*]} ${SYN_GRID[*]} --seed $s --beliefs ${SYN_BELIEFS[*]} --tag ms_${arm}_main_s$s" >> "$CMDS"
  echo "python scale_experiment.py ${args[*]} ${SYN_GRID[*]} --seed $s --beliefs baseline --eta-fixed 1e6 --tag ms_${arm}_noband_s$s" >> "$CMDS"
  echo "python scale_experiment.py ${args[*]} --seed $s --beliefs baseline --eta-fixed 1e6 --eps-grid 1e6 --tag ms_${arm}_range_s$s" >> "$CMDS"
  echo "python scale_experiment.py ${args[*]} --seed $s --beliefs baseline --eps-grid 1 2 3 4 --tag ms_${arm}_sweep_s$s" >> "$CMDS"
}
for s in "${SEEDS[@]}"; do
  if [[ $WHAT == sw || $WHAT == all ]]; then syn sw "${SW[@]}"; fi
  if [[ $WHAT == full ]]; then syn full "${FS[@]}"; fi
  if [[ $WHAT == real || $WHAT == all ]]; then
    for d in eurlex wiki; do
      echo "python scale_experiment.py --real $d ${RD[*]} ${REAL_GRID[*]} --seed $s --pb-fail-alpha 0.3 0.4 --beliefs ${REAL_BELIEFS[*]} --tag ms_${d}_main_s$s" >> "$CMDS"
      echo "python scale_experiment.py --real $d ${RD[*]} ${REAL_GRID[*]} --seed $s --beliefs baseline --eta-fixed 1e6 --tag ms_${d}_noband_s$s" >> "$CMDS"
      echo "python scale_experiment.py --real $d ${RD[*]} --seed $s --beliefs baseline --eta-fixed 1e6 --eps-grid 1e6 --tag ms_${d}_range_s$s" >> "$CMDS"
    done
  fi
done
echo "$(wc -l < "$CMDS") runs, $JOBS in parallel"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
# each line runs inside conda with its own log; scale_experiment writes into results_offcem_global, the JSONs are moved into multiseed/ at the end
xargs -P "$JOBS" -I{} bash -c 'cmd="{}"; tag=$(echo "$cmd" | sed -E "s/.*--tag ([^ ]+).*/\1/"); conda run -n offcem --no-capture-output $cmd > '"$OUT"'/$tag.log 2>&1 && echo "done $tag" || echo "FAILED $tag"' < "$CMDS"
mv ../../results_offcem_global/scale_ms_*_seed*.json "$OUT"/ 2>/dev/null || true
rm -f "$CMDS"; echo "all done: $(date)"
