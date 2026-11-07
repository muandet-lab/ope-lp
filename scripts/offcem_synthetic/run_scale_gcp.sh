#!/usr/bin/env bash
# OffCEM-scale run of scale_experiment.py on the GCP VM (offcem-runner, n2-highmem-16). One process per clustering so the three run in parallel.
# Usage (on the VM, inside tmux):   bash run_scale_gcp.sh [tag] [seed]        -> logs in ../../results_offcem_global/logs/, JSON scale_<tag>_seed<seed>.json
# Resume after an interruption: re-run the same command; finished items are skipped.
set -euo pipefail
TAG=${1:-offcem_default}; SEED=${2:-7000}
cd "$(dirname "$0")"
mkdir -p ../../results_offcem_global/logs
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
for M in kmeans agglomerative spectral; do
  nohup conda run -n offcem --no-capture-output python scale_experiment.py --K 1000 --n 3000 --users 1000 --C 50 --seed "$SEED" --clusterings "$M" --tag "${TAG}_${M}" \
    > "../../results_offcem_global/logs/scale_${TAG}_${M}_seed${SEED}.log" 2>&1 &
  echo "started $M (pid $!)"
done
wait
echo "all clusterings finished: $(date)"
