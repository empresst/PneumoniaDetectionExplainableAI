#!/usr/bin/env bash
# Train + evaluate + explain the main model and all ablations, then build the results table.
# Usage: bash scripts/run_all.sh [config ...]   (default: every config except smoke.yaml)
set -euo pipefail
configs=("$@")
if [ ${#configs[@]} -eq 0 ]; then
  configs=(configs/default.yaml configs/ablation_*.yaml)
fi
for cfg in "${configs[@]}"; do
  echo "=== $cfg ==="
  run_dir=$(python -c "from pneumo.config import load_config as l; c=l('$cfg'); print(c['output_dir']+'/'+c['run_name'])")
  python -m pneumo.train --config "$cfg"
  python -m pneumo.evaluate --run "$run_dir"
  python -m pneumo.explain --run "$run_dir"
done
python -m pneumo.summarize
