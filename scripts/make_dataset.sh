#!/usr/bin/env bash
set -euo pipefail

# Fixed params (same as code defaults)
OUTDIR="data/processed/run_$(date +%Y%m%d_%H%M%S)"

python -m src.uavnet.make_dataset \
  --outdir "$OUTDIR" \
  --num_uavs 20 \
  --duration 50 \
  --dt 0.1 \
  --radius 0.9 \
  --seed 42

echo "Done. Outputs in: $OUTDIR"
ls -lh "$OUTDIR"
