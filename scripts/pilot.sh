#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root. GPU is selected automatically.
data_dir="${1:-/kaggle/working/packets_data}"
run_dir="${2:-/kaggle/working/packets_outputs}"
if [ ! -f "$data_dir/manifest.json" ]; then
  python -m packets_exp.cli prepare --data "$data_dir" --train-limit 1000 --valid-limit 100 --test-limit 100
fi
python -m packets_exp.cli train --stage receiver --data "$data_dir" --out "$run_dir/receiver" --steps 5000
python -m packets_exp.cli teacher --data "$data_dir" --receiver "$run_dir/receiver/best.pt" --teacher "$run_dir/teacher_expected_v2.pt" --teacher-windows 20000 --teacher-noise-draws 8
python -m packets_exp.cli train --stage sender --data "$data_dir" --out "$run_dir/sender_v2" --teacher "$run_dir/teacher_expected_v2.pt" --steps 3000
python -m packets_exp.cli evaluate --data "$data_dir" --receiver "$run_dir/receiver/best.pt" --sender "$run_dir/sender_v2/best.pt" --out "$run_dir/valid_clean_v2" --split valid --eval-limit 20
python -m packets_exp.cli evaluate --data "$data_dir" --receiver "$run_dir/receiver/best.pt" --sender "$run_dir/sender_v2/best.pt" --out "$run_dir/valid_burst_v2" --split valid --loss .03 --burst --eval-limit 20
