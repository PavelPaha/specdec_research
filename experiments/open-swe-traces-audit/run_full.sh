#!/usr/bin/env bash
# Full v1.2 pass on a many-core Linux server. Run from the work dir (e.g. ~/specdec-open-swe):
#   nohup bash scripts/run_full.sh > logs/run_full.log 2>&1 &
set -euo pipefail
PY=".venv/bin/python -I"
S=scripts
D=data/open-swe-traces
TOK=$D/tokenizer/qwen3.8-27b@1d4bf0f2ff
W=${WORKERS:-76}
mapfile -t FILES < <(find "$PWD/$D/shards/data" -path '*qwen38_27b*' -name '*.parquet' | sort)
echo "$(date -Is) files: ${#FILES[@]}"
[ "${#FILES[@]}" -eq 100 ] || { echo "expected 100 parquet files"; exit 1; }
mkdir -p $D/audit/full
echo "$(date -Is) audit.py"
$PY $S/audit.py "${FILES[@]}" --tok $TOK --out $D/audit/full --workers $W --chunk 500
echo "$(date -Is) reasoning_share.py"
$PY $S/reasoning_share.py $D/audit/full/calls.csv.gz $TOK $D/audit/full/reasoning_share.csv "${FILES[@]}" | tail -5
echo "$(date -Is) signals.py"
$PY $S/signals.py "$PWD/$D/shards" $D/audit/full/signals.csv $W
echo "$(date -Is) scan_strings.py"
$PY $S/scan_strings.py "$PWD/$D/shards" $D/audit/full/scan_strings.json $W
echo "$(date -Is) DONE"
