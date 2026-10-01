#!/bin/sh
# Re-creates every baseline in raw/. Run from the project root:
#
#     sh results/baseline_2026-09-28/run_all.sh
#
# Takes about 15 minutes with cached features. The RUL comparisons save 6 model
# files to models/ (rul_*_<date>_<time>.joblib); delete them afterwards if you
# don't need them. The other models are written to a temporary folder.

set -e
OUT=results/baseline_2026-09-28/raw
TMP=$(mktemp -d)
mkdir -p "$OUT"

python results/baseline_2026-09-28/constant_guess.py        > "$OUT/rul_constant_guess.txt" 2>&1
python -m tools.rul compare --protocol XJTU_TO_PHM               > "$OUT/rul_xjtu_to_phm.txt" 2>&1
python -m tools.rul compare --protocol PHM_LEARNING_TO_FULL_TEST > "$OUT/rul_phm_learning_to_full_test.txt" 2>&1

python -c "
import json
from predictive.dataset_jobs import run_job
print(json.dumps(run_job('mechanical', {}), indent=1, default=str))
" > "$OUT/fault_validation.json" 2>&1

python -m tools.mechanical_anomaly train --model "$TMP/mechanical_anomaly.joblib" > "$OUT/mechanical_anomaly.txt" 2>&1
cp "$TMP/mechanical_anomaly_report.json" "$OUT/"

python -m tools.partner_data template --out "$TMP/partner_example" --mode fault_labels > "$OUT/partner_synthetic.txt" 2>&1
python -m tools.partner_data validate --root "$TMP/partner_example"                  >> "$OUT/partner_synthetic.txt" 2>&1
python -m tools.partner_data train --root "$TMP/partner_example" --model "$TMP/partner.joblib" >> "$OUT/partner_synthetic.txt" 2>&1

rm -rf "$TMP"
echo "Done. Results in $OUT"
