#!/bin/sh
set -eu

OUT_PATH=${OUTPUT_PATH:-/output/output.csv}
ART_DIR=${ARTIFACTS_DIR:-/app/artifacts}
CANDIDATES=${CANDIDATES_PATH:-data/candidates.jsonl}

mkdir -p "$(dirname "$OUT_PATH")"

echo "[ENTRYPOINT] Running ranking pipeline..."
echo "  artifacts:  $ART_DIR"
echo "  candidates: $CANDIDATES"
echo "  output:     $OUT_PATH"

python rank.py \
  --candidates "$CANDIDATES" \
  --artifacts "$ART_DIR" \
  --out "$OUT_PATH"

echo "[ENTRYPOINT] Validating generated submission..."
python tests/validate_submission.py "$OUT_PATH"

echo "[ENTRYPOINT] Done. Submission written to $OUT_PATH"
