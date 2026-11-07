#!/usr/bin/env bash
# Cross-check our scorer with the organisers' starter-kit scorer for one domain.
# Usage: scripts/official_score.sh results/runs/e0_bm25/1a_dev iota dev
set -euo pipefail
RUN_DIR=$1; DOMAIN=$2; SPLIT=${3:-dev}
KIT=third_party/RETECO/starter_kit
DATA=data/raw/reteco/track1_tempo/$DOMAIN
python "$KIT/format_checker.py" "$RUN_DIR/$DOMAIN.trec" --qrels "$DATA/qrels_$SPLIT.txt" --corpus "$DATA/documents.jsonl"
python "$KIT/scorer.py" --track 1 --run "$RUN_DIR/$DOMAIN.trec" --qrels "$DATA/qrels_$SPLIT.txt" --steps "$DATA/steps_$SPLIT.jsonl"
