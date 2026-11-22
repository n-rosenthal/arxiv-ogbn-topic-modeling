#!/usr/bin/env bash
# Full ogbn-arxiv pipeline. Run from the repository root (the folder that contains src/).
# Every step is resumable: if it stops (crash, Ctrl-C, reboot), just run it again.
#
#   K=50 SEEDS="0 1 2 3 4" nohup bash src/Ogbn/run_all.sh > /dev/null 2>&1 &     # overnight, in the background
#   tail -f logs/run_all_*.log                                                     # follow progress
#
# Loop order is seed-major: after seed 0 finishes you already have ALL models for one seed,
# so an interrupted night still yields a complete (if thinner) comparison.
#
# Title-only experiment (scarce text), reusing the raw files already downloaded:
#   ROOT=data/ogbn_title TEXT_MODE=title RAW_DIR=data/ogbn/raw SEEDS="0 1 2" MODELS="prodlda ctm ctm_knn ctm_both ctm_rand" \
#     nohup bash src/Ogbn/run_all.sh > /dev/null 2>&1 &
# Sensitivity (full text; each of these is a separate night):
#   K=20  SEEDS="0 1 2" MODELS="ctm ctm_knn ctm_both" nohup bash src/Ogbn/run_all.sh > /dev/null 2>&1 &
#   K=100 SEEDS="0 1 2" MODELS="ctm ctm_knn ctm_both" nohup bash src/Ogbn/run_all.sh > /dev/null 2>&1 &
#   NBR_K=5  TAG=k5  SEEDS="0 1 2" MODELS="ctm_knn" nohup bash src/Ogbn/run_all.sh > /dev/null 2>&1 &
#   NBR_K=20 TAG=k20 SEEDS="0 1 2" MODELS="ctm_knn" nohup bash src/Ogbn/run_all.sh > /dev/null 2>&1 &
#   python -m src.Ogbn.ablation --root data/ogbn --emb minilm
# Controls on the full text (random neighbours, embedding-only encoder):
#   MODELS="ctm_rand zeroshot zeroshot_both" SEEDS="0 1 2" nohup bash src/Ogbn/run_all.sh > /dev/null 2>&1 &
set -euo pipefail
ROOT=${ROOT:-data/ogbn}
EMB=${EMB:-minilm}
K=${K:-50}
SEEDS=${SEEDS:-"0 1 2 3 4"}
MAXN=${MAXN:-}                       # e.g. MAXN=60000 to subsample the corpus (CPU budget / smoke test)
TEXT_MODE=${TEXT_MODE:-full}         # full = title + abstract | title = title only
RAW_DIR=${RAW_DIR:-}                 # reuse the raw files of another root instead of downloading
MODELS=${MODELS:-"lda nmf prodlda ctm ctm_knn ctm_cite ctm_both"}
NBR_K=${NBR_K:-10}                   # kNN neighbours; if != 10 pass TAG=k<NBR_K> so the runs are kept apart
TAG=${TAG:-}

mkdir -p logs
exec > >(tee -a "logs/run_all_$(date +%F_%H%M).log") 2>&1
echo "== run_all: ROOT=$ROOT EMB=$EMB K=$K SEEDS=[$SEEDS] MAXN=${MAXN:-all} TEXT_MODE=$TEXT_MODE MODELS=[$MODELS]"

if [ -z "$RAW_DIR" ]; then
  python -m src.Ogbn.download    --root "$ROOT"
fi
python -m src.Ogbn.corpus        --root "$ROOT" --text-mode "$TEXT_MODE" ${RAW_DIR:+--raw-dir $RAW_DIR} ${MAXN:+--max-nodes $MAXN}
python -m src.Ogbn.text_features --root "$ROOT" --min-df 20 --max-df 0.3 --max-features 5000
python -m src.Ogbn.embeddings    --root "$ROOT" --model "$EMB"

for S in $SEEDS; do
  for M in $MODELS; do
    python -m src.Ogbn.run_experiment --root "$ROOT" --model "$M" --topics "$K" --seeds "$S" --emb "$EMB" --k "$NBR_K" ${TAG:+--tag $TAG}
  done
  python -m src.Ogbn.aggregate --root "$ROOT" --emb "$EMB" --topics "$K" --ref ctm ${TAG:+--tag $TAG} || true   # partial table after each seed
done
echo "== done"
