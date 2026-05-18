#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-conns}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/conns-cache}"
PYTHON_BIN="${PYTHON:-python3}"
TORCHRUN_BIN="${TORCHRUN:-torchrun}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  "$PYTHON_BIN" main_pretrain.py "$@"
  exit 0
fi

NPROC_PER_NODE=1
MASTER_PORT=12345
NUM_WORKERS="${NUM_WORKERS:-10}"
BATCH_SIZE=48
EPOCHS="${EPOCHS:-100}"

required=(
  "data/raw_dataset/MIMIC-CXR-JPG/files"
  "data/conns_training/reports_extract_concepts"
  "data/conns_training/mimic_conns_training.csv"
  "data/conns_training/concepts.json"
  "data/conns_training/yes_expressions"
  "data/conns_training/no_expressions"
  "external/rad-dino-maira-2"
  "external/BiomedVLP-CXR-BERT-specialized"
)

for path in "${required[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
done

"$TORCHRUN_BIN" --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" main_pretrain.py \
  --num_workers "${NUM_WORKERS}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --save_freq 1 \
  --init_logit_bias -10 \
  --data_path data/raw_dataset/MIMIC-CXR-JPG/files \
  --report_root data/conns_training/reports_extract_concepts \
  --metadata_csv data/conns_training/mimic_conns_training.csv \
  --concepts_path data/conns_training/concepts.json \
  --yes_expressions_dir data/conns_training/yes_expressions \
  --no_expressions_dir data/conns_training/no_expressions \
  --vision_model_path external/rad-dino-maira-2 \
  --text_model_path external/BiomedVLP-CXR-BERT-specialized \
  --nli_model_path cross-encoder/nli-deberta-v3-small \
  --is_augmentation \
  --note "conns_release" \
  --warmup_iterations 2000 \
  --use_vision_cls_token \
  --script train.sh \
  --special_class_sampling_prob 0.2 \
  --use_counterfactual \
  --proj_dim 768 \
  --num_hidden_layers 2 \
  --aug_degrees 20 \
  --aug_shear 0 \
  --aug_scale 0.8 1.0 \
  --all_view \
  "$@"
