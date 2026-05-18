#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-conns}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/conns-cache}"
PYTHON_BIN="${PYTHON:-python3}"

COMMON_REQUIRED=(
  "trained_models/conns.pth"
  "external/rad-dino-maira-2"
  "external/BiomedVLP-CXR-BERT-specialized"
)

for path in "${COMMON_REQUIRED[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
done

run_task() {
  case "$1" in
    classification_chestxdet10)
      "$PYTHON_BIN" evaluation/evaluate_cls_ChestXDet10.py "${@:2}"
      ;;
    classification_openi)
      "$PYTHON_BIN" evaluation/evaluate_cls_openi.py "${@:2}"
      ;;
    classification_chestxray14)
      "$PYTHON_BIN" evaluation/evaluate_cls_chestxray14.py "${@:2}"
      ;;
    classification_chexpert)
      "$PYTHON_BIN" evaluation/evaluate_cls_chexpert.py "${@:2}"
      ;;
    classification_padchest_gr)
      "$PYTHON_BIN" evaluation/evaluate_cls_padchest-gr.py "${@:2}"
      ;;
    grounding_chestxdet10)
      "$PYTHON_BIN" evaluation/evaluate_gr_ChestXDet10.py "${@:2}"
      ;;
    grounding_ms_cxr)
      "$PYTHON_BIN" evaluation/evaluate_gr_ms_cxr.py "${@:2}"
      ;;
    grounding_padchest_gr)
      "$PYTHON_BIN" evaluation/evaluate_gr_padchest-gr.py "${@:2}"
      ;;
    all)
      run_task classification_chestxdet10 "${@:2}"
      run_task classification_openi "${@:2}"
      run_task classification_chestxray14 "${@:2}"
      run_task classification_chexpert "${@:2}"
      run_task classification_padchest_gr "${@:2}"
      run_task grounding_chestxdet10 "${@:2}"
      run_task grounding_ms_cxr "${@:2}"
      run_task grounding_padchest_gr "${@:2}"
      ;;
    *)
      echo "Unknown task: $1" >&2
      echo "Tasks: all, classification_chestxdet10, classification_openi, classification_chestxray14, classification_chexpert, classification_padchest_gr, grounding_chestxdet10, grounding_ms_cxr, grounding_padchest_gr" >&2
      exit 1
      ;;
  esac
}

run_task "${1:-all}" "${@:2}"
