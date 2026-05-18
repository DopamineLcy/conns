import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from conns.model import CoNNSModel


def repo_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def require_path(path, label):
    path = repo_path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path}")
    return path


def add_common_args(parser):
    parser.add_argument("--checkpoint", default="trained_models/conns.pth")
    parser.add_argument("--vision-model", default="external/rad-dino-maira-2")
    parser.add_argument("--text-model", default="external/BiomedVLP-CXR-BERT-specialized")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def model_args():
    class Args:
        world_size = 1
        rank = 0
        init_logit_scale = np.log(10)
        init_logit_bias = -5.0
        use_vision_cls_token = True
        attn_temperature = None
        proj_dim = 768
        num_hidden_layers = 2
        vision_model_path = "external/rad-dino-maira-2"
        text_model_path = "external/BiomedVLP-CXR-BERT-specialized"
        nli_model_path = "cross-encoder/nli-deberta-v3-small"
        load_nli_model = False

    return Args()


def clean_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        if key.startswith("_orig_mod."):
            key = key[10:]
        if key.startswith("criterion.nli_model."):
            continue
        cleaned[key] = value
    return cleaned


def load_model_and_processors(args):
    checkpoint_path = require_path(args.checkpoint, "checkpoint")
    vision_model = require_path(args.vision_model, "vision model")
    text_model = require_path(args.text_model, "text model")

    margs = model_args()
    margs.vision_model_path = str(vision_model)
    margs.text_model_path = str(text_model)
    margs.device = args.device
    model = CoNNSModel(args=margs)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    msg = model.load_state_dict(clean_state_dict(state_dict), strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Load status: {msg}")

    device = torch.device(args.device)
    model.to(device)
    model.eval()

    image_processor = AutoImageProcessor.from_pretrained(str(vision_model), trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(str(text_model), trust_remote_code=True)
    return model, image_processor, tokenizer, device


def tokenize(tokenizer, prompts, max_length=128):
    return tokenizer(prompts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")


def ensure_output_dir(path):
    path = repo_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, obj):
    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def autocast_context(device):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.amp.autocast(device_type="cpu", enabled=False)


def print_multilabel_metrics(targets, preds, class_names, title):
    from sklearn.metrics import roc_auc_score

    print(f"\n{title}")
    print("-" * 60)
    auc_scores = []
    for i, name in enumerate(class_names):
        if len(np.unique(targets[:, i])) < 2:
            auc = np.nan
            print(f"{name:<24} AUROC: NaN (single class in targets)")
        else:
            auc = roc_auc_score(targets[:, i], preds[:, i])
            print(f"{name:<24} AUROC: {auc:.4f}")
        auc_scores.append(auc)

    print("-" * 60)
    mean_auc = np.nan if np.isnan(auc_scores).all() else np.nanmean(auc_scores)
    print(f"Mean AUROC: {mean_auc:.4f}")


def base_parser(description):
    parser = argparse.ArgumentParser(description=description)
    return add_common_args(parser)
